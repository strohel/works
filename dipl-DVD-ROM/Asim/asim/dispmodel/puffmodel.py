"""Implementation of the dispersion model based on the concept of puffs."""

import numpy as np
import pybayes.wrappers._numpy as nw

from copy import deepcopy

from asim.dispmodel.iface import DispersionModel
from asim.support import mathwrapper as math


# initialised here so that Puff._gauss_legendre_quadrature() doesn't recreate it all the time
GLq_weights = np.array([0.236926885056, 0.478628670499, 0.568888888889, 0.478628670499, 0.236926885056])
GLq_absicas = np.array([-0.906179845939, -0.538469310106, 0.0, 0.538469310106, 0.906179845939])

class Puff(DispersionModel):
    """One puff. Makes sense only as a part of :class:`PuffModel`."""

    def __init__(self, time_step, inventory, Q, loc, time):
        """
        :param int time_step: number of seconds the integration step lasts.
            :meth:`~asim.dispmodel.iface.DispersionModel.propagate` advances *time_step* seconds
        :param inventory: radionuclides present in this release
        :type inventory: array of :class:`~asim.dispmodel.iface.Nuclide` objects
        :param Q: per-nuclide initial activities in Bacquerels
        :type Q: array of :obj:`floats <float>`
        :param loc: initial location of the puff; Puff references it (no copy)
        :type loc: :class:`~asim.dispmodel.iface.Location`
        :param int time: total time in seconds since start of the simulation
        """
        self.time_step = time_step
        self.inventory = inventory
        assert Q.shape == inventory.shape
        self.Q = Q
        self.loc = loc
        self.time = time

        self.total_distance = 0.  # total flewn distance in meters
        self.sigma_xy = 0.
        self.sigma_z = 0.
        self.mix_layer_height = 0.
        self.total_rad_decay = np.ones((inventory.shape[0],))
        self.total_dry_depo = np.ones((inventory.shape[0],))
        self._ret = nw.vector(inventory.shape[0])  # for concentration_at(), dose_at()...

    def propagate(self, meteo_model, source_model):
        wd = meteo_model.wind_direction_at(self.loc, self.time)
        ws = meteo_model.wind_speed_at(self.loc, self.time)

        vx = math.sin(wd) * ws  # x-velocity of the wind
        vy = math.cos(wd) * ws  # y-velocity of the wind

        self.loc.x += vx * self.time_step
        self.loc.y += vy * self.time_step

        self.total_distance += ws * self.time_step
        self.time += self.time_step

        for i in range(self.inventory.shape[0]):
            # account for loses due to dry deposition, uses self.sigma_z and self.loc internally:
            self.total_dry_depo[i] *= self._dry_depo(i)
            # account for loses due to radioactive decay
            self.total_rad_decay[i] *= self._rad_decay(i)

        self.sigma_xy = meteo_model.dispersion_xy(self.loc, self.time, self.total_distance)
        self.sigma_z = meteo_model.dispersion_z(self.loc, self.time, self.total_distance)
        self.mix_layer_height = meteo_model.mixing_layer_height_at(self.loc, self.time)

    def concentration_at(self, loc):
        self._ret[:] = self.Q
        # self._ret *= self.total_rad_decay:
        nw.multiply_vv(self._ret, self.total_rad_decay, self._ret)
        # self._ret *= self.total_dry_depo:
        nw.multiply_vv(self._ret, self.total_dry_depo, self._ret)
        # self._ret *= self._unit_concentration_at(...):
        nw.multiply_vs(self._ret, self._unit_concentration_at(loc.x, loc.y, loc.z), self._ret)
        return self._ret

    def dose_at(self, loc):
        """Computes dose at location *loc* using m/mju method"""
        return nw.multiply_vv(self.unit_dose_at(loc), self.Q, self._ret)

    def unit_dose_at(self, loc):
        """Same as :meth:`dose_at`, but assume unit (1 Bacquerel) Puff activity"""
        d = self.loc.distance_to(loc)
        m = 15.  # m for m/mju

        for i in range(self.inventory.shape[0]):
            nuclide = self.inventory[i]
            if d < (4.*self.sigma_xy + m/nuclide.dose_mju):
                gamma_flux_rate = self._gauss_legendre_quadrature(loc, i, nuclide)
                self._ret[i] = gamma_flux_rate * nuclide.dose_mjue * nuclide.dose_cb / 1.225 * self.time_step
            else:
                self._ret[i] = 0.  # we don't reset this elsewhere, must be here
        return self._ret

    def deposition_at(self, loc):
        raise NotImplementedError()

    def _rad_decay(self, i):
        """Return a coefficient that can be used to account for radioactive decay
        loses of the i-th radionuclide during current time_step"""
        nuclide = self.inventory[i]  # work-around cython bug
        return math.exp(-nuclide.lambda_ * self.time_step)

    def _dry_depo(self, i):
        """Return a coefficient that can be used to account for dry deposition loses
        of the i-th radionuclide during current time_step"""
        nuclide = self.inventory[i]  # work-around cython bug
        vG = nuclide.depo_vg
        H = self.loc.z
        ret = 1.0
        if (self.sigma_z > 0):
            ret -= math.sqrt(2.0/math.pi) * vG/self.sigma_z * self.time_step * math.exp(-(H**2)/(2.0*self.sigma_z**2))
        return ret

    def _unit_concentration_at(self, x, y, z):
        """Same as :meth:`concentration_at`, but assume unitary activity (1 Bacquerel).
        Does not account for radioactive decay or dry deposition."""
        factor0 = (2*math.pi)**1.5 * self.sigma_z * self.sigma_xy**2
        factor1 = math.exp(- (self.loc.x - x)**2 / (2.0 * self.sigma_xy**2))
        factor2 = math.exp(- (self.loc.y - y)**2 / (2.0 * self.sigma_xy**2))
        factor3 = self._reflection_at_mix_layer(z) + self._reflection_at_ground(z)
        return 1.0 / factor0 * factor1 * factor2 * factor3

    def _reflection_at_mix_layer(self, z):
        """Return vertical reflection at the top of the mixing layer.

        :param float z: height where the reflection should be computed
        """
        H = self.loc.z
        sz = self.sigma_z
        M = self.mix_layer_height
        return math.exp(-(2*M - z + H)**2/(2*sz**2)) + math.exp(-(2*M+ z+ H)**2/(2*sz**2)) + math.exp(-(2*M- z - H)**2/(2*sz**2)) + math.exp(-(2*M+ z - H)**2/(2*sz**2))

    def _reflection_at_ground(self, z):
        """Return reflection from the ground, account for exponential loss

        :param float z: height where the reflection should be computed
        """
        H = self.loc.z
        sz = self.sigma_z
        return math.exp(-(z + H)**2/(2*sz**2)) + math.exp(-(z - H)**2/(2*sz**2))

    def _gauss_legendre_quadrature(self, loc, l, nuclide):
        """Compute Gamma flux rate using Gauss-Legendre quadrature in 3D for unit
        concentration

        :param loc: location of the centre of the area where to compute
        :type loc: :class:`~asim.dispmodel.iface.Location`
        :param int l: index of the nuclide into the self.inventory array
        :param nuclide: nuclide object at index l (Cython GIL work-around)
        :type nuclide: :class:`~asim.dispmodel.iface.Nuclide`
        """
        n = GLq_absicas.shape[0] # number of a abscissae

        a1 = 0.0
        b1 = 5.0 / nuclide.dose_mju # average_range_5
        a2 = 0.0
        b2 = math.pi / 2.0
        a3 = 0.0
        b3 = 2*math.pi

        ret1 = 0.
        for i in range(n):
            r = (b1-a1)/2.0 * GLq_absicas[i] + (b1+a1)/2.0  # radius
            ret2 = 0.
            for j in range(n):
                th = (b2-a2)/2.0 * GLq_absicas[j] + (b2+a2)/2.0  # theta
                ret3 = 0.
                for k in range(n):
                    phi = (b3-a3)/2.0 * GLq_absicas[k] + (b3+a3)/2.0
                    ret3 += GLq_weights[k]*self._gamma_flux_rate_arg(r, th, phi, loc, l, nuclide)
                ret2 += GLq_weights[j] * ret3
            ret1 += GLq_weights[i] * ret2

        return (b1-a1) * (b2-a2) * (b3-a3) / 8.0 * ret1

    def _gamma_flux_rate_arg(self, r, th, phi, base, l, nuclide):
        """Return integrand for the Gauss-Legendre quadrature, spheric coordinates, for
        unit concentration

        :param base: center of the spherical coord system
        :type base: :class:`~asim.dispmodel.iface.Location`
        :param float r: radius from loc
        :param float th: vertical angle from loc
        :param float phi: horizontal angle from loc
        :param int l: index of the nuclide into the self.inventory array
        :param nuclide: nuclide object at index l (Cython GIL work-around)
        :type nuclide: :class:`~asim.dispmodel.iface.Nuclide`
        """
        x = base.x + r*math.sin(th)*math.cos(phi)
        y = base.y + r*math.sin(th)*math.sin(phi)
        z = base.z + r*math.cos(th)
        C = self._unit_concentration_at(x, y, z)
        C *= self.total_rad_decay[l] * self.total_dry_depo[l]

        B = 1.0 + nuclide.dose_a * nuclide.dose_mju * math.exp(nuclide.dose_b * nuclide.dose_mju * r)  # nelinearni koeficient nakupeni
        ret = C*B*math.exp(-1.0*nuclide.dose_mju*r) / (4.0*math.pi*r**2)
        ret *= r**2*math.sin(th)  # Jacobian determinant
        return ret

    def __repr__(self):
        return "<Puff loc={0} inventory={1} Q={2} d={3} s_xy={4} s_z={5}>".format(self.loc,
               np.array(self.inventory, dtype=object), np.array(self.Q), self.total_distance,
               self.sigma_xy, self.sigma_z)

    def __str__(self):
        return self.__repr__();

    def __deepcopy__(self, memo):
        ret = type(self).__new__(type(self))  # work even for sub-classes
        ret.time_step = self.time_step
        ret.time = self.time
        ret.inventory = self.inventory  # NOTE: we don't copy, inventory is assumed to be immutable!
        ret.Q = self.Q[:]
        ret.total_rad_decay = self.total_rad_decay[:]
        ret.total_dry_depo = self.total_dry_depo[:]
        ret.loc = self.loc.__copy__()
        ret.total_distance = self.total_distance
        ret.sigma_xy = self.sigma_xy
        ret.sigma_z = self.sigma_z
        ret.mix_layer_height = self.mix_layer_height
        ret._ret = nw.vector(self._ret.shape[0])
        return ret


class PuffModel(DispersionModel):
    """Implementation of the dispersion model based on the concept of :class:`puffs <Puff>`."""

    def __init__(self, time_step, puff_sampling_step, source_model):
        """
        :param int time_step: :meth:`~asim.dispmodel.iface.DispersionModel.propagate` advances
            *time_step* seconds
        :param int puff_sampling_step: inteval between 2 puff releases; must be multiple
            of *time_step*. :class:`Source model's <asim.dispmodel.iface.SourceModel>`
            :meth:`~asim.dispmodel.iface.SourceModel.release_rate` is queried in
            *puff_sampling_step*-long intervals
        :param release_loc: location where the puffs get released
        :type release_loc: :class:`~asim.dispmodel.iface.Location`
        :param source_model: source term of the release
        :type source_model: :class:`~asim.dispmodel.iface.SourceModel`
        """
        self.time = 0
        self.time_step = time_step
        self.puff_sampling_step = puff_sampling_step
        self.puffs = []
        self._ret = nw.vector(source_model.inventory().shape[0])  # for concentration_at(), dose_at()...

    def propagate(self, meteo_model, source_model):
        if self.time % self.puff_sampling_step == 0:
            self._create_kill_puffs(source_model)
        for puff in self.puffs:
            puff.propagate(meteo_model, source_model)
        self.time += self.time_step

    def concentration_at(self, loc):
        self._ret[:] = 0.
        for puff in self.puffs:  # TODO: prange in future
            # self._ret += puff.concentration_at(...)
            nw.add_vv(self._ret, puff.concentration_at(loc), self._ret)
        return self._ret

    def dose_at(self, loc):
        self._ret[:] = 0.
        for puff in self.puffs:  # TODO: prange in future
            # self._ret += puff.dose_at(...)
            nw.add_vv(self._ret, puff.dose_at(loc), self._ret)
        return self._ret

    def unit_puff_dose_at(self, loc, out=None):
        """Similar to :meth:`dose_at`, but return per-puff doses (i.e. a matrix rather
        than an array) and assume that each puff has unitary (1 Bacquerel) activity

        :param out: optional matrix to write the results to must have shape
        (puff count, nuclide count)
        """
        if out is None:
            out = nw.matrix(len(self.puffs), self.puffs[0].Q.shape[0])
        for i, puff in enumerate(self.puffs):
            out[i, :] = puff.unit_dose_at(loc)
        return out

    def deposition_at(self, loc):
        self._ret[:] = 0.
        for puff in self.puffs:  # TODO: prange in future
            # self._ret += puff.deposition_at(...)
            nw.add_vv(self._ret, puff.deposition_at(loc), self._ret)
        return self._ret

    def puff_activities(self, out=None):
        """Return activity of each puff in the puff model in Bacquerels

        :param out: optional matrix to write the results to must have shape
        (puff count, nuclide count)
        """
        if out is None:
            out = nw.matrix(len(self.puffs), self.puffs[0].Q.shape[0])
        for i, puff in enumerate(self.puffs):
            out[i, :] = puff.Q
        return out

    def _create_kill_puffs(self, source_model):
        """
        Kill puffs sentenced to death (those too far away or with activity below
        threshold) and eventually create a new one according to the source_model.
        """
        # keep this zero as assimilation creates unit puff and only then assigns activity
        create_threshold = 0.0
        kill_threshold = 1.E12  # TODO: this is chosen arbitrarily

        # first, kill puffs:
        for i in range(len(self.puffs) -1, -1, -1):
            puff = self.puffs[i]
            if nw.sum_v(puff.Q) < kill_threshold:
                del self.puffs[i]

        curr_activity = nw.multiply_vs(source_model.release_rate(self.time), self.puff_sampling_step)
        if nw.sum_v(curr_activity) > create_threshold:
            puff = Puff(time_step=self.time_step, inventory=source_model.inventory(),
                        Q=curr_activity, loc=source_model.location(), time=self.time)
            self.puffs.append(puff)
        return True

    def __deepcopy__(self, memo):
        ret = type(self).__new__(type(self))  # work even for sub-classes
        ret.time =  self.time
        ret.time_step = self.time_step
        ret.puff_sampling_step = self.puff_sampling_step
        ret.puffs = deepcopy(self.puffs, memo)
        ret._ret = nw.vector(self._ret.shape[0])
        return ret
