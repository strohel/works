cimport cython

cimport numpy as np
cimport pybayes.wrappers._numpy as nw

from asim.dispmodel.iface cimport DispersionModel, Location, Nuclide, SourceModel
from asim.support cimport mathwrapper as math


# Workaround: Cython doesn't allow double[:] inside @cython.locals()
ctypedef double[:] double_1D

cdef double[:] GLq_weights
cdef double[:] GLq_absicas

cdef class Puff(DispersionModel):
    cdef public int time_step, time
    cdef public Nuclide[:] inventory
    cdef public double[:] Q, total_rad_decay, total_dry_depo
    cdef public Location loc
    cdef public double total_distance, sigma_xy, sigma_z, mix_layer_height
    cdef public double[:] _ret

    @cython.locals(nuclide=Nuclide)
    cpdef double[:] unit_dose_at(self, Location loc)

    @cython.locals(nuclide=Nuclide)
    cdef double _rad_decay(self, int i)
    @cython.locals(nuclide=Nuclide)
    cdef double _dry_depo(self, int i)
    @cython.locals(factor0=double, factor1=double, factor2=double, factor3=double)
    cdef double _unit_concentration_at(self, double x, double y, double z) nogil
    @cython.locals(H=double, sz=double, M=double)
    cdef double _reflection_at_mix_layer(self, double z) nogil
    @cython.locals(H=double, sz=double)
    cdef double _reflection_at_ground(self, double z) nogil
    @cython.locals(nuclide=Nuclide, n=int, a1=double, b1=double, a2=double, b2=double,
                   a3=double, b3=double, ret1=double, i=int, ret2=double, j=int,
                   ret3=double, k=int, r=double, th=double, phi=double)
    cdef double _gauss_legendre_quadrature(self, Location loc, int l, Nuclide nuclide) nogil
    @cython.locals(x=double, y=double, z=double, C=double, B=double, ret=double)
    cdef double _gamma_flux_rate_arg(self, double r, double th, double phi, Location base,
                                     int l, Nuclide nuclide) nogil

    @cython.locals(ret=Puff)
    cpdef Puff __deepcopy__(self, memo)


cdef class PuffModel(DispersionModel):
    cdef public int time, time_step, puff_sampling_step
    cdef public list puffs
    cdef public double[:] _ret

    @cython.locals(puff=Puff)
    cpdef double[:] concentration_at(self, Location loc)
    @cython.locals(puff=Puff)
    cpdef double[:] dose_at(self, Location loc)
    @cython.locals(puff=Puff)
    cpdef double[:, :] unit_puff_dose_at(self, Location loc, double[:, :] out = *)
    @cython.locals(puff=Puff)
    cpdef double[:] deposition_at(self, Location loc)
    @cython.locals(puff=Puff)
    cpdef double[:, :] puff_activities(self, double[:, :] out = *)

    @cython.locals(puff=Puff, curr_activity=double_1D)
    cdef bint _create_kill_puffs(self, SourceModel source_model) except False
