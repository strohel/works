"""Module with interfaces (empty classes) for core dispersion model objects."""

from asim.support import mathwrapper as math


class Location(object):
    """Describes a point in 3D space."""

    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def distance_to(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)

    def __copy__(self):
        ret = type(self).__new__(type(self))  # work even for subclasses
        (ret.x, ret.y, ret.z) = (self.x, self.y, self.z)
        return ret

    def __deepcopy__(self, memo):
        return self.__copy__()  # not a compound object, deepcopy is same as copy

    def __repr__(self):
        return "<Loc. x={0} y={1} z={2}>".format(self.x, self.y, self.z)

    def __str__(self):
        return self.__repr__()


class Nuclide(object):
    """Nuclide describes properties of one radionuclide."""

    def __init__(self, dose_mju, dose_mjue, dose_a, dose_b, dose_e, dose_cb, half_life, depo_vg):
        self.dose_mju = dose_mju
        self.dose_mjue = dose_mjue
        self.dose_a = dose_a
        self.dose_b = dose_b
        self.dose_e = dose_e
        self.dose_cb = dose_cb
        self.half_life = half_life  #: nuclide half-life in seconds
        self.lambda_ = math.log(2) / half_life  #: decay constant :math:`\lambda = \frac{\ln 2}{\mathrm{half\_life}}`
        self.depo_vg = depo_vg

    def __repr__(self):
        return "<Nucl. half_life={0}s>".format(self.half_life)

    def __str__(self):
        return self.__repr__()

    def __copy__(self):
        raise NotImplementedError("Prevent silent failure to copy if compiled by Cython")

    def __deepcopy__(self, memo):
        raise NotImplementedError("Prevent silent failure to deepcopy if compiled by Cython")


class SourceModel(object):
    """Model of the source of an atmospheric radioactive release."""

    def inventory(self):
        """Return an array of radionuclides this release contains.

        :rtype: array of :class:`Nuclides <Nuclide>`
        """
        raise NotImplementedError("Abstract method")

    def release_rate(self, time):
        """Return release rate in Bq/s of all radionuclides at given time. Rate of
        radionuclide :meth:`inventory()[i] <inventory>` is
        :meth:`release_rate(time)[i] <release_rate>`.

        :rtype: array of :obj:`floats <float>`
        """
        raise NotImplementedError("Abstract method")

    def location(self):
        """Return location of the release. All three coordinates are to be used.

        :rtype: :class:`Location`
        """
        raise NotImplementedError("Abstract method")

    def __copy__(self):
        raise NotImplementedError("Prevent silent failure to copy if compiled by Cython")

    def __deepcopy__(self, memo):
        raise NotImplementedError("Prevent silent failure to deepcopy if compiled by Cython")


class MeteoModel(object):
    """Model of weather."""

    def mixing_layer_height_at(self, loc, time):
        """Return height of the mixing layer in meters at given location and time.

        :param loc: where the height is to be computed
        :type loc: :class:`Location`
        :param int time: time in seconds since start of the simulation
        :rtype: :obj:`float`
        """
        raise NotImplementedError("Abstract method")

    def wind_speed_at(self, loc, time):
        """Return speed of the wind in m/s at given location and time.

        :param loc: where the speed is to be computed
        :type loc: :class:`Location`
        :param int time: time in seconds since start of the simulation
        :rtype: :obj:`float`
        """
        raise NotImplementedError("Abstract method")

    def wind_direction_at(self, loc, time):
        """Return direction of the wind in radians at given location and time.

        :param loc: where the direction is to be computed
        :type loc: :class:`Location`
        :param int time: time in seconds since start of the simulation
        :rtype: :obj:`float`
        """
        raise NotImplementedError("Abstract method")

    def dispersion_xy(self, loc, time, total_distance):
        """Return horizonal dispersion coefficient for a puff that is currently at location
        *loc*, time *time* and has flewn *total_distance* meters in total.
        """
        raise NotImplementedError("Abstract method")

    def dispersion_z(self, loc, time, total_distance):
        """Return vertical dispersion coefficient for a puff that is currently at location
        *loc*, time *time* and has flewn *total_distance* meters in total.
        """
        raise NotImplementedError("Abstract method")

    def __copy__(self):
        raise NotImplementedError("Prevent silent failure to copy if compiled by Cython")

    def __deepcopy__(self, memo):
        raise NotImplementedError("Prevent silent failure to deepcopy if compiled by Cython")


class DispersionModel(object):
    """Model of propagation of radionuclides in atmosphere."""

    def propagate(self, meteo_model, source_model):
        """Propagate model to next time step.

        :param meteo_model: model of weather the calculation should use
        :type meteo_model: :class:`MeteoModel`
        :param source_model: model of the source of the release the calculation should use
        :type source_model: :class:`SourceModel`
        """
        raise NotImplementedError("Abstract method")

    def concentration_at(self, loc):
        """Compute radioactive concentration at given location (:class:`Location`) and
        model's current time. [Bq/m^3]

        :return: view on 1D array with per-nuclide concentration. You may *not* modify
            values in the view. The view may become invalid when this or other method
            is called on this instance again
        """
        raise NotImplementedError("Abstract method")

    def dose_at(self, loc):
        """Compute radioactive dose at given location (:class:`Location`) and model's
        current time in Sieverts.

        :return: view on 1D array with per-nuclide dose. You may *not* modify
            values in the view. The view may become invalid when this or other method
            is called on this instance again
        """
        raise NotImplementedError("Abstract method")

    def deposition_at(self, loc):
        """Compute combined dry and wet radioactive deposition at given location
        (:class:`Location`) and model's current time.

        :return: view on 1D array with per-nuclide deposition. You may *not* modify
            values in the view. The view may become invalid when this or other method
            is called on this instance again
        """
        raise NotImplementedError("Abstract method")

    def __copy__(self):
        raise NotImplementedError("Prevent silent failure to copy if compiled by Cython")

    def __deepcopy__(self, memo):
        raise NotImplementedError("Prevent silent failure to deepcopy if compiled by Cython")
