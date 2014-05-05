cimport cython

from asim.support cimport mathwrapper as math


@cython.final
cdef class Location(object):
    cdef public double x, y, z

    cpdef double distance_to(self, Location other) nogil except -1.
    @cython.locals(ret = Location)
    cpdef Location __copy__(self)


@cython.final
cdef class Nuclide(object):
    cdef public double dose_mju, dose_mjue, dose_a, dose_b, dose_e, dose_cb, half_life, lambda_, depo_vg


cdef class SourceModel(object):
    cpdef Nuclide[:] inventory(self)
    cpdef double[:] release_rate(self, int time)
    cpdef Location location(self)


cdef class MeteoModel(object):
    cpdef double mixing_layer_height_at(self, Location loc, int time) except -1.
    cpdef double wind_speed_at(self, Location loc, int time) except -1.
    cpdef double wind_direction_at(self, Location loc, int time) except 7.
    cpdef double dispersion_xy(self, Location loc, int time, double total_distance) except -1.
    cpdef double dispersion_z(self, Location loc, int time, double total_distance) except -1.


cdef class DispersionModel(object):
    cpdef int propagate(self, MeteoModel meteo_model, SourceModel source_model) except -1
    cpdef double[:] concentration_at(self, Location loc)
    cpdef double[:] dose_at(self, Location loc)
    cpdef double[:] deposition_at(self, Location loc)
