cimport cython

cimport pybayes.wrappers._numpy as nw

from asim.dispmodel.iface cimport Location, Nuclide, SourceModel, MeteoModel, DispersionModel
from asim.dispmodel.puffmodel cimport PuffModel


cdef class StaticCompoLocSourceModel(SourceModel):
    pass


cdef class SimulatedSourceModel(StaticCompoLocSourceModel):
    cdef public int puff_sampling_step
    cdef public double[:, :] rates


cdef class PasquillsMeteoModel(MeteoModel):
    cdef public int stability_category
    cdef public double[:] height_per_category


cdef class StaticMeteoModel(PasquillsMeteoModel):
    cdef public double[:, :, :, :] METEO
    cdef public int grid_step_xy, grid_step_time

ctypedef Location[:] Location_1D
ctypedef double[:] double_1D
ctypedef double[:, :, :] double_3D

@cython.locals(simulation_length=int, time_step=int, puff_sampling_step=int, actibivies=list,
               puff_model=PuffModel, dose=double, i=int, j=int, time_steps=int, puff_count=int, clock_start=double, time_start=double,
               index=int, loc=Location, receptors=Location_1D, trajectories=double_3D,
               total_doses=double_1D, total_concentrations=double_1D, dose_this_round=double)
cpdef run(bint show_plot, bint create_video)
