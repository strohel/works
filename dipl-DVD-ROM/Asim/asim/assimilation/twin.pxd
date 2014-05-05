cimport cython

cimport numpy as np
cimport pybayes
cimport pybayes.wrappers._numpy as nw

from asim.dispmodel.iface cimport Location, SourceModel
from asim.dispmodel.puffmodel cimport PuffModel
from asim.simulation.simple cimport StaticCompoLocSourceModel, SimulatedSourceModel
from asim.simulation.simple cimport PasquillsMeteoModel, StaticMeteoModel
from asim.support cimport mathwrapper as math
from asim.support.newton cimport Function, find_root

ctypedef double[:] double_1D
ctypedef double[:, :] double_2D


cdef class AssimilationSourceModel(StaticCompoLocSourceModel):
    cdef public int puff_sampling_step


cdef class AssimilationMeteoModel(PasquillsMeteoModel):
    cdef public double static_ws, static_wd, a, b

    cpdef bint set_at_bt(self, double at, double bt)


cdef class Params:
    cdef public double gamma_a, sigma_b, gamma_y, gamma_ws, sigma_wd, ws_tilde, wd_tilde
    cdef public double natural_dose_per_puff_sampling_step


cdef class PuffModelEmpPdf(pybayes.EmpPdf):
    cdef public object models  # numpy.ndarray
    cdef public AssimilationMeteoModel meteo_model
    cdef public SourceModel source_model
    cdef public Location[:] receptors
    cdef public object ax

    cdef public double[:, :] receptor_doses
    cdef public object trajectories
    cdef public double n_eff
    cdef public dict unit_receptor_doses
    cdef public Params params
    cdef public int i, last_puff

    @cython.locals(i=int)
    cpdef bint resample(self)
    @cython.locals(model=PuffModel)
    cpdef bint transition_using(self, int i, pybayes.CPdf TransitionCPdf)
    @cython.locals(doses=double_2D)
    cdef bint _compute_unit_doses(self, PuffModel model) except False
    @cython.locals(activities=double_1D, unit_doses=double_2D, puff_doses=double_1D)
    cdef bint _compute_doses(self, PuffModel model) except False


cdef class TransitionCPdf(pybayes.CPdf):
    cdef public pybayes.GammaCPdf a_cpdf
    cdef public pybayes.MLinGaussCPdf b_cpdf


cdef class ObservationCPdf(pybayes.CPdf):
    cdef public PuffModelEmpPdf emp
    cdef AssimilationMeteoModel meteo_model
    cdef public pybayes.InverseGammaCPdf dose_cpdf
    cdef public pybayes.InverseGammaCPdf ws_cpdf
    cdef public pybayes.MLinGaussCPdf wd_cpdf
    cdef public int i


cdef class LogProposalCPdfFunc(Function):
    cdef public double alpha
    cdef public double[:] beta, m, y

    @cython.locals(ret=double, beta_j_Q_plus_m_j=double)
    cdef double eval(self, double x) nogil


cdef class LogProposalCPdfDerivative(Function):
    cdef public LogProposalCPdfFunc func

    @cython.locals(ret=double)
    cdef double eval(self, double x) nogil


cdef class ActivityProposalCPdf(pybayes.CPdf):
    cdef public pybayes.TruncatedNormPdf Q_tnorm
    cdef public LogProposalCPdfFunc func
    cdef public LogProposalCPdfDerivative derivative
    cdef public PuffModelEmpPdf emp
    cdef public double last_a_t

    @cython.locals(doses=double_2D)
    cpdef double[:] sample(self, double[:] cond = *)
