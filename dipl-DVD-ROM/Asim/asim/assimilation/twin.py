#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Atmospheric radioactive release twin experiment using :mod:`asim.simulation.simple` and
:mod:`asim.dispmodel.puffmodel`"""

import matplotlib.pyplot as plt
import numpy as np
import pybayes
import pybayes.wrappers._numpy as nw
import scipy.io

import cProfile
from copy import deepcopy
import glob
from os.path import dirname, join
import pickle
import pstats
import subprocess
from time import clock

from asim.dispmodel.iface import Location
from asim.dispmodel.puffmodel import PuffModel
from asim.simulation.simple import StaticCompoLocSourceModel, SimulatedSourceModel
from asim.simulation.simple import PasquillsMeteoModel, StaticMeteoModel, read_receptor_locations
from asim.support.directories import results_dir
from asim.support import mathwrapper as math
from asim.support.newton import Function
from asim.support.plotters import plot_map, plot_stations, plot_trajectories, plot_contour


class AssimilationSourceModel(StaticCompoLocSourceModel):
    """Source model used for assimilation of released activity."""

    def __init__(self, puff_sampling_step):
        """
        :param int time_step: inverval between rate changes
        """
        self.puff_sampling_step = puff_sampling_step

    def release_rate(self, time):
        # time is ignored, it is assumed to be kept up-to date from "above"
        ret = nw.vector(1)
        # yeah, we return rate so that the sampled puff is of unitary activity
        ret[0] = 1. / self.puff_sampling_step
        return ret


class AssimilationMeteoModel(PasquillsMeteoModel):
    """Meteorologic model used for assimilation of wind speed and direction."""

    def __init__(self, stability_category, static_ws, static_wd):
        """Construct assimilation model.

        :param int stability_category: Pasquills statiblity category A=0 ... F=5
        """
        PasquillsMeteoModel.__init__(self, stability_category)
        self.static_ws = static_ws
        self.static_wd = static_wd
        self.a = 1.
        self.b = 0.

    def wind_speed_at(self, loc, time):
        return self.static_ws * self.a

    def wind_direction_at(self, loc, time):
        wd = self.static_wd + self.b
        if not (0. <= wd < 2.*math.pi):
            k = math.floor(wd / (2.*math.pi))
            wd -= k * 2.*math.pi
        return wd

    def set_at_bt(self, at, bt):
        self.a = at
        self.b = bt


class PuffModelEmpPdf(pybayes.EmpPdf):

    def __init__(self, init_particles, time_step, puff_sampling_step, source_model,
                 meteo_model, receptors, params, ax = None):
        """"
        :param ax: MatPlotLib axis to draw particle trajectories to or None
        """
        pybayes.EmpPdf.__init__(self, init_particles)
        n = init_particles.shape[0]  # number of particles
        self.models = np.empty(shape=(n,), dtype=PuffModel)
        for i in range(n):
            self.models[i] = PuffModel(time_step, puff_sampling_step, source_model)

        self.source_model = source_model
        self.meteo_model = meteo_model
        self.receptors = receptors
        self.ax = ax

        puff_count = 6  # TODO
        self.receptor_doses = nw.matrix(init_particles.shape[0], receptors.shape[0])
        self.trajectories = np.zeros((2, n * puff_count, 2))  # time steps, total puffs, x and y
        self.n_eff = 0.
        self.unit_receptor_doses = {}
        self.params = params
        self.i = 0  # index of the model currently worked on. Needed by various external objects
        self.last_puff = -1

    def resample(self):
        """Drop low-weight particles, replace them with copies of more weighted
        ones. Also reset weights to uniform."""
        self.n_eff = 1. / nw.sum_v(nw.power_vs(self.weights, 2.))

        # this is a good place to plot trajectories, because we have the weights
        # normalised, but not yet resampled
        if self.ax:
            threshold = 1. / self.weights.shape[0] * 0.99  # at least 99% of uniform weight
            alphas = np.array(self.weights)
            selected = alphas >= threshold
            alphas = alphas[selected]  # filter out weights below threshold

            wantmin, wantmax = 0.01, 1./math.sqrt(len(alphas))
            alphas = alphas*(wantmax - wantmin)/alphas.max() + wantmin
            colors = np.tile(["b", "g", "r", "c", "m", "y"], len(alphas))  # TODO: different puff count
            alphas = np.repeat(alphas, 6)  # TODO: don't hard-code puff count
            selected = np.repeat(selected, 6)
            plot_trajectories(self.ax, np.array(self.trajectories)[:, selected, :], alphas, colors)
            self.trajectories[:, :, :] = 0.  # reset

        resample_indices = self.get_resample_indices()
        nw.reindex_mv(self.particles, resample_indices)
        self.models = self.models[np.asarray(resample_indices)]  # makes references, not hard copies
        for i in range(self.models.shape[0]):
            if resample_indices[i] == i:  # copy only when needed
                continue
            self.models[i] = deepcopy(self.models[i])  # we need to deep copy ith PuffModel

        self.weights[:] = 1./self.weights.shape[0]
        return True

    def transition_using(self, i, transition_cpdf):
        model = self.models[i]

        # record start points for trajectory plotting
        puff_count = self.trajectories.shape[1] / self.particles.shape[0]
        for j in range(min(puff_count, len(model.puffs))):
            loc = model.puffs[j].loc
            self.trajectories[0, i*puff_count + j, 0] = loc.x
            self.trajectories[0, i*puff_count + j, 1] = loc.y

        self._reset_unit_doses(i)
        self.particles[i] = transition_cpdf.sample(self.particles[i])  # sample new a_t, b_t, Q_t
        if self.last_puff >= 0:  # if we need to update activity of the last sampled puff
            model.puffs[self.last_puff].Q[0] = self.particles[i, 2]  # set the real last puff activity
            #print "Model {0:3d}, setting dose of puff {1:2d} to {2:.2e}".format(i, self.last_puff, self.particles[i, 2])
        else:
            # the following is just that naive proposal that doesn't assimilate Q_t at all
            # shows the correct Q_t:
            self.particles[i, 2] = nw.sum_v(self.source_model.release_rate(model.time)) * model.puff_sampling_step
        self.meteo_model.set_at_bt(self.particles[i, 0], self.particles[i, 1])
        self._compute_doses(model)

        # record end points for trajectory plotting
        for j in range(min(puff_count, len(model.puffs))):
            loc = model.puffs[j].loc
            self.trajectories[1, i*puff_count + j, 0] = loc.x  # let all trajectories start at 0
            self.trajectories[1, i*puff_count + j, 1] = loc.y
        return True

    def _compute_unit_doses(self, model):
        """Compute self.unit_receptor_doses[self.i][:, :] if it has not been already computed."""
        if self.i in self.unit_receptor_doses:
            return True  # already computed
        doses = None
        for j in range(model.puff_sampling_step / model.time_step):
            model.propagate(self.meteo_model, self.source_model)
            # we postpone creating unit_receptor_doses array for this model to the place
            # where the potential new puff was already created
            if doses is None:
                doses = nw.matrix(self.receptors.shape[0], len(model.puffs))
                doses[:, :] = 0.
                self.unit_receptor_doses[self.i] = doses

            # normally, the doses should be computed in ObservationCPdf, but as we do
            # integration in time while propagating the model, it woudn't have enough
            # information, so we do it here.
            for k in range(self.receptors.shape[0]):
                loc = self.receptors[k]
                # doses[k, :] += (sum per all nuclides of) model.unit_puff_dose_at(loc)
                nw.add_vv(doses[k, :], nw.rowwise_sum(model.unit_puff_dose_at(loc)), doses[k, :])
        return True

    def _compute_doses(self, model):
        """Compute self.receptor_doses[self.i], computing unit doses along the way"""
        self._compute_unit_doses(model)
        activities = nw.rowwise_sum(model.puff_activities())
        unit_doses = self.unit_receptor_doses[self.i]
        for k in range(self.receptors.shape[0]):
            puff_doses = nw.multiply_vv(activities, unit_doses[k, :])
            self.receptor_doses[self.i, k] = nw.sum_v(puff_doses)
            self.receptor_doses[self.i, k] += self.params.natural_dose_per_puff_sampling_step
        return True

    def _reset_unit_doses(self, i):
        """Reset unit doses of the i-th model and set current index to i"""
        if i in self.unit_receptor_doses:
            del self.unit_receptor_doses[i]
        self.i = i
        self.last_puff = -1
        return True


class TransitionCPdf(pybayes.CPdf):

    def __init__(self, params):
        self.a_cpdf = pybayes.GammaCPdf(params.gamma_a)
        self.b_cpdf = pybayes.MLinGaussCPdf(cov=np.array([[params.sigma_b**2.]]), A=np.eye(1), b=np.zeros(1))

        self._set_rvs(3, None, 3, None)

    def sample(self, cond = None):
        """Sample from this CPdf.

        :param cond: condition: previous a_t, b_t, Q_t
        """
        self._check_cond(cond)
        # sample new a_t, b_t, Q_t
        ret = nw.vector(3)
        ret[0] = self.a_cpdf.sample(cond[0:1])[0]  # old a_t
        ret[1] = self.b_cpdf.sample(cond[1:2])[0]  # old b_t
        ret[2] = 0.  # assimilating Q is infeasible with a bootstrap proposal, just fill zero
        if not (0. <= ret[1] < 2.*math.pi):
            k = math.floor(ret[1] / (2.*math.pi))
            ret[1] -= k * 2.*math.pi
        return ret

    def eval_log(self, x, cond = None):
        self._check_x(x)
        self._check_cond(cond)
        if not (0. <= x[1] < 2.*math.pi):
            k = math.floor(x[1] / (2.*math.pi))
            x[1] -= k * 2.*math.pi
        if not (0. <= cond[1] < 2.*math.pi):
            k = math.floor(cond[1] / (2.*math.pi))
            cond[1] -= k * 2.*math.pi
        ret = 0.
        ret += self.a_cpdf.eval_log(x[0:1], cond[0:1])
        ret += self.b_cpdf.eval_log(x[1:2], cond[1:2])
        # the third part assigns same value for all particles
        return ret


class ObservationCPdf(pybayes.CPdf):
    """(wind speed, wind direction, dose) is not a complete state - it needs to include
    puff locations, total flewn distances etc. Therefore ObservationCPdf references
    PuffModelEmpPdf to acces full state through its models attribute.
    """

    def __init__(self, emp, receptors, meteo_model, params):
        self.emp = emp
        self.meteo_model = meteo_model

        self.dose_cpdf = pybayes.InverseGammaCPdf(params.gamma_y)
        self.ws_cpdf = pybayes.InverseGammaCPdf(params.gamma_ws)
        self.wd_cpdf = pybayes.MLinGaussCPdf(cov=np.array([[params.sigma_wd**2.]]), A=np.eye(1), b=np.zeros(1))

        self._set_rvs(2 + receptors.shape[0], None, 3, None)

    def eval_log(self, x, cond = None):
        """Evalate cpdf in point *x* given condition *cond*.

        :param x: observation vector (wind speed, wind direction, doses measured at receptors)
        :param cond: state vector (a_t, b_t, Q_t)
        """
        self._check_x(x)
        self._check_cond(cond)
        # just check that self.emp.i is correct:
        assert(cond[0] == self.emp.particles[self.emp.i, 0] and
               cond[1] == self.emp.particles[self.emp.i, 1] and
               cond[2] == self.emp.particles[self.emp.i, 2])
        doses = self.emp.receptor_doses[self.emp.i]

        ret = 0.  # 1 is neutral element in multiplication; log(1) = 0

        self.meteo_model.set_at_bt(cond[0], cond[1])
        simulated_ws = self.meteo_model.wind_speed_at(None, 0)
        coef = self.ws_cpdf.eval_log(x[0:1], np.array([simulated_ws]))
        #print "  Wind speed  simulated: {0:.9f} measured: {1:.9f} exp(coef): {2:.1e}".format(simulated_ws, x[0], math.exp(coef))
        ret += coef
        simulated_wd = self.meteo_model.wind_direction_at(None, 0)
        coef = self.wd_cpdf.eval_log(x[1:2], np.array([simulated_wd]))
        #print "  Wind direct simulated: {0:.9f} measured: {1:.9f} exp(coef): {2:.1e}".format(simulated_wd, x[1], math.exp(coef))
        ret += coef

        # doses
        for i in range(doses.shape[0]):
            obs_i = i + 2
            coef = self.dose_cpdf.eval_log(x[obs_i:obs_i+1], np.array([doses[i]]))  # TODO: get rid of np
            #print "  Receptor {0:>2} simulated: {1:.9f} measured: {2:.9f} exp(coef): {3:.1e}".format(i+1, doses[i], x[obs_i], math.exp(coef))
            ret += coef
        #print "Model {0:3>} {1:.2f} m/s {2:.2f} rad weight *= {3:.1e}".format(self.i-1, simulated_ws, simulated_wd, math.exp(ret))
        return ret


class LogProposalCPdfFunc(Function):
    r"""Derivative of the logarithm of the optimal proposal conditional probability
    density function, i.e: :math:`log(p(Q|y_t))`

    Used to find maxima of the proposal cpdf using Newton method.
    """

    def __init__(self, alpha, receptor_count):
        self.alpha = alpha
        self.beta = nw.vector(receptor_count)
        self.m = nw.vector(receptor_count)
        self.y = None  # referenced, not copied in

    def eval(self, Q):
        ret = 0.
        for j in range(self.beta.shape[0]):
            # in theory, alpha may be different for each receptor, but it isn't
            ret += self.alpha*self.beta[j] / (self.beta[j]*Q + self.m[j])
            ret -= self.beta[j]/self.y[j]
        return ret


class LogProposalCPdfDerivative(Function):
    r"""Derivative of the logarithm of the optimal proposal conditional probability
    density function, i.e: :math:`log(p(Q|y_t))`

    Used to find maxima of the proposal cpdf using Newton method.
    """

    def __init__(self, func):
        self.func = func

    def eval(self, Q):
        ret = 0.
        for j in range(self.func.beta.shape[0]):
            # in theory, alpha may be different for each receptor, but it isn't
            ret -= self.func.alpha * self.func.beta[j]**2. / (self.func.beta[j]*Q + self.func.m[j])**2.
        return ret


class ActivityProposalCPdf(pybayes.CPdf):
    """:math:`p(Q_t|y_{1:t}, l_{1:t}) p(l_t|l_{t-1}, a_t, b_t)` where :math:`l_t` is
    deterministic given :math:`a_t, b_t` and :math:`l_{t-1}` is sourced from the model.
    The actual conditioning variable is therefore only :math:`(a_t, b_t)`"""

    def __init__(self, emp, rv, cond_rv):
        self._set_rvs(1, rv, 2, cond_rv)
        # b defaults to +infinity, mu, sigma set later:
        self.Q_tnorm = pybayes.TruncatedNormPdf(0., 1., a=0.)
        self.func = LogProposalCPdfFunc(emp.params.gamma_y**-2. + 2, len(emp.receptors))
        self.derivative = LogProposalCPdfDerivative(self.func)
        self.emp = emp

    def sample(self, cond = None):
        self._check_cond(cond)
        self.last_a_t = cond[0]  # this is just for an assert below

        self.emp.meteo_model.set_at_bt(cond[0], cond[1])
        model = self.emp.models[self.emp.i]
        self.emp._compute_doses(model)
        self.emp.last_puff = len(model.puffs) - 1
        assert model.puffs[self.emp.last_puff].Q[0] == 1.
        doses = self.emp.unit_receptor_doses[self.emp.i]
        self.func.beta[:] = doses[:, self.emp.last_puff]
        self.func.m[:] = self.emp.receptor_doses[self.emp.i, :]
        # m -= beta:
        nw.subtract_vv(self.func.m, self.func.beta, self.func.m)
        # beta *= gamma_y**-2 + 1
        nw.multiply_vs(self.func.beta, self.emp.params.gamma_y**-2. + 1, self.func.beta)
        # m *= gamma_y**-2 + 1
        nw.multiply_vs(self.func.m, self.emp.params.gamma_y**-2. + 1, self.func.m)

        self.Q_tnorm.mu = find_root(self.func, self.derivative, start=1., precision=1.E8, max_iters=40)
        #print "alfa=", self.func.alpha, "beta=", np.asarray(self.func.beta)
        #print "m=", np.asarray(self.func.m)
        #print "self.derivative.eval(", self.Q_tnorm.mu, ") =", self.derivative.eval(self.Q_tnorm.mu)
        self.Q_tnorm.sigma_sq = -1./(2. * self.derivative.eval(self.Q_tnorm.mu))
        #print "Q_t ~ {0:.2e} ± {1:.2e}".format(self.Q_tnorm.mu, math.sqrt(self.Q_tnorm.sigma_sq))

        # the following is a protection against inefficient samplig algo in Q_tnorm and
        # also round-towards zero in eval_log; it should have virtually no effect because
        # tnorm scales the pdf to integrate to one and the shape should be virtually the
        # same
        if self.Q_tnorm.mu + math.sqrt(self.Q_tnorm.sigma_sq) < 0.:
            self.Q_tnorm.mu = -math.sqrt(self.Q_tnorm.sigma_sq)
        return self.Q_tnorm.sample()

    def eval_log(self, x, cond = None):
        self._check_x(x)
        self._check_cond(cond)
        assert self.last_a_t == cond[0]
        return self.Q_tnorm.eval_log(x)

    def set_measured_doses(self, doses):
        self.func.y = doses


class CongujateProposalFilter(pybayes.Filter):

    def __init__(self, emp):
        self.emp = emp
        p = emp.params
        a_t, b_t, Q_t = pybayes.RVComp(1, 'a_t'), pybayes.RVComp(1, 'b_t'), pybayes.RVComp(1, 'Q_t')
        self.ws_gamma = pybayes.GammaPdf(1., 1., rv=a_t)  # both params set later
        self.ws_gamma.k = p.gamma_ws**(-2.) + p.gamma_a**(-2.) + 2.
        self.wd_gauss = pybayes.GaussPdf(mean=np.zeros(1), cov=np.eye(1), rv=b_t)  # both params set later
        self.wd_gauss.R[0, 0] = 1./(p.sigma_b**-2. + p.sigma_wd**-2.)
        self.Q_tnorm = ActivityProposalCPdf(emp=emp, rv=[Q_t], cond_rv=[a_t, b_t])
        self.prod = pybayes.ProdCPdf([self.ws_gamma, self.wd_gauss, self.Q_tnorm],
                                     rv=[a_t, b_t, Q_t], cond_rv=[])

    def bayes(self, yt, cond):
        # cond is x_t^(i)
        a_tp, b_tp = cond[0], cond[1]  # previous a, b
        v_t, phi_t = yt[0], yt[1]  # measured wind speed, wind direction
        p = self.emp.params

        self.ws_gamma.theta = 1./((p.gamma_ws**-2. + 1.) / v_t * p.ws_tilde + p.gamma_a**(-2.) / a_tp)
        self.wd_gauss.mu[0] = self.wd_gauss.R[0, 0] * (p.sigma_b**-2. * b_tp + p.sigma_wd**-2. * (phi_t - p.wd_tilde))
        self.Q_tnorm.set_measured_doses(yt[2:])
        # Q_tnorm doesn't need other updating, it is fed sampled a_t, b_t by ProdCPdf

        #print "Congujate for a_tp={0:.2f} b_tp={1: .2f}=>{2: .2f}:".format(a_tp, b_tp, b_tp + p.wd_tilde),
        #print "a_t = {0:.2f} ± {1:.2f} b_t = {2: .2f}=>{3: .2f} ± {4:.2f}".format(
              #self.ws_gamma.mean()[0], math.sqrt(self.ws_gamma.variance()[0]),
              #self.wd_gauss.mean()[0], self.wd_gauss.mean()[0] + p.wd_tilde, math.sqrt(self.wd_gauss.variance()[0]))
        return True

    def posterior(self):
        return self.prod


def main(show_plot=True, create_video=False, profile=False):
    if profile:
        filename = "profile_" + __name__ + ".prof"
        cProfile.runctx("run(show_plot, create_video)",
                        globals(), locals(), filename)
        s = pstats.Stats(filename)
        s.sort_stats("time")  # "time" or "cumulative"
        s.print_stats(30)
        s.print_callers(30)
    else:
        run(show_plot, create_video)


class Params(object):

    def __init__(self):
        self.gamma_a = 0.2  # article: 0.2
        self.sigma_b = 15./360. * 2.*math.pi  # article: 15°
        self.gamma_y = 0.2  # relative standard deviation for dosimeters; article: 0.2
        self.gamma_ws = 0.1  # relative standard deviation for wind speed measurement; article: 0.1
        self.sigma_wd = 5./360. * 2.*math.pi  # standard deviation for wind direction measurement; article: 5°
        self.ws_tilde = 2.1  # assume numerical model gives this m/s prediction
        self.wd_tilde = 30./360. * 2*math.pi  # assume numerical model gives this direction
        self.natural_dose_per_puff_sampling_step = 7.6103501e-11 * 60 * 10  # TODO: check


def run(show_plot, create_video):
    simulation_length = 4*60*60  # 4 hours
    time_step = 2*60  # 2 minutes
    puff_sampling_step = 5*time_step
    release_loc = Location(0., 0., 50.)
    activities = np.array([[.1E+16], [3.0E+16], [4.0E+16], [2.0E+16], [3.0E+16], [1.0E+16]])
    stability_category = 3
    n = 1000  # number of particles

    source_dir = dirname(__file__)
    simulation_dir = join(dirname(dirname(__file__)), "simulation")
    receptors_file = join(simulation_dir, "receptory_ETE")
    receptors = np.array(read_receptor_locations(receptors_file), dtype=Location)
    measured_meteo_file = join(simulation_dir, "meteo")
    measured_meteo = StaticMeteoModel(meteo_array=scipy.io.loadmat(measured_meteo_file)["METEO"],
                                      stability_category=stability_category)
    measured_doses = scipy.io.loadmat(results_dir('simulation.doses_at_locations'))["dose_of_model"]

    params = Params()

    init_a_pdf = pybayes.GammaPdf(4., 0.25)  # mean = 1, std dev = 0.5
    init_b_pdf = pybayes.GaussPdf(np.array([0.]), np.array([[(math.pi/2.)**2.]]))
    init_Q_pdf = pybayes.UniPdf(np.array([0.]), np.array([10.0E+16]))  # very wild guess
    init_pdf = pybayes.ProdPdf((init_a_pdf, init_b_pdf, init_Q_pdf))

    names, proposals, source_models, figs, axes = [], [], [], [], []
    #names.append("No dose naive"); proposals.append(None); obs_lens.append(2)
    #names.append("No dose conjugate"); proposals.append(CongujateProposalFilter(params)); obs_lens.append(2)
    names.append("a-b-naive-propo"); proposals.append(type(None));              source_models.append(SimulatedSourceModel(puff_sampling_step, activities))
    names.append("a-b-Q-conjugate"); proposals.append(CongujateProposalFilter); source_models.append(AssimilationSourceModel(puff_sampling_step))
    for name in names:
        if show_plot or create_video:
            fig_ = plt.figure(name)
            axis_ = fig_.add_subplot(1,1,1)
            plot_stations(axis_, receptors)
            plot_map(axis_)
            axis_.grid(True)

            figs.append(fig_)
            axes.append(axis_)
            del fig_
            del axis_
        else:
            figs.append(None)
            axes.append(None)

    meteo_model = AssimilationMeteoModel(stability_category, params.ws_tilde, params.wd_tilde)
    emps = [PuffModelEmpPdf(init_pdf.samples(n), time_step, puff_sampling_step, source_model,
            meteo_model, receptors, params, axis) for (source_model, axis) in zip(source_models, axes)]
    proposals = [ProposalType(emp) if ProposalType != type(None) else None for (ProposalType, emp) in zip(proposals, emps)]
    p_xt_xtp = TransitionCPdf(params)
    p_yt_xts = [ObservationCPdf(emp, receptors, meteo_model, params) for emp in emps]
    pfs = [pybayes.ParticleFilter(n, emp, p_xt_xtp, p_yt_xt, proposal) for (emp, p_yt_xt, proposal) in zip(emps, p_yt_xts, proposals)]
    n_effs = [[] for name in names]
    times = [[] for name in names]
    Qs = [[] for name in names]
    Q_stddevs = [[] for name in names]

    time_steps = simulation_length / puff_sampling_step

    origin = Location(0., 0., 0.)
    measurement = np.empty((2 + measured_doses.shape[1]))
    ws_noise = pybayes.GammaCPdf(params.gamma_ws)  # relative noise in wind speed measurement
    wd_noise = pybayes.MLinGaussCPdf(cov=np.array([[params.sigma_wd**2.]]), A=np.eye(1), b=np.zeros(1))
    y_noise = pybayes.GammaCPdf(params.gamma_y)  # relative noise in dosimeter measurements

    try:
        for i in range(time_steps):
            time = i*puff_sampling_step + 2*time_step  # measure approximately in the middle
            # add noise to meteo and receptor observation
            ws_real = measured_meteo.wind_speed_at(origin, time)
            wd_real = measured_meteo.wind_direction_at(origin, time)
            measurement[0] = ws_noise.sample(np.array([ws_real]))[0]
            measurement[1] = wd_noise.sample(np.array([wd_real]))[0]
            for j in range(measured_doses.shape[1]):
                measurement[j+2:j+3] = y_noise.sample(np.array([measured_doses[i, j] +
                                                                params.natural_dose_per_puff_sampling_step]))

            print "Passed ws={0:.2f} wd={1: .2f};         ".format(measurement[0], measurement[1]),
            print "                    real ws={0:.2f}        wd={1: .2f}".format(ws_real, wd_real)

            for (name, pf, p_yt_xt, fig, n_eff, time, Q, Q_stddev) in zip(names, pfs, p_yt_xts, figs, n_effs, times, Qs, Q_stddevs):
                t = clock()
                pf.bayes(measurement)
                t = clock() - t

                emp = pf.posterior()
                mean = emp.mean()
                var = emp.variance()
                ws = params.ws_tilde*mean[0]
                wd = params.wd_tilde + mean[1]
                distance = math.sqrt((math.sin(wd)*ws-math.sin(wd_real)*ws_real)**2.
                                   + (math.cos(wd)*ws-math.cos(wd_real)*ws_real)**2.)
                distance *= puff_sampling_step
                print "{0} [{1:5}s .. {2:5}s]: Neff={3:>7.3f}; t={4:>5.2f}".format(
                    name, i*puff_sampling_step, (i+1)*puff_sampling_step, emp.n_eff, t),
                print "estimated ws={0:.2f} ± {1:>4.2f}; wd={2: .2f} ± {3:>4.2f};".format(
                    ws, params.ws_tilde*math.sqrt(var[0]),
                    wd, math.sqrt(var[1])),
                print "dist={0:>5.1f} m;".format(distance),
                print "Q = {0:.2e} ± {1:.2e} Bq".format(mean[2], math.sqrt(var[2]))

                n_eff.append(emp.n_eff)
                time.append(t)
                Q.append(mean[2])
                Q_stddev.append(math.sqrt(var[2]))

                if create_video:
                    fig.savefig(join(source_dir, "video", "{0}-{1:0>5}s-{2:0>5}s.jpg".format(
                        name, i*puff_sampling_step, (i+1)*puff_sampling_step)))

    except KeyboardInterrupt as e:  # AttributeError
        print "Error occured:", e

    pickle.dump({"n_effs": n_effs, "times": times, "Qs": Qs, "Q_stddevs": Q_stddevs, "names": names},
                open(results_dir('assimilation.neffs_times.pickle'), 'wb'))

    if create_video:
        for name in names:
            file = open(join(source_dir, "video", name+"video.avi"), 'wb')
            args = ['jpegtoavi', '-f', '2', '800', '600']
            files = sorted(glob.glob(join(source_dir, "video", name + "*.jpg")))
            args.extend(files)
            subprocess.check_call(args, stdout=file)
            print "Video created:", file.name;

    if show_plot:
        plot_result_stats(n_effs, times, Qs, Q_stddevs, names)
        plt.show()


def plot_result_stats(n_effs, times, Qs, Q_stddevs, names):
    fig, axes = plt.subplots(1, 2, num='Result statistics')
    ax1, ax3 = axes

    names = ('bootstrap', 'a, b, Q proposal')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Effective particles')
    for (n_eff, name) in zip(n_effs, names):
        ax1.plot(n_eff, label=name)
    ax1.legend(loc=0)

    #ax2.set_xlabel('Time step')
    #ax2.set_ylabel('Computational Costs')
    #for (time, name) in zip(times, names):
        #ax2.plot(time, label=name)
    #ax2.legend()

    names = ('Simulation', 'Assimilation')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Q [Bq]')
    for (Q, Q_stddev, name) in zip(Qs, Q_stddevs, names):
        X = range(len(Q))
        ax3.errorbar(X, Q, yerr=Q_stddev, label=name)
    ax3.legend(loc=0)
    plt.tight_layout()


if __name__ == "__main__":
    main()
