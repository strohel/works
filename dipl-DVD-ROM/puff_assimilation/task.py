#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from asim.assimilation.twin import Params, AssimilationMeteoModel, AssimilationSourceModel
from asim.assimilation.twin import PuffModelEmpPdf, CongujateProposalFilter, TransitionCPdf, ObservationCPdf
from asim.dispmodel.iface import Location
import numpy as np
import simplejson as json
import pybayes

import math
from os.path import dirname, join
import time


def compute(input):
    params = Params()  # default parameters, also an object storing them

    # get parameters out of supplied JSON
    simulation_length = input['simulation_length']
    time_step = input['time_step']
    puff_sampling_step = input['source_model']['puff_sampling_step']
    receptors = np.array([Location(r['x'], r['y'], r['z']) for r in input['off_grid_receptors']], dtype=Location)
    measured_doses = np.array(input['off_grid_doses'])

    assimilation_params = input['assimilation'] if 'assimilation' in input else {}
    if 'N' in assimilation_params:
        n = assimilation_params['N']
    else:
        n = 1000  # number of particles
        print "Using default value for assimilation.N of", n

    meteo_model_params = input['meteo_model']
    if 'static_ws' in meteo_model_params:
        params.ws_tilde = meteo_model_params['static_ws']
    else:
        meteo_model_params['static_ws'] = params.ws_tilde
        print "Using default value for meteo_model.static_ws of", params.ws_tilde
    if 'static_wd' in meteo_model_params:
        params.wd_tilde = meteo_model_params['static_wd']
    else:
        meteo_model_params['static_wd'] = params.wd_tilde
        print "Using default value for meteo_model.static_wd of", params.wd_tilde

    wind_speeds_at_origin = input['wind_speeds_at_origin']
    wind_directions_at_origin = input['wind_directions_at_origin']


    # prepare the assimilation
    init_a_pdf = pybayes.GammaPdf(4., 0.25)  # mean = 1, std dev = 0.5
    init_b_pdf = pybayes.GaussPdf(np.array([0.]), np.array([[(math.pi/2.)**2.]]))
    init_Q_pdf = pybayes.UniPdf(np.array([0.]), np.array([10.0E+16]))  # not actually used
    init_pdf = pybayes.ProdPdf((init_a_pdf, init_b_pdf, init_Q_pdf))

    source_model = AssimilationSourceModel(puff_sampling_step)
    meteo_model = AssimilationMeteoModel(**meteo_model_params)
    emp = PuffModelEmpPdf(init_pdf.samples(n), time_step, puff_sampling_step, source_model,
                          meteo_model, receptors, params)
    proposal = CongujateProposalFilter(emp)
    p_xt_xtp = TransitionCPdf(params)
    p_yt_xt = ObservationCPdf(emp, receptors, meteo_model, params)
    pf = pybayes.ParticleFilter(n, emp, p_xt_xtp, p_yt_xt, proposal)

    time_steps = simulation_length / puff_sampling_step
    measurement = np.empty(2 + measured_doses.shape[1])

    n_eff, spent_time, Q, Q_stddev = [], [], [], []  # result arrays
    clock_start, time_start = time.clock(), time.time()  # timing

    # actual run!
    for i in range(time_steps):
        measurement[0] = wind_speeds_at_origin[i]
        measurement[1] = wind_directions_at_origin[i]
        measurement[2:] = measured_doses[i, :] + params.natural_dose_per_puff_sampling_step
        print "Passed", 44*' ', "ws={0:.2f}".format(measurement[0]), 7*' ', "wd={0: .2f}".format(measurement[1])

        t = time.clock()
        pf.bayes(measurement)
        t = time.clock() - t

        emp = pf.posterior()
        mean = emp.mean()
        var = emp.variance()
        ws = params.ws_tilde*mean[0]
        wd = params.wd_tilde + mean[1]
        print "[{0:5}s .. {1:5}s]: Neff={2:>7.3f}; t={3:>5.2f}".format(
            i*puff_sampling_step, (i+1)*puff_sampling_step, emp.n_eff, t),
        print "estimated ws={0:.2f} ± {1:>4.2f}; wd={2: .2f} ± {3:>4.2f};".format(
            ws, params.ws_tilde*math.sqrt(var[0]) if var[0] >= 0. else float('nan'),
            wd, math.sqrt(var[1]) if var[1] >= 0. else float('nan')),
        curr_Q_stddev = math.sqrt(var[2]) if var[2] >= 0. else float('nan')
        print "Q = {0:.2e} ± {1:.2e} Bq".format(mean[2], curr_Q_stddev)

        n_eff.append(emp.n_eff)
        spent_time.append(t)
        Q.append(mean[2])
        Q_stddev.append(curr_Q_stddev)

    print "Wall time:", time.time() - time_start, "s, CPU time:", time.clock() - clock_start
    ret = {
        'n_eff': n_eff,
        'spent_time': spent_time,
        'Q': Q,
        'Q_stddev': Q_stddev
    }
    return ret


if __name__ == '__main__':
    with open(join(dirname(__file__), 'example_input.json')) as file:
        input = json.load(file)
    ouput = compute(input)
    with open(join(dirname(__file__), 'example_output.json'), 'w') as file:
        json.dump(ouput, file, sort_keys=True, indent=4*' ')
