#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from asim.simulation.simple import SimulatedSourceModel, StaticMeteoModel
from asim.dispmodel.iface import Location
from asim.dispmodel.puffmodel import PuffModel
import simplejson as json
import numpy as np

from os.path import dirname, join
import time


def compute(input):
    simulation_length = input['simulation_length']
    time_step = input['time_step']
    puff_sampling_step = input['source_model']['puff_sampling_step']
    activities = input['source_model']['activities']

    source_model = SimulatedSourceModel(**input['source_model'])
    meteo_model = StaticMeteoModel(**input['meteo_model'])
    puff_model = PuffModel(time_step=time_step, puff_sampling_step=puff_sampling_step,
        source_model=source_model)

    receptors_ = []
    for i in range(-20, 21):
        for j in range(-20, 21):
            receptors_.append(Location(j*1000.0, i*1000.0, 0.))
    receptors = np.array(receptors_, dtype=Location)
    off_grid_receptors = np.array([Location(r['x'], r['y'], r['z']) for r in input['receptors']], dtype=Location)

    time_steps = (simulation_length // time_step)
    puff_count = len(activities)
    trajectories = np.zeros((time_steps + 1, puff_count, 2))
    total_doses = np.zeros(receptors.shape[0])
    off_grid_doses = np.zeros((simulation_length // puff_sampling_step, off_grid_receptors.shape[0]))
    wind_speeds_at_origin, wind_directions_at_origin = [], []

    clock_start, time_start = time.clock(), time.time()
    for i in range(time_steps):
        if i % (puff_sampling_step/time_step) == 0:
            print
        wind_speeds_at_origin.append(meteo_model.wind_speed_at(source_model.location(), (i + 0.5)*time_step))
        wind_directions_at_origin.append(meteo_model.wind_direction_at(source_model.location(), (i + 0.5)*time_step))
        puff_model.propagate(meteo_model, source_model)

        dose_this_round = 0.0
        for j in range(len(puff_model.puffs)):
            loc = puff_model.puffs[j].loc
            trajectories[i + 1, j, 0] = loc.x  # let all trajectories start at 0
            trajectories[i + 1, j, 1] = loc.y
        for j in range(receptors.shape[0]):
            loc = receptors[j]
            dose = np.sum(puff_model.dose_at(loc))
            total_doses[j] += dose
            dose_this_round += dose
        for j in range(off_grid_receptors.shape[0]):
            loc = off_grid_receptors[j]
            index = (i * time_step) // puff_sampling_step
            off_grid_doses[index, j] += np.sum(puff_model.dose_at(loc))
        print "Total dose in period [{0:5}s .. {1:5}s]: {2:.9f}".format(i*time_step, (i+1)*time_step, dose_this_round)

    print "Wall time:", time.time() - time_start, "s, CPU time:", time.clock() - clock_start
    ret = {
        "simulation_length": simulation_length,
        "time_step": time_step,
        "source_model": {"puff_sampling_step": puff_sampling_step},
        "meteo_model": {"stability_category": input['meteo_model']['stability_category']},
        "wind_speeds_at_origin": wind_speeds_at_origin,
        "wind_directions_at_origin": wind_directions_at_origin,

        "trajectories": trajectories,
        "grid_receptors": receptors,
        "grid_doses_total": total_doses,
        "off_grid_receptors": off_grid_receptors,
        "off_grid_doses": off_grid_doses
    }
    return ret

def location_json_converter(obj):
    if isinstance(obj, Location):
        return {'x': obj.x, 'y': obj.y, 'z': obj.z}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(repr(obj) + " is not JSON serializable")


if __name__ == '__main__':
    with open(join(dirname(__file__), 'example_input.json')) as file:
        input = json.load(file)
    ouput = compute(input)
    with open(join(dirname(__file__), 'example_output.json'), 'w') as file:
        json.dump(ouput, file, sort_keys=True, indent=4*' ', default=location_json_converter)
