#!/usr/bin/env python

"""Run a simulation of an atmospheric radioactive realease using the puff model"""

import matplotlib.pyplot as plt
import numpy as np
import pybayes.wrappers._numpy as nw
import scipy.io

import cProfile
import glob
import os.path
import pstats
import subprocess
import time

from asim.dispmodel.iface import Location, Nuclide, SourceModel, MeteoModel, DispersionModel
from asim.dispmodel.nuclides import db as nuclide_db
from asim.dispmodel.puffmodel import PuffModel
from asim.support.directories import results_dir
from asim.support.plotters import plot_trajectories, plot_contour, plot_stations, plot_map


class StaticCompoLocSourceModel(SourceModel):
    """Source model that provides static implementation of
    :meth:`~asim.dispmodel.iface.SourceModel.inventory` and
    :meth:`~asim.dispmodel.iface.SourceModel.location`.
    """

    def inventory(self):
        ret = np.array([nuclide_db["Ar-41"]], dtype=Nuclide)
        return ret

    def location(self):
        return Location(x=0.0, y=0.0, z=50.0)


class SimulatedSourceModel(StaticCompoLocSourceModel):
    """Source model that releases static pre-defined ammount of radionuclides."""

    def __init__(self, puff_sampling_step, activities):
        """
        :param int time_step: inverval between rate changes
        :param list rates: total activities per while time slots
        """
        self.puff_sampling_step = puff_sampling_step
        self.rates = np.asarray(activities) / puff_sampling_step

    def release_rate(self, time):
        index = time // self.puff_sampling_step  # integer-wise division
        if index >= 0 and index < self.rates.shape[0]:
            return self.rates[index]
        else:
            return np.zeros((self.rates.shape[1]))  # rate is zero before and after release


class PasquillsMeteoModel(MeteoModel):
    """Abstract meteo model based on Pasquills stability category"""

    def __init__(self, stability_category):
        """
        :param int stability_category: Pasquills statiblity category A=0 ... F=5
        """
        self.stability_category = stability_category
        self.height_per_category = np.array([1300.0, 920.0, 840.0, 500.0, 400.0, 150.0])

    def mixing_layer_height_at(self, loc, time):
        return self.height_per_category[self.stability_category]

    def dispersion_xy(self, loc, time, total_distance):
        sigmaxyA = [1.503, 0.876, 0.659, 0.640, 0.801, 1.294]
        sigmaxyB = [0.833, 0.823, 0.807, 0.784, 0.754, 0.718]
        index = self.stability_category
        if total_distance > 10000.:
            exponent = sigmaxyB[index] - 0.5
            return (sigmaxyA[index]*10000.**(exponent))*(total_distance**(0.5))
        else:
            return (sigmaxyA[index]*total_distance)**sigmaxyB[index]

    def dispersion_z(self, loc, time, total_distance):
        sigmazA  = [0.151, 0.127, 0.165, 0.215, 0.264, 0.241]
        sigmazB  = [1.219, 1.108, 0.996, 0.885, 0.774, 0.662]
        index = self.stability_category
        return (sigmazA[index]*total_distance)**sigmazB[index]


class StaticMeteoModel(PasquillsMeteoModel):
    """Weather model that returns pre-computed numbers stored in .mat file."""

    def __init__(self, meteo_array, stability_category, grid_step_xy = 3, grid_step_time = 3600):
        """
        :param string matfile: path to Matlab .mat file with pre-computed grid of wind speed and direction
        :param int stability_category: Pasquills statiblity category A=0 ... F=5
        """
        PasquillsMeteoModel.__init__(self, stability_category)
        self.METEO = np.asarray(meteo_array)
        self.grid_step_xy = grid_step_xy
        self.grid_step_time = grid_step_time

    def wind_speed_at(self, loc, time):
        tt = time // self.grid_step_time  # integer-wise division.

        ws11_0 = self.METEO[tt, 1, 0, 0]
        ws12_0 = self.METEO[tt, 1, 0, 0]
        ws21_0 = self.METEO[tt, 1, 0, 0]
        ws22_0 = self.METEO[tt, 1, 0, 0]

        ws11_1 = self.METEO[tt+1, 1, 0, 0]
        ws12_1 = self.METEO[tt+1, 1, 0, 0]
        ws21_1 = self.METEO[tt+1, 1, 0, 0]
        ws22_1 = self.METEO[tt+1, 1, 0, 0]

        """
        xgrid = loc.x // self.grid_step_xy + (self.METEO.shape[2]-1)/2.0
        x = loc.x % self.grid_step_xy
        ygrid = loc.y // self.grid_step_xy + (self.shape[3]-1)/2.0
        y = loc.y % self.grid_step_xy

        ws11_0 = METEO[tt, 1, xgrid, ygrid]
        ws12_0 = METEO[tt, 1, xgrid, ygrid+1]
        ws21_0 = METEO[tt, 1, xgrid+1, ygrid]
        ws22_0 = METEO[tt, 1, xgrid, ygrid+1]

        ws11_1 = METEO[tt+1, 1, xgrid, ygrid]
        ws12_1 = METEO[tt+1, 1, xgrid, ygrid+1]
        ws21_1 = METEO[tt+1, 1, xgrid+1, ygrid]
        ws22_1 = METEO[tt+1, 1, xgrid, ygrid+1]
        """

        ws_0 = self._bilinear_interp(ws11_0, ws12_0, ws21_0, ws22_0, 0, 0, self.grid_step_xy, self.grid_step_xy, loc.x, loc.y)
        ws_1 = self._bilinear_interp(ws11_1, ws12_1, ws21_1, ws22_1, 0, 0, self.grid_step_xy, self.grid_step_xy, loc.x, loc.y)

        coef = float(time % self.grid_step_time) / self.grid_step_time
        return coef*(ws_1 - ws_0) + ws_0

    def wind_direction_at(self, loc, time):
        tt = time // self.grid_step_time  # integer-wise division.

        wd11_0 = self.METEO[tt, 0, 0, 0]
        wd12_0 = self.METEO[tt, 0, 0, 0]
        wd21_0 = self.METEO[tt, 0, 0, 0]
        wd22_0 = self.METEO[tt, 0, 0, 0]

        wd11_1 = self.METEO[tt+1, 0, 0, 0]
        wd12_1 = self.METEO[tt+1, 0, 0, 0]
        wd21_1 = self.METEO[tt+1, 0, 0, 0]
        wd22_1 = self.METEO[tt+1, 0, 0, 0]

        """
        xgrid = loc.x // self.grid_step_xy + (self.METEO.shape[2]-1)/2.0
        x = loc.x % self.grid_step_xy
        ygrid = loc.y // self.grid_step_xy + (self.shape[3]-1)/2.0
        y = loc.y % self.grid_step_xy

        wd11_0 = METEO[tt, 0, xgrid, ygrid]
        wd12_0 = METEO[tt, 0, xgrid, ygrid+1]
        wd21_0 = METEO[tt, 0, xgrid+1, ygrid]
        wd22_0 = METEO[tt, 0, xgrid, ygrid+1]

        wd11_1 = METEO[tt+1, 0, xgrid, ygrid]
        wd12_1 = METEO[tt+1, 0, xgrid, ygrid+1]
        wd21_1 = METEO[tt+1, 0, xgrid+1, ygrid]
        wd22_1 = METEO[tt+1, 0, xgrid, ygrid+1]
        """

        wd_0 = self._bilinear_interp(wd11_0, wd12_0, wd21_0, wd22_0, 0, 0, self.grid_step_xy, self.grid_step_xy, loc.x, loc.y)
        wd_1 = self._bilinear_interp(wd11_1, wd12_1, wd21_1, wd22_1, 0, 0, self.grid_step_xy, self.grid_step_xy, loc.x, loc.y)

        coef = float(time % self.grid_step_time) / self.grid_step_time
        return coef*(wd_1 - wd_0) + wd_0

    def _bilinear_interp(self, q11, q12, q21, q22, x1, y1, x2, y2, x, y):
        """Bilinear interpolation."""
        return ( q11*(x2-x)*(y2-y)/(x2-x1)/(y2-y1) + q21*(x-x1)*(y2-y)/(x2-x1)/(y2-y1)
               + q12*(x2-x)*(y-y1)/(x2-x1)/(y2-y1) + q22*(x-x1)*(y-y1)/(x2-x1)/(y2-y1) )


def main(show_plot=True, create_video=False, profile=False):
    if profile:
        filename = "profile_" + __name__ + ".prof"
        cProfile.runctx("run(show_plot, create_video)", globals(), locals(), filename)
        s = pstats.Stats(filename)
        s.sort_stats("time")  # "time" or "cumulative"
        s.print_stats(30)
        s.print_callers(30)
    else:
        run(show_plot, create_video)


def run(show_plot, create_video):
    simulation_length = 4*60*60  # 4 hours
    time_step = 2*60  # 2 minutes
    puff_sampling_step = 5*time_step
    activities = np.array([[.1E+16], [3.0E+16], [4.0E+16], [2.0E+16], [3.0E+16], [1.0E+16]])

    source_dir = os.path.dirname(__file__)
    matfile = os.path.join(source_dir, "meteo.mat")
    souce_model = SimulatedSourceModel(puff_sampling_step=puff_sampling_step, activities=activities)
    meteo_model = StaticMeteoModel(meteo_array=scipy.io.loadmat(matfile)["METEO"], stability_category=3)
    puff_model = PuffModel(time_step=time_step, puff_sampling_step=puff_sampling_step,
        source_model=souce_model)

    receptors_ = []
    for i in range(-20, 21):
        for j in range(-20, 21):
            receptors_.append(Location(j*1000.0, i*1000.0, 0.))
    receptors = np.array(receptors_, dtype=Location)
    receptors_ = read_receptor_locations(os.path.join(source_dir, "receptory_ETE"))
    off_grid_receptors = np.array(receptors_, dtype=Location)
    del receptors_

    time_steps = (simulation_length // time_step)
    puff_count = len(activities)
    trajectories = np.zeros((time_steps + 1, puff_count, 2))
    total_doses = np.zeros(receptors.shape[0])
    off_grid_doses = np.zeros((simulation_length // puff_sampling_step, off_grid_receptors.shape[0]))

    clock_start, time_start = time.clock(), time.time()
    for i in range(time_steps):
        if i % (puff_sampling_step/time_step) == 0:
            print
        puff_model.propagate(meteo_model, souce_model)

        dose_this_round = 0.0
        for j in range(len(puff_model.puffs)):
            loc = puff_model.puffs[j].loc
            trajectories[i + 1, j, 0] = loc.x  # let all trajectories start at 0
            trajectories[i + 1, j, 1] = loc.y
        for j in range(receptors.shape[0]):
            loc = receptors[j]
            dose = nw.sum_v(puff_model.dose_at(loc))
            total_doses[j] += dose
            dose_this_round += dose
        for j in range(off_grid_receptors.shape[0]):
            loc = off_grid_receptors[j]
            index = (i * time_step) // puff_sampling_step
            off_grid_doses[index, j] += nw.sum_v(puff_model.dose_at(loc))
        print "Total dose in period [{0:5}s .. {1:5}s]: {2:.9f}".format(i*time_step, (i+1)*time_step, dose_this_round)
        if create_video:
            file = os.path.join(os.path.dirname(__file__), "video", "{0:0>5}s-{1:0>5}s.jpg".format(i*time_step, (i+1)*time_step))
            fig = plt.figure()
            axis = fig.add_subplot(1,1,0)

            total_doses_np = np.reshape(total_doses, (41, 41))
            plot_trajectories(axis, trajectories[0:i + 2, ...])
            plot_contour(axis, total_doses_np, 41, 41, 1.0, 5)
            plot_stations(axis, off_grid_receptors)
            plot_map(axis)
            axis.grid(True)

            fig.savefig(file)
            plt.close('all')

    print "Wall time:", time.time() - time_start, "s, CPU time:", time.clock() - clock_start
    scipy.io.savemat(results_dir('simulation.doses_at_locations'),
                     {"dose_of_model": off_grid_doses})

    if create_video:
        file = open(os.path.join(source_dir, "video", "video.avi"), 'wb')
        args = ['jpegtoavi', '-f', '10', '800', '600']
        files = sorted(glob.glob(os.path.join(source_dir, "video", "*.jpg")))
        args.extend(files)
        subprocess.check_call(args, stdout=file)
        print "Video created:", file.name;
    if show_plot:
        fig = plt.figure()
        axis = fig.add_subplot(1,1,1)

        total_doses_np = np.reshape(total_doses, (41, 41))
        plot_trajectories(axis, trajectories)
        plot_contour(axis, total_doses_np, 41, 41, 1.0, 5)
        plot_stations(axis, off_grid_receptors)
        plot_map(axis)
        axis.grid(True)

        plt.show()


def read_receptor_locations(path):
    """Loads list of receptors from a file

    :return: list of :class:`Locations <asim.dispmodel.iface.Location>` with receptor locations
    """
    receptors = []
    f = open(path, "r")
    s = f.readlines()
    for line in s[0:len(s)]:
        singleLine = line.rstrip()
        name = singleLine[14:42]  # currently unused
        north = float(singleLine[42:54])
        east = float(singleLine[54:62])
        receptors.append(Location(north, east, 0.))
    f.close()
    return receptors


if __name__ == "__main__":
    main()
