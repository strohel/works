#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import os.path
import pickle


def plot_neff_cost(n_effs, times, names):
    styles = ["-", "--"]

    axis = plt.figure('Effective particles').add_subplot(1, 1, 1)
    axis.set_xlabel('Time step')
    axis.set_ylabel('Effective particles')
    for (n_eff, name, style) in zip(n_effs, names, styles):
        axis.plot(n_eff, style, label=name)
    axis.legend(loc='center right', bbox_to_anchor=(1.0, 0.34))

    axis = plt.figure("Computational Costs").add_subplot(1, 1, 1)
    axis.set_xlabel('Time step')
    axis.set_ylabel('Computational Costs')
    for (time, name, style) in zip(times, names, styles):
        axis.plot(time, style, label=name)
    axis.legend()

    axis = plt.figure("Cost per effective particle").add_subplot(1, 1, 1)
    axis.set_xlabel('Time step')
    axis.set_ylabel('Cost per effective particle')
    for (neff, time, name, style) in zip(n_effs, times, names, styles):
        axis.semilogy(np.array(time)/np.array(neff), style, label=name)
    axis.legend()

    axis = plt.figure("Effective particles per second").add_subplot(1, 1, 1)
    axis.set_xlabel('Time step')
    axis.set_ylabel('Effective particles per second')
    for (neff, time, name, style) in zip(n_effs, times, names, styles):
        axis.plot(np.array(neff)/np.array(time), style, label=name)
    axis.legend(loc='upper left')


def main():
    source = os.path.dirname(__file__)
    filename = os.path.join(source, "neffs_times.pickle")
    data = pickle.load(open(filename, "rb"))
    n_effs = data["n_effs"][2:]
    times = data["times"][2:]
    names = data["names"][2:]

    plot_neff_cost(n_effs, times, names)
    plt.show()


if __name__ == "__main__":
    main()
