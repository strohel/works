from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import copy
import math
import os.path


def plot_trajectories(ax, TRAJECTORIES, alphas = None, colors = None):
    """TRAJECTORIES = numpy.zeros((pC, tC, 2)) #0 = x, 1 = y"""
    pC = TRAJECTORIES.shape[1]
    if alphas is not None and len(alphas) != pC:
        raise AssertionError("Expected len(alphas) = {0}, but len(alphas) = {1}".format(pC, len(alphas)))
    if colors is not None and len(colors) != pC:
        raise AssertionError("Expected len(colorss) = {0}, but len(colors) = {1}".format(pC, len(colors)))
    for i in range(pC):
        params = {}
        if alphas is not None:
            params["alpha"] = alphas[i]
        else:
            params["alpha"] = 0.5
        if colors is not None:
            params["color"] = colors[i]
        ax.plot(TRAJECTORIES[:, i, 0], TRAJECTORIES[:, i, 1], **params)

def plot_contour(ax, M, dimX, dimY, deltaStep, rady, own_levels = None):
    x = np.arange(-math.floor(dimX/2.0)*1000, math.ceil(dimX/2.0)*1000, deltaStep*1000)
    y = np.arange(-math.floor(dimY/2.0)*1000, math.ceil(dimY/2.0)*1000, deltaStep*1000)

    X, Y = np.meshgrid(x, y)
    logMax = math.log(M.max(), 10.0)
    radMaxima = math.floor(logMax)

    if own_levels is not None:
        levels = own_levels
    else:
        levels = np.logspace(radMaxima - rady + 1, radMaxima + 1, num=rady + 1)

    cset1 = ax.contourf(X, Y, M, levels, alpha=0.3, cmap=plt.cm.jet, norm=matplotlib.colors.LogNorm())
    ax.contour(X, Y, M, levels, colors='k', hold='on', linewidths=(1,))
    plt.colorbar(cset1)

def plot_stations(ax, locations, add_center=True):
    """Plot points representing stations to axis *ax*

    :param locations: iterable of Location objects with station locations
    """
    for loc in locations:
        ax.scatter(loc.x, loc.y, marker='o', s=10, c='b')
    if add_center:
        ax.scatter([0.0], [0.0], marker='o', s=20, color="red")

def plot_map(ax):
    radius = 20.0
    map_path = os.path.join(os.path.dirname(__file__), "teme_osm_aligned.png")
    orig_size = 45.61 #puvodni mapa ma 90km
    map = get_square_map_of_given_side(map_path, orig_size, 2*radius+1.0)

    const = 1000
    ax.imshow(map, extent=(-1.*radius*const, radius*const, radius*const, -1.*radius*const), origin='lower')
    ax.set_xlim(-3.5*const, radius*const)
    ax.set_ylim(-3.5*const, radius*const)

def get_square_map_of_given_side(path, a1, a2):
    """vyrizne z mapy na ceste path a puvodnim rozmeru a1 novou mapu s rozmerem a2"""
    #mam tu jen x0, pac mapa je ctvercova....
    img = Image.open(path)
    x0 = img.size[0] #sirka obrazku
    y0 = img.size[1] #vyska obrazku
    x1 = int(x0/2 - x0*a2/(2*a1))
    y1 = int(y0/2 + y0*a2/(2*a1))
    cropped = img.crop([ x1, x1, y1, y1])
    #plt.imshow(cropped, extent=(-a2/2, a2/2, -a2/2, a2/2), origin='lower')
    #plt.show()
    return cropped
