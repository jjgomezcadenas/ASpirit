import numpy as np
import matplotlib.pyplot as plt

from cycler     import cycler
from contextlib import contextmanager


color_sequence = ("k", "m", "g", "b", "r",
                  "gray", "aqua", "gold", "lime", "purple",
                  "brown", "lawngreen", "tomato", "lightgray", "lightpink")

def auto_plot_style(overrides = dict()):
    plt.rcParams[ "figure.figsize"               ] = 10, 8
    plt.rcParams[   "font.size"                  ] = 25
    plt.rcParams[  "lines.markersize"            ] = 25
    plt.rcParams[  "lines.linewidth"             ] = 3
    plt.rcParams[  "patch.linewidth"             ] = 3
    plt.rcParams[   "axes.linewidth"             ] = 2
    plt.rcParams[   "grid.linewidth"             ] = 3
    plt.rcParams[   "grid.linestyle"             ] = "--"
    plt.rcParams[   "grid.alpha"                 ] = 0.5
    plt.rcParams["savefig.dpi"                   ] = 300
    plt.rcParams["savefig.bbox"                  ] = "tight"
    plt.rcParams[   "axes.formatter.use_mathtext"] = True
    plt.rcParams[   "axes.formatter.limits"      ] = (-3 ,4)
    plt.rcParams[  "xtick.major.size"            ] = 10
    plt.rcParams[  "ytick.major.size"            ] = 10
    plt.rcParams[  "xtick.minor.size"            ] = 5
    plt.rcParams[  "ytick.minor.size"            ] = 5
    plt.rcParams[   "axes.prop_cycle"            ] = cycler(color=color_sequence)
    plt.rcParams[  "image.cmap"                  ] = "gnuplot2"
    plt.rcParams.update(overrides)


@contextmanager
def temporary(name, new_value):
    old_value          = plt.rcParams[name]
    plt.rcParams[name] = new_value
    try    : yield
    finally: plt.rcParams[name] = old_value


def normhist(x, *args, normto=100, normfactor=None, **kwargs):
    if "histtype" not in kwargs:
        kwargs["histtype"] = "step"
    if normfactor is None:
        w = np.full(len(x), normto/len(x))
    else:
        w = np.full(len(x), normfactor)
    return plt.hist(x, *args, weights=w, **kwargs)


def normhist2d(x, y, *args, normto=100, normfactor=None, **kwargs):
    if normfactor is None:
        w = np.full(len(x), normto/len(x))
    else:
        w = np.full(len(x), normfactor)
    if "cmin" not in kwargs:
        kwargs["cmin"] = w[0]
    return plt.hist2d(x, y, *args, weights=w, **kwargs)
