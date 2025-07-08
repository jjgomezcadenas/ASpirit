
import matplotlib 
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing  import List, Tuple, Optional 
import numpy as np

from stats  import mean_and_std

@dataclass
class PlotLabels:
    x     : str
    y     : str
    title : str

def labels(pl : PlotLabels):
    """
    Set x and y labels.
    """
    plt.xlabel(pl.x)
    plt.ylabel(pl.y)
    plt.title (pl.title)


def plot_histo(pltLabels, ax, legend=True, legendsize=10, legendloc='best', log=False, labelsize=11):
    """
    Configures axis labels, title, and optional legend for a matplotlib histogram plot.

    Parameters
    ----------
    pltLabels : PlotLabels
        Object containing the x-axis label, y-axis label, and optional title.
    ax : matplotlib.axes.Axes
        Axis object to apply the plot labels and legend.
    legend : bool, default True
        Whether to display a legend on the plot.
    legendsize : int, default 10
        Font size of the legend text.
    legendloc : str, default 'best'
        Location of the legend in the plot.
    labelsize : int, default 11
        Font size of the x-axis and y-axis labels.

    Returns
    -------
    None
    """

    if legend:
        ax.legend(fontsize=legendsize, loc=legendloc)
    ax.set_xlabel(pltLabels.x, fontsize=labelsize)
    ax.set_ylabel(pltLabels.y, fontsize=labelsize)
    if pltLabels.title:
        ax.set_title(pltLabels.title)
    if log:
        ax.set_yscale('log')


def plot_h2d(fig, axs, hm, xedges, yedges, xlabel="X", ylabel="Y", title="Map"):
    """
    Plot a 2D histogram heatmap using precomputed bin edges and values.

    Parameters:
    - fig: matplotlib.figure.Figure object
    - axs: matplotlib.axes.Axes object to plot on
    - hm: 2D array of values (heatmap)
    - xedges: 1D array of bin edges along the X-axis
    - yedges: 1D array of bin edges along the Y-axis
    - xlabel: label for the X-axis
    - ylabel: label for the Y-axis
    - title: plot title
    """
    mesh = axs.pcolormesh(xedges, yedges, hm.T, cmap='viridis', shading='auto')
    fig.colorbar(mesh, ax=axs, label='Value')
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)
    

def h1d(
    x: np.ndarray,
    bins: int,
    xrange: Tuple[float, float],
    weights: Optional[np.ndarray] = None,
    log: bool = False,
    normed: bool = False,
    color: str = 'black',
    width: float = 1.5,
    style: str = 'solid',
    stats: bool = True,
    lbl: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 4),
    ax: Optional[matplotlib.axes.Axes] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Plot a 1D histogram with optional statistics.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    bins : int
        Number of bins.
    xrange : Tuple[float, float]
        Axis limits.
    weights : np.ndarray, optional
        Data weights.
    log : bool, optional
        Log scale y-axis.
    normed : bool, optional
        Normalize histogram.
    color : str, optional
        Edge color.
    width : float, optional
        Line width.
    style : str, optional
        Line style.
    stats : bool, optional
        Append stats to label.
    lbl : str, optional
        Extra legend label.
    figsize : Tuple[int, int], optional
        Size of figure if ax is None.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on.

    Returns
    -------
    n : np.ndarray
        Bin counts.
    b : np.ndarray
        Bin edges.
    mu : float
        Mean.
    std : float
        Standard deviation.
    """
    mu, std = mean_and_std(x, xrange)

    label_parts = []
    if stats:
        label_parts += [f'Entries = {len(x)}',
                        r'$\mu$ = {:7.2f}'.format(mu),
                        r'$\sigma$ = {:7.2f}'.format(std)]
    if lbl:
        label_parts.append(lbl)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    hist_args = dict(
        bins=bins, range=xrange, weights=weights, log=log,
        density=normed, histtype='step', linewidth=width,
        linestyle=style, edgecolor=color, label='\n'.join(label_parts)
    )

    n, b, _ = ax.hist(x, **hist_args)

    if ax is None:
        plt.show()

    return n, b, mu, std


def h1dm(
    x: List[np.ndarray],
    bins: List[int],
    xrange: List[Tuple[float, float]],
    weights: Optional[List[Optional[np.ndarray]]] = None,
    log: Optional[List[bool]] = None,
    normed: Optional[List[bool]] = None,
    color: Optional[List[str]] = None,
    width: Optional[List[float]] = None,
    style: Optional[List[str]] = None,
    stats: Optional[List[bool]] = None,
    lbl: Optional[List[Optional[str]]] = None,
    pltLabels: Optional[List[PlotLabels]] = None,
    legendloc: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (6, 6)
):
    """
    Plot multiple 1D histograms with statistics.

    Parameters
    ----------
    x : List[np.ndarray]
        Data arrays.
    bins : List[int]
        Number of bins.
    xrange : List[Tuple[float, float]]
        Axis ranges.
    weights : List[np.ndarray], optional
        Data weights.
    log, normed : List[bool], optional
        Log-scale and normalization.
    color, style : List[str], optional
        Edge color and line style.
    width : List[float], optional
        Line widths.
    stats : List[bool], optional
        Show stats in label.
    lbl : List[str], optional
        Extra label.
    pltLabels : List[PlotLabels], optional
        Axis labels and titles.
    legendloc : List[str], optional
        Legend position.
    figsize : Tuple[float, float], optional
        Size of each subplot.

    Returns
    -------
    n_stats, b_stats : List[np.ndarray]
        Histogram counts and bin edges.
    mu_stats, std_stats : List[float]
        Means and standard deviations.
    """
    n_plots = len(x)

    if n_plots <= 2:
        rows, cols = 1, n_plots
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 8:
        rows, cols = 2, 4
    else:
        cols = 4
        rows = (n_plots + 2) // 4

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0]*cols, figsize[1]*rows))
    axes = np.atleast_1d(axes).flatten()

    n_stats, b_stats, mu_stats, std_stats = [], [], [], []

    for i in range(n_plots):
        ax = axes[i]
        n, b, mu, std = h1(
            x[i],
            bins=bins[i],
            xrange=xrange[i],
            weights=weights[i] if weights else None,
            log=log[i] if log else False,
            normed=normed[i] if normed else False,
            color=color[i] if color else 'black',
            width=width[i] if width else 1.5,
            style=style[i] if style else 'solid',
            stats=stats[i] if stats else True,
            lbl=lbl[i] if lbl else None,
            ax=ax
        )

        plabels = pltLabels[i] if pltLabels else PlotLabels(x='x', y='y', title=None)
        plot_histo(plabels, ax, legendloc=legendloc[i] if legendloc else 'best')

        n_stats.append(n)
        b_stats.append(b)
        mu_stats.append(mu)
        std_stats.append(std)

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return n_stats, b_stats, mu_stats, std_stats


def h2d(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    nbins_x: int,
    nbins_y: int,
    range_x: Tuple[float, float],
    range_y: Tuple[float, float],
    cmap: str = "viridis",
    pltLabels: PlotLabels = PlotLabels(x='x', y='y', title=None),
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[matplotlib.axes.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    over_color: str = 'darkred',
    bad_color: str = 'gray'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot a weighted 2D histogram with custom color handling.

    Parameters
    ----------
    x, y : np.ndarray
        Input data.
    weights : np.ndarray
        Weights for each (x, y) pair.
    nbins_x, nbins_y : int
        Number of bins along x and y axes.
    range_x, range_y : Tuple[float, float]
        Axis limits.
    cmap : str, optional
        Colormap (default 'viridis').
    pltLabels : PlotLabels, optional
        Axis labels and title.
    figsize : Tuple[int, int], optional
        Size of the figure if ax not provided.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on (creates new if None).
    vmin, vmax : float, optional
        Min and max color scale values.
    over_color : str, optional
        Color for values above vmax (default 'darkred').
    bad_color : str, optional
        Color for bad/NaN values (default 'gray').

    Returns
    -------
    hh : np.ndarray
        2D histogram values.
    xedges, yedges : np.ndarray
        Bin edges.
    """
    if not (len(x) == len(y) == len(weights)):
        raise ValueError("x, y, and weights must have the same length")

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

    xbins = np.linspace(*range_x, nbins_x + 1)
    ybins = np.linspace(*range_y, nbins_y + 1)

    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_under(color='white')
    cmap_obj.set_over(color=over_color)
    cmap_obj.set_bad(color=bad_color)

    hh, xedges, yedges, img = ax.hist2d(
        x, y,
        bins=[xbins, ybins],
        weights=weights,
        cmap=cmap_obj,
        vmin=vmin or 1e-10,
        vmax=vmax
    )

    plt.colorbar(img, ax=ax, label='Value')
    labels(pltLabels)

    if ax is None:
        plt.show()

    return hh, xedges, yedges
