import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def make_pretty_plot(dpi=600):
    """
    Use first thing after setting up the plt.figure() to make things pretty.
    """
    mpl.rcParams["figure.dpi"] = dpi

    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['xtick.major.top'] = True
    mpl.rcParams['xtick.minor.top'] = True
    mpl.rcParams['xtick.major.bottom'] = True
    mpl.rcParams['xtick.minor.bottom'] = True
    mpl.rcParams['xtick.direction'] = 'in'

    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['ytick.left'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['ytick.major.right'] = True
    mpl.rcParams['ytick.minor.right'] = True
    mpl.rcParams['ytick.major.left'] = True
    mpl.rcParams['ytick.minor.left'] = True
    mpl.rcParams['ytick.direction'] = 'in'

    mpl.rcParams['legend.frameon'] = False

def figure(ncols=1, nrows=1, axwidth=3, axheight=None, padx=0., pady=0., height_ratios=None, width_ratios=None,
           add_title_space=False, sharex='none', sharey='none'):
    if axheight is None:
        axheight = axwidth

    if height_ratios is None:
        height_ratios = np.ones(nrows)
    if width_ratios is None:
        width_ratios = np.ones(ncols)
    gridspec = dict(hspace=pady, wspace=padx, height_ratios=height_ratios, width_ratios=width_ratios)

    height = nrows * axheight + (nrows-1) * pady + 0.
    height *= np.sum(height_ratios) / nrows
    if add_title_space:
        height += 1.

    width = ncols * axwidth + (ncols-1) * padx + 0.
    width *= np.sum(width_ratios) / ncols

    figsize = (width, height)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey, gridspec_kw=gridspec)

    return fig, axes

def organize_plot(ax, title=None, title_size=14, show_grid=False, grid_size=None, legend_location=None, legend_size=6,
                  xlims=None, ylims=None, xticks=None, xticks_rotation=None, yticks_rotation=None, yticks=None, xticks_size=None, yticks_size=None,
                  xlabel=None, xlabel_pad=4., xlabel_size=12, xlabel_rotation=0, ylabel=None, ylabel_pad=4., ylabel_size=12, ylabel_rotation=0, xlabel_color=None, ylabel_color=None,
                  xscale=None, yscale=None, ticks_inside=False, ticks_length=None, ticks_width=None):
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if show_grid:
        ax.grid(True)
        if grid_size is not None:
            ax.grid(which="major", linewidth=grid_size[0])
            ax.grid(which="minor", linewidth=grid_size[1])
    if legend_location is not None:
        ax.legend(loc=legend_location, fontsize=legend_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_size, rotation=xlabel_rotation, labelpad=xlabel_pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_size, rotation=90 - ylabel_rotation, labelpad=ylabel_pad)
    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None:
        ax.set_ylim(ylims[0], ylims[1])
    if xticks is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xticks[0]))
        if len(xticks) > 1:
            ax.xaxis.set_minor_locator(MultipleLocator(xticks[1]))
    if yticks is not None:
        ax.yaxis.set_major_locator(MultipleLocator(yticks[0]))
        if len(yticks) > 1:
            ax.yaxis.set_minor_locator(MultipleLocator(yticks[1]))
    if xticks_size is not None:
        plt.xticks(fontsize=xticks_size)
    if yticks_size is not None:
        plt.yticks(fontsize=yticks_size)
    if xlabel_color is not None:
        ax.xaxis.label.set_color(xlabel_color)
        ax.tick_params(axis='x', colors=xlabel_color)
    if ylabel_color is not None:
        ax.yaxis.label.set_color(ylabel_color)
        ax.tick_params(axis='y', colors=ylabel_color)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xticks_rotation is not None:
        ax.tick_params(axis='x', labelrotation=xticks_rotation)
    if yticks_rotation is not None:
        ax.tick_params(axis='y', labelrotation=yticks_rotation)
    if ticks_inside:
        ax.tick_params(which='both', direction='in')
    if ticks_length is not None:
        ax.minorticks_on()
        ax.tick_params(which='major', length=ticks_length[0])
        ax.tick_params(which='minor', length=ticks_length[1])
    if ticks_width is not None:
        ax.tick_params(which='major', width=ticks_width[0])
        ax.tick_params(which='minor', width=ticks_width[1])

