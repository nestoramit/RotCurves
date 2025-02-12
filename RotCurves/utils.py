import sys
import os
import time
import datetime
import itertools
import pandas as pd
import numpy as np
import math
import parmap
import logging
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import *
import astropy.constants as const
import astropy.units as u
import scipy.optimize as scp_opt
import scipy.integrate as scp_integrate
import scipy.special as scp_functions
import scipy.interpolate as scp_interp
import scipy.signal as scp_sig
import scipy.ndimage as scp_img
import seaborn as sns
import pickle
import scipy.optimize as slv
import random
from scipy.ndimage import gaussian_filter1d, rotate
import scipy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurve')


# Global constants
pi = np.pi
G = const.G.to(u.m**3 / u.kg / u.s**2).value  # m^3 kg^-1 s^-2
pc = const.pc.to(u.m).value  # m
kpc = 1e3 * pc  # m
H0 = 70 * 10 ** 3 / (1e3 * kpc)  # m / s / Mpc
M_solar = const.M_sun.to(u.kg).value  # kg
FWHM2sig = 2.35482

# Set lookuptable folders path
lookuptables_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lookup_tables")
noor_lookuptables_dir = os.path.join(lookuptables_dir, "Noordermeer_lookup_tables")
gauss_lookuptables_dir = os.path.join(lookuptables_dir, "GaussianRing_lookup_tables")
gauss_BTmin_lookuptables_dir = os.path.join(lookuptables_dir, "GaussianRing_BTmin_lookup_tables")

# Load Noordermeer lookuptables with the first value of n and q
def load_noordermeer_tables():
    Noordermeer_lookup_tables = {}
    q_list = np.asarray(list(
        set([float(x[x.find('_q') + 2:x.find('.npy')]) for x in os.listdir(noor_lookuptables_dir) if x.find('npy') > 0])))
    n_list = np.asarray(list(
        set([float(x[x.find('_n') + 2:x.find('_q')]) for x in os.listdir(noor_lookuptables_dir) if x.find('npy') > 0])))
    Noordermeer_lookup_tables['q_list'] = q_list
    Noordermeer_lookup_tables['n_list'] = n_list

    for n in n_list:
        Noordermeer_lookup_tables[n] = {}
        for q in q_list:
            try:
                Noordermeer_lookup_tables[n][q] = np.load(
                    os.path.join(noor_lookuptables_dir, "noor_n%2.2f_q%2.2f.npy" % (n, q)))
            except:
                pass

    return Noordermeer_lookup_tables

def load_gaussian_tables():
    GaussianRing_lookup_tables = {}
    invh_list = np.asarray(list(
        set([float(x[x.find('_invh') + 6:x.find('.csv')]) for x in os.listdir(gauss_lookuptables_dir) if
             x.find('csv') > 0])))
    GaussianRing_lookup_tables['invh_list'] = invh_list

    for invh in invh_list:
        GaussianRing_lookup_tables[invh] = np.load(os.path.join(gauss_lookuptables_dir, "Gauss_invh_%2.2f.npy" % invh))

    return GaussianRing_lookup_tables

def load_gaussian_bt_tables():

    GaussianRing_BTmin_lookup_tables = {}
    invh_list = np.asarray(list(
        set([float(x[x.find('_invh') + 6:x.find('.npy')]) for x in os.listdir(gauss_BTmin_lookuptables_dir) if
             x.find('npy') > 0])))
    GaussianRing_BTmin_lookup_tables['invh_list'] = invh_list

    for invh in invh_list:
        GaussianRing_BTmin_lookup_tables[invh] = np.load(
            os.path.join(gauss_BTmin_lookuptables_dir, "Gauss_BTmin_invh_%2.2f.npy" % invh))

    return GaussianRing_BTmin_lookup_tables

Noordermeer_lookup_tables = load_noordermeer_tables()
GaussianRing_lookup_tables = load_gaussian_tables()
GaussianRing_BTmin_lookup_tables = load_gaussian_bt_tables()


# ------------------------------------ General functions ------------------------------------

class cosmology:
    def __init__(self, z, H0=H0, omega_m=0.3, omega_lamda=0.7):
        self.H0 = H0
        self.z = z
        self.omega_m = omega_m
        self.omega_lamda = omega_lamda
        self.H = self.hubble_at_z(H0, omega_m, omega_lamda, z)
        self.h = H0 / (10 ** 5 / (1e3*kpc))
        self.rho_c = 3 * self.H ** 2 / (8 * pi * G)

        if isinstance(z, float) or isinstance(z, int):
            self.angular_distance = const.c.value / (1 + z) * scp_integrate.quad(lambda x: 1 / self.hubble_at_z(H0, self.omega_m, self.omega_lamda, x), 0, z)[0]
            self.kpc2arcsec = self.angular_distance * np.deg2rad(1 / 3600) / kpc
            self.luminosity_distance = (1 + z)**2 * self.angular_distance

    def hubble_at_z(self, H0, omega_m, omega_lamda, z):
        return H0 * np.sqrt(omega_lamda + omega_m * np.power(1 + z, 3) + (1 - omega_m - omega_lamda) * np.power(1 + z, 2))

def extension(file):
    if file.find(".") < 0:
        return None
    else:
        return file[file.find(".") + 1:]

def open_file(path):
    if extension(path) == "csv":
        df = pd.read_csv(path, index_col=0)
    elif extension(path) == "xlsx" or extension(path) == "xls":
        df = pd.read_excel(path, index_col=0)
    else:
        raise Exception("file extension is not 'csv' or 'xlsx'")

    return df

def solve_numerical_using_brentq(func, p0, p1_oom=4., N=1000, verbose=False, last=False):
    """
    :param func: function to find 0 value for
    :param p0: where the solution should be
    :return: zero point solution for func
    """

    pvalue = None
    p0 *= 1.
    sign_at_initial = np.sign(func(p0))
    go_higher = False
    higher_values = np.logspace(0., p1_oom, num=N) * p0
    lower_values = np.logspace(0., -p1_oom, num=N) * p0
    higher_values_sign = np.sign(func(higher_values))
    lower_values_sign = np.sign(func(lower_values))

    # check if the sign changes at higher values than p0
    if any(list(higher_values_sign == - sign_at_initial)):
        p1 = next(x for x in higher_values if np.sign(func(x)) == -sign_at_initial)
        go_higher = True

    # check if the sign changes at lower values than p0
    elif any(list(lower_values_sign == - sign_at_initial)):
        p1 = next(x for x in lower_values if np.sign(func(x)) == -sign_at_initial)

    else:
        if not last:
            pvalue = solve_numerical_using_brentq(func=func, p0=-p0, p1_oom=p1_oom, last=True)
        else:
            X1 = np.logspace(-p1_oom, p1_oom, num=N) * p0
            X2 = np.logspace(-p1_oom, p1_oom, num=N) * -p0
            Y1 = func(X1)
            Y2 = func(X2)
            plt.figure()
            plt.plot(X1 / p0, Y1)
            plt.plot(X2 / p0, Y2)
            plt.xlabel('values / p0')
            plt.ylabel('func')
            plt.show()
            raise Exception('Brentq solver couldnt find two points with different signs...')

    if verbose:
        print('Went %s' % ('higher' if go_higher else 'lower'))
        print('Bounds: %s %s (%s %s)' % (p0, p1, np.sign(func(p0)), np.sign(func(p1))))

    if pvalue is None:
        pvalue = slv.brentq(func, p0, p1)

    return pvalue

def integrate_an_array(func, lower_lim, upper_lim):
    if isinstance(lower_lim, float) or isinstance(lower_lim, int):
        if isinstance(upper_lim, float) or isinstance(upper_lim, int):
            output = scp_integrate.quad(func, a=lower_lim, b=upper_lim)[0]
        else:
            output = np.zeros_like(upper_lim)
            for i in range(len(upper_lim)):
                output[i] = scp_integrate.quad(func, a=lower_lim, b=upper_lim[i])[0]
    elif isinstance(upper_lim, float) or isinstance(upper_lim, int):
        output = np.zeros_like(lower_lim)
        for i in range(len(lower_lim)):
            output[i] = scp_integrate.quad(func, a=lower_lim[i], b=upper_lim)[0]

    return np.asarray(output)

def gaussian(x, sig, mu, A=1):
    return A * np.exp(-0.5 * np.power((x - mu) / sig, 2.))

def create_r_space(edge, resolution, scale=kpc):
    plus = np.arange(start=resolution, stop=edge + resolution, step=resolution)
    minus = -plus[::-1]
    R = np.array([*minus, 0, *plus]) * scale

    return R

def moving_average(x, y, w, smooth=False):

    df = pd.DataFrame(data=np.array([x, y]).transpose(), columns=['x', 'y'])
    df = df.sort_values('x', ascending=True)

    x = np.asarray(df['x'])
    y = np.asarray(df['y'])

    x_average = []
    y_average = []

    N = len(x)
    for idx in range(N-w):
        x_idx = x[idx:idx+w]
        y_idx = y[idx:idx+w]

        x_average.append(np.sum(x_idx) / w)
        y_average.append(np.sum(y_idx) / w)

    x_average = np.asarray(x_average)
    y_average = np.asarray(y_average)

    if smooth:
        y_average = scipy.ndimage.gaussian_filter(y_average, sigma=w/3)

    return x_average, y_average

def find_maximum_frequency(data_array, bins, axis=0):
    '''
    find the most frequent value in a binned histogram of an array.
    works on a columns-basis in the given data_array.
    '''

    N = data_array.shape[1]
    argmax_array = [np.argmax(np.histogram(data_array[:, i], bins=bins)[0]) for i in range(N)]
    max_freq_lower_values = [np.histogram(data_array[:, i], bins=bins)[1][argmax_array[i]] for i in range(N)]
    max_freq_upper_values = [np.histogram(data_array[:, i], bins=bins)[1][argmax_array[i] + 1] for i in range(N)]
    max_freq_values = np.average([max_freq_lower_values, max_freq_upper_values], axis=0)

    return max_freq_values

def sersic_b(n):
    return scp_functions.gammaincinv(2 * n, 0.5)
# ------------------------------------ Functions for plotting ------------------------------------

colors_list = sns.color_palette('deep', n_colors=10)
colors = {
    'blue': colors_list[0],
    'orange': colors_list[1],
    'green': colors_list[2],
    'red': colors_list[3],
    'purple': colors_list[4],
    'brown': colors_list[5],
    'pink': colors_list[6],
    'grey': colors_list[7],
    'beige': colors_list[8],
    'teal': colors_list[9],
}

def make_pretty_plot(dpi=600):
    """
    Use first thing after setting up the plt.figure() to make things pretty.
    """
    mpl.rcParams["figure.dpi"] = dpi

    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['xtick.major.top'] = True
    mpl.rcParams['xtick.minor.top'] = True
    mpl.rcParams['xtick.major.bottom'] = True
    mpl.rcParams['xtick.minor.bottom'] = True
    mpl.rcParams['xtick.direction'] = 'in'

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
