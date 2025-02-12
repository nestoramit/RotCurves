import numpy as np
from scipy.interpolate import CubicSpline


def Mvir_Moster2018(z, log_mstar):
    z1 = 1 + z
    n = np.power(10, 1.507 - 0.124 * z / z1)
    b = -0.621 - 0.059 * z / z1
    g = 1.055 + 0.838 * z / z1 - 3.083 * (z / z1)**2

    log_mvir = log_mstar + np.log10(0.5) + np.log10(n) + np.log10((10**(log_mstar - 10.6))**b + (10**(log_mstar - 10.6))**g)

    return log_mvir


def c_Dutton2014(z, log_Mvir):
    a = 0.520 + np.exp(-0.617 * z**1.21) * (0.905 - 0.520)
    b = -0.101 + 0.026*z

    logc = a + b * (log_Mvir - 12.)
    c = 10 ** logc

    return c


def Reff_VanDerWel2014(z, log_Mstar):
    zs = np.asarray([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    logAs = np.asarray([0.86, 0.78, 0.70, 0.65, 0.55, 0.51])
    alphas = np.asarray([0.25, 0.22, 0.22, 0.23, 0.22, 0.18])
    logReffs = logAs + alphas * (log_Mstar - np.log10(5e10))
    Reffs = 10 ** logReffs

    if z in zs:
        return Reffs[list(zs).index(z)]

    else:
        zclose_idx = np.argmin(np.abs(np.asarray(zs)-z))

        if zclose_idx == 0:
            return Reffs[zclose_idx]

        elif zclose_idx == len(zs)-1:
            return Reffs[zclose_idx]

        else:
            if zs[zclose_idx] > z:
                zhiger_idx = zclose_idx
                zlower_idx = zclose_idx-1
            else:
                zhiger_idx = zclose_idx+1
                zlower_idx = zclose_idx

            zlower = zs[zlower_idx]
            zhiger = zs[zhiger_idx]

            logReff_lower = Reff_VanDerWel2014(zlower, log_Mstar)
            logReff_higher = Reff_VanDerWel2014(zhiger, log_Mstar)

            interpolator = CubicSpline(x=[zlower, zhiger], y=[logReff_lower, logReff_higher])
            Reff = float(interpolator(z))

            return Reff


def Reff_VanDerWel2014b(z, log_Mstar):
    # take from Tacconi+2018
    Reff = 8.9 * (1+z)**-0.75 * (10**log_Mstar / 5e10)**0.22

    return Reff


def Reff_Mowla2018(z, log_Mstar):
    logA = -0.29*np.log10(1+z) + 0.91
    alpha = -0.15*np.log10(1+z) + 0.31
    logReff = logA + alpha * (log_Mstar - np.log10(7e10))
    Reff = 10**logReff

    return Reff


def Reff_Mo1998(Rvir, lamda=0.04, specific_angular_momentum=1.):
    Rd_expo =  1/np.sqrt(2) * specific_angular_momentum * lamda * Rvir

    return 1.68 * Rd_expo


def BT_Lang2014(log_Mstar):
    # my fit to SFGs in Lang+14
    G = 0.247
    H = 0.217

    BT = G + H*(log_Mstar-10.3)

    return np.maximum(BT, 0)


def Mgas_to_Mstar_Tacconi2018(z, log_Mstar, Reff=None, delMS=1):
    """
    Taken from Tacconi et al. 2018
    uses BEST beta=2, S14

    """
    beta = 2.
    A = 0.12
    B = -3.62
    F = 0.66
    C = 0.53
    D = -0.35
    E = 0.11

    if Reff is None:
        Reff = Reff_VanDerWel2014b(z, log_Mstar)

    logMgas_to_Mstar = A + B*(np.log10(1+z)-F)**beta + C*np.log10(delMS) + D*(log_Mstar-10.7) + E*np.log10(Reff/Reff_VanDerWel2014b(z, log_Mstar))
    Mgas_to_Mstar = 10**logMgas_to_Mstar

    return Mgas_to_Mstar


def f_gas_Tacconi2018(z, log_Mstar, Reff=None, delMS=1):
    Mgas_to_Mstar = Mgas_to_Mstar_Tacconi2018(z, log_Mstar, Reff, delMS)
    fgas = Mgas_to_Mstar / (1 + Mgas_to_Mstar)

    return fgas


def sigma0_Ubler2019(z):
    # Taken from Ubler+2019
    # redshift dependence only
    # based on KMOS3D

    return 21 + 11.3*z


def MS_Whitaker2014(z, logMstar):
    # Whitaker+2014
    # logSFR = a + b*log(Mstar) + c*log(Mstar)^2
    # logMstar in Msol

    if z <= 1.0:
        a, b, c = -27.40, 5.02, -0.22
    elif 1.0 < z < 1.5:
        a, b, c = -26.03, 4.62, -0.19
    elif 1.5 <= z < 2.0:
        a, b, c = -24.04, 4.17, -0.16
    elif 2.0 <= z <= 2.5:
        a, b, c = -19.99, 3.44, -0.13

    logSFR = a + b*logMstar + c*logMstar**2

    return logSFR
