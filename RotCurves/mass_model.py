import numpy as np
import logging

from RotCurves.dm_halos import alhpaNFWHalo
from RotCurves.baryons import FreemanDisk, SersicProfile, GaussianRingProfile, LightFreemanDiskProfile, LightSersicProfile, LightGaussianRingProfile
from RotCurves.dm_halos import NFWHalo, BurkertHalo, EinastoHalo, DekelZhaoHalo

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')


def create_components(include_halo=False, include_disk=False, include_ring=False, include_bulge=False,
                      z=None, halo_profile='NFW', logM_vir=None, c=None, alpha=1., beta=3., gamma=1., AC=False,
                      logM_baryon=None, DT=None, disk_re=None, disk_n=1.0, disk_q=0.2, disk_lw=True,
                      BT=None, bulge_n=4.0, bulge_q=1.0, bulge_lw=False,
                      ring_rpeak=None, ring_FWHM=None, ring_lw=False,
                      running_in_cluster=False, apply2D=True):
    """
    :param mass_components_switches: dictionary of component switches    [halo, disk, ring, bulge]
    :param z: redshift
    :param halo_profile: type of halo to use                    [NFW (default), Burkert, Einasto, Dekel-Zhao]
    :param logM_vir: log virial mass                            [Msol]
    :param c: halo concentration parameter                      [dimless]
    :param alpha: halo inner slope                              [def = 1]
    :param beta: halo outer slope                               [def = 3]
    :param gamma: halo transition between slopes                [def = 1]
    :param AC: include adiabatic contraction                    [True / False]
    :param logM_baryon: log baryonic mass                       [Msol]
    :param DT: disk-to-total ratio                              [dimless]
    :param disk_re: disk effective radius                       [kpc]
    :param disk_n: disk Sersic index                            [def 1.0]
    :param disk_q: disk axis-ratio                              [def 0.2]
    :param disk_lw: whether to apply light weighting            [True / False]
    :param BT: bulge-to-total ratio                             [dimless]
    :param bulge_n: bulge Sersic index                          [def 4.0]
    :param bulge_q: bulge axis ratio                            [def 1.0 - spherical]
    :param bulge_lw: whether to apply light weighting           [True / False]
    :param ring_rpeak: ring peak radius                         [kpc]
    :param ring_FWHM: Full-width half-maximum of Gaussian ring  [kpc]
    :param ring_re: ring effective radius                       [kpc]
    :param ring_invh: ring_Rpeak / ring_FWHM                    [dimless]
    :param ring_lw: whether to apply light weighting            [True / False]
    :param apply2D: use a 2D grid for the RC                    [True / False]
    :param running_in_cluster: if running in CCA cluster        [True / False]

    :return: disk, bulge, halo Objects
    """

    halo = None
    disk = None
    ring = None
    bulge = None

    if include_disk:
        if include_ring:
            RT = 1 - BT - DT
        else:
            RT = 0.
            DT = 1 - BT
    else:
        DT = 0.
        if include_ring:
            if include_bulge:
                RT = 1 - BT
            else:
                RT = 1
                BT = 0
        else:
            RT = 0.

    if include_ring and BT == 0:
        logM_ring_tmp = logM_baryon + np.log10(RT)
        ring_tmp = GaussianRingProfile(mass=np.power(10, logM_ring_tmp),
                                  r_s=ring_rpeak,
                                  FWHM_ring=ring_FWHM,
                                  mass_to_light=float(ring_lw))
        # TODO: add the method of finding the minimal bulge to the GaussianRingProfile class
        # ring_tmp.find_minimal_bulge()
        # BT = ring_tmp.BT_min
        # include_bulge = True

    if RT < 0:
        logger.warning('Mass components: Negative ring mass with: BT=%0.2f, DT=%0.2f, RT=%0.2f' % (BT, DT, RT))

    if DT < 0:
        logger.warning('Mass components: Negative disk mass with: BT=%0.2f, DT=%0.2f, RT=%0.2f' % (BT, DT, RT))

    if include_halo:
        halo = alhpaNFWHalo(z=z,
                            mass=np.power(10, logM_vir),
                            concentration=c,
                            alpha=alpha,
                            adiabatic_contraction=AC)

    if include_disk:
        logM_disk = logM_baryon + np.log10(DT)
        disk = SersicProfile(mass=np.power(10, logM_disk),
                            r_eff=disk_re,
                            n=disk_n,
                            q0=disk_q,
                            mass_to_light=float(disk_lw))
    if not include_disk:
        if disk_lw:
            disk = LightSersicProfile(r_eff=disk_re,
                                      n=disk_n)

    if include_ring:
        logM_ring = logM_baryon + np.log10(RT)
        ring = GaussianRingProfile(mass=np.power(10, logM_ring),
                                  r_s=ring_rpeak,
                                  FWHM_ring=ring_FWHM,
                                  mass_to_light=float(ring_lw))
    if not include_ring:
        if ring_lw:
            ring = LightGaussianRingProfile(r_s=ring_rpeak, FWHM_ring=ring_FWHM)

    if include_bulge:
        logM_bulge = logM_baryon + np.log10(BT)
        bulge = SersicProfile(mass=np.power(10, logM_bulge),
                             r_eff=1.,
                             n=bulge_n,
                             q0=bulge_q,
                             mass_to_light=float(bulge_lw))

    mass_components = {'halo': halo, 'disk': disk, 'ring': ring, 'bulge': bulge}

    return mass_components


