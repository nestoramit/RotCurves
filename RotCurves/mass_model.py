import numpy as np
import logging

from RotCurves.dm_halos import alhpaNFWHalo
from RotCurves.baryons import FreemanDisk, SersicProfile, GaussianRingProfile, LightFreemanDiskProfile, LightSersicProfile, LightGaussianRingProfile
from RotCurves.dm_halos import halo_selector

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')


def create_components(
        include_halo=False,
        include_disk=False,
        include_ring=False,
        include_bulge=False,
        z=None,
        halo_profile='NFW',
        logM_vir=None,
        c=None,
        alpha=1.,
        beta=3.,
        gamma=1.,
        AC=False,
        logM_baryon=None,
        DT=None,
        disk_re=None,
        disk_n=1.0,
        disk_q=0.2,
        disk_lw=False,
        BT=None,
        bulge_n=4.0,
        bulge_q=1.0,
        bulge_lw=False,
        ring_rpeak=None,
        ring_FWHM=None,
        ring_lw=False):
    """
    Create mass model components (halo, disk, ring, bulge) for galaxy rotation curve modeling.

    Parameters
    ----------
    include_halo : bool, optional
        Whether to include a dark matter halo component. Default is False.
    include_disk : bool, optional
        Whether to include a disk component. Default is False.
    include_ring : bool, optional
        Whether to include a ring component. Default is False.
    include_bulge : bool, optional
        Whether to include a bulge component. Default is False.
    z : float, optional
        Redshift used for halo calculations. Default is None.
    halo_profile : str, optional
        Type of halo profile to use. Options: 'NFW', 'Burkert', 'Einasto', 'Dekel-Zhao'.
        Default is 'NFW'.
    logM_vir : float, optional
        Logarithm of virial halo mass in solar masses. Required if include_halo=True.
    c : float, optional
        Halo concentration parameter. Required if include_halo=True.
    alpha : float, optional
        Halo inner slope parameter (for alpha-NFW and Dekel-Zhao profiles). Default is 1.0.
    beta : float, optional
        Unused parameter (kept for backward compatibility). For Dekel-Zhao, use 'g' instead.
    gamma : float, optional
        Unused parameter (kept for backward compatibility). For Dekel-Zhao, use 'b' instead.
    AC : bool, optional
        Whether to include adiabatic contraction (not yet implemented). Default is False.
    logM_baryon : float, optional
        Logarithm of total baryonic mass in solar masses. Required for baryonic components.
    DT : float, optional
        Disk-to-total baryonic mass ratio. Default is None.
    disk_re : float, optional
        Disk effective radius in kpc. Required if include_disk=True or disk_lw=True.
    disk_n : float, optional
        Disk Sersic index. Default is 1.0 (exponential disk).
    disk_q : float, optional
        Disk axis ratio (0 < q <= 1). Default is 0.2.
    disk_lw : bool, optional
        Whether to create a light-weighted disk (zero mass). Default is False.
    BT : float, optional
        Bulge-to-total baryonic mass ratio. Default is None.
    bulge_n : float, optional
        Bulge Sersic index. Default is 4.0 (de Vaucouleurs profile).
    bulge_q : float, optional
        Bulge axis ratio (0 < q <= 1). Default is 1.0 (spherical).
    bulge_lw : bool, optional
        Whether to create a light-weighted bulge (zero mass). Default is False.
    ring_rpeak : float, optional
        Ring peak radius in kpc. Required if include_ring=True or ring_lw=True.
    ring_FWHM : float, optional
        Full-width half-maximum of Gaussian ring in kpc. Required if include_ring=True or ring_lw=True.
    ring_lw : bool, optional
        Whether to create a light-weighted ring (zero mass). Default is False.

    Returns
    -------
    dict
        Dictionary containing mass components with keys: 'halo', 'disk', 'ring', 'bulge'.
        Components that are not created will be None.

    Notes
    -----
    Mass ratios (BT, DT) are automatically clipped to the range [0.01, 0.99] to avoid
    numerical issues. The ring-to-total ratio (RT) is calculated as RT = 1 - BT - DT.

    For Dekel-Zhao halos, the parameters 'beta' and 'gamma' are ignored. Use 'g' and 'b'
    parameters directly when calling halo_selector (not implemented in this function).
    """

    halo = None
    disk = None
    ring = None
    bulge = None

    # Clip mass ratios to valid range, handling None values
    BT = np.clip(BT, 1e-3, 1-1e-3) if BT is not None else None
    DT = np.clip(DT, 1e-3, 1-1e-3) if DT is not None else None

    # Calculate mass ratios (RT = ring-to-total)
    # Handle None values properly
    BT = BT if BT is not None else 0.0
    DT = DT if DT is not None else 0.0

    if include_disk:
        if include_ring:
            RT = 1 - BT - DT
        else:
            RT = 0.
            if BT is None:
                DT = 1.0
            else:
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

    # Check for negative mass ratios and warn
    if RT < 0:
        logger.warning(
            f'Mass components: Negative ring mass with: BT={BT:.2f}, DT={DT:.2f}, RT={RT:.2f}')

    if DT < 0:
        logger.warning(
            f'Mass components: Negative disk mass with: BT={BT:.2f}, DT={DT:.2f}, RT={RT:.2f}')

    # Create halo component
    if include_halo:
        if logM_vir is None:
            raise ValueError("logM_vir must be provided when include_halo=True")
        if c is None:
            raise ValueError("c (concentration) must be provided when include_halo=True")
        
        # Prepare halo kwargs - only pass relevant parameters
        halo_kwargs = {
            'z': z if z is not None else 0.0,
            'mass': np.power(10, logM_vir),
            'concentration': c,
            'adiabatic_contraction': AC
        }
        
        # Add profile-specific parameters
        if halo_profile.lower() in ['alpha-nfw', 'alpha_nfw', 'alphanfw']:
            halo_kwargs['alpha'] = alpha
        elif halo_profile.lower() in ['dekel-zhao', 'dekel_zhao', 'dekelzhao', 'dz']:
            halo_kwargs['alpha'] = alpha
            halo_kwargs['beta'] = beta
            halo_kwargs['gamma'] = gamma
        halo = halo_selector(component_type=halo_profile, **halo_kwargs)

    # Create disk component
    if include_disk:
        if logM_baryon is None:
            raise ValueError("logM_baryon must be provided when include_disk=True")
        if disk_re is None:
            raise ValueError("disk_re must be provided when include_disk=True")
        if DT is None or DT <= 0:
            raise ValueError("DT must be provided and > 0 when include_disk=True")
        
        logM_disk = logM_baryon + np.log10(DT)
        disk = SersicProfile(mass=np.power(10, logM_disk),
                            r_eff=disk_re,
                            n=disk_n,
                            q0=disk_q,
                            mass_to_light=float(disk_lw))
    elif disk_lw:
        # Create light-weighted disk (zero mass)
        if disk_re is None:
            raise ValueError("disk_re must be provided when disk_lw=True")
        disk = LightSersicProfile(r_eff=disk_re, n=disk_n)

    # Create ring component
    if include_ring:
        if logM_baryon is None:
            raise ValueError("logM_baryon must be provided when include_ring=True")
        if ring_rpeak is None:
            raise ValueError("ring_rpeak must be provided when include_ring=True")
        if ring_FWHM is None:
            raise ValueError("ring_FWHM must be provided when include_ring=True")
        if RT <= 0:
            raise ValueError("Ring-to-total ratio RT must be > 0 when include_ring=True. "
                           f"Current RT={RT:.3f} (BT={BT:.3f}, DT={DT:.3f})")
        
        logM_ring = logM_baryon + np.log10(RT)
        ring = GaussianRingProfile(mass=np.power(10, logM_ring),
                                  r_s=ring_rpeak,
                                  FWHM_ring=ring_FWHM,
                                  mass_to_light=float(ring_lw))
    elif ring_lw:
        # Create light-weighted ring (zero mass)
        if ring_rpeak is None:
            raise ValueError("ring_rpeak must be provided when ring_lw=True")
        if ring_FWHM is None:
            raise ValueError("ring_FWHM must be provided when ring_lw=True")
        ring = LightGaussianRingProfile(r_s=ring_rpeak, FWHM_ring=ring_FWHM)

    # Create bulge component
    if include_bulge:
        if logM_baryon is None:
            raise ValueError("logM_baryon must be provided when include_bulge=True")
        if BT is None or BT <= 0:
            raise ValueError("BT must be provided and > 0 when include_bulge=True")
        
        logM_bulge = logM_baryon + np.log10(BT)
        bulge = SersicProfile(mass=np.power(10, logM_bulge),
                             r_eff=1.,
                             n=bulge_n,
                             q0=bulge_q,
                             mass_to_light=float(bulge_lw))

    mass_components = {'halo': halo, 'disk': disk, 'ring': ring, 'bulge': bulge}

    return mass_components


