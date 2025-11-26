import numpy as np
import time
import logging
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.signal import windows

from RotCurves.baryons import *
from RotCurves.dm_halos import *
from RotCurves.base_utils import (
    create_r_space,
    safe_sqrt,
    )

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')


class RotationCurveObject:
    r"""
    Rotation curve calculator with beam smearing and pressure support corrections.

    This class computes rotation curves for galaxy models composed of baryonic
    components (disk, bulge, and/or ring) and a dark matter halo. It accounts for
    observational effects including beam smearing, inclination, and pressure support corrections.

    The rotation curve is computed as:

    .. math::

       v_{\rm rot}^2(r) = v_{\rm circ}^2(r) + v_{\sigma}^2(r),

    where :math:`v_{\rm circ}^2 = v_{\rm baryon}^2 + v_{\rm halo}^2` is the
    circular velocity squared, and :math:`v_{\sigma}^2` is the pressure support
    correction. The observed line-of-sight velocity is convolved over the PSF beam kernel:

    .. math::

       v_{\rm obs}(r) = v_{\rm rot}(r) \sin(i) \otimes PSF(r),

    where :math:`i` is the inclination angle and :math:`PSF` a circular
    Gaussian PSF beam kernel.

    The pressure support correction accounts fact that gas velocity
    dispersion (in most cases) reduces the observed rotation velocity below the circular
    velocity (Burkert et al. 2010):

    .. math::

       v_{\sigma}^2 = 2 \sigma^2 \sum_{i} \frac{\mathrm{d}\ln\Sigma_i}{\mathrm{d}\ln r},
    

    where :math:`\sigma` is the velocity dispersion and the logarithmic slope
    is computed from the sum of the disk and ring components surface densities.

    Parameters
    ----------
    galaxy : object, optional
        Galaxy object containing spatial grid parameters (``dx``, ``edge``,
        ``oversample``, ``oversample_edge``, ``sigma_inst``). If provided,
        these parameters override individual arguments.
    edge : float, optional
        Maximum radius for the rotation curve calculation [kpc]. Required if
        ``galaxy`` and ``rarray`` are not provided.
    dx : float, optional
        Spatial resolution (pixel size) [kpc]. Default is 0.1 kpc if not
        specified.
    rarray : array_like, optional
        Custom radial array for evaluation [kpc]. If provided, ``edge`` and
        ``dx`` are inferred from this array.
    sigma_inst : float, optional
        Instrumental velocity dispersion [km/s]. Default is 0.
    oversample : float, optional
        Oversampling factor for spatial resolution. The effective pixel size
        becomes ``dx / oversample``. Default is 1.
    oversample_edge : float, optional
        Number of beam widths to extend the sampling region beyond ``edge``
        for beam smearing calculations. Default is 4.
    Halo : object, optional
        Dark matter halo profile instance (e.g., :class:`NFWHalo`).
    Disk : object, optional
        Disk baryonic component (e.g., :class:`FreemanDisk` or
        :class:`SersicProfile`).
    Ring : object, optional
        Ring baryonic component (e.g., :class:`GaussianRingProfile`).
    Bulge : object, optional
        Bulge baryonic component (e.g., :class:`SersicProfile` with :math:`n=4`).
    sigma_dispersion : float, optional
        Central velocity dispersion for pressure support calculation [km/s].
        Default is ``None`` (no pressure support).
    dispersion_function : str, optional
        Functional form for the velocity dispersion profile. Options:
        - ``'const'`` or ``'constant'``: Constant dispersion
        - ``'constant_h'`` or ``'constant_height'``: Dispersion proportional to
          square root of surface density
        - ``'power_law'``: Power-law radial dependence
        Default is ``'const'``.
    pressure_support : str, optional
        Method for calculating pressure support correction. Options:
        - ``'general'`` or ``'burkert'``: General formula using logarithmic
          density slope (Burkert et al. 2010)
        - ``'exponential'``: Simplified formula for exponential profiles
        Default is ``'general'``.
    inclination : float, optional
        Inclination angle of the galaxy [degrees]. :math:`i=90^\circ` is
        edge-on, :math:`i=0^\circ` is face-on. Default is 90.
    PA : float, optional
        Position angle of the major axis [degrees]. Used for 2D rotation
        curve calculations. Default is 0.
    sigma_beam : float, optional
        Standard deviation of the Gaussian beam [kpc]. If not provided but
        ``FWHM_beam`` is given, it is calculated as
        :math:`\sigma_{\rm beam} = \mathrm{FWHM} / (2\sqrt{2\ln 2})`.
    FWHM_beam : float, optional
        Full Width at Half Maximum of the Gaussian beam [kpc]. If provided,
        ``sigma_beam`` is calculated automatically.
    apply_2D : bool, optional
        If ``True``, compute 2D beam smearing on a spatial grid. If ``False``,
        use 1D beam smearing along the major axis. Default is ``True``.
    include_beam_smearing : bool, optional
        If ``True``, apply beam smearing to the rotation curve. If ``False``,
        only compute the intrinsic (unsmeared) rotation curve. Default is
        ``True``.
    printtime : bool, optional
        If ``True``, print computation time. Default is ``False``.
    ndim : float, optional
        Output dimensionality. ``1`` for 1D major-axis extraction,
        ``2`` for full 2D velocity field. Default is 1.
    radial_velocity : float, optional
        Radial velocity component (inflow/outflow) [km/s]. Default is 0.

    Attributes
    ----------
    R_majoraxis : ndarray
        Radial array along the major axis [kpc].
    intrinsic : ndarray
        Intrinsic rotation velocity including pressure support [km/s].
    intrinsic_with_inclination : ndarray
        Intrinsic line-of-sight velocity [km/s].
    intrinsic_no_dispersion : ndarray
        Intrinsic rotation velocity without pressure support [km/s].
    intrinsic_no_dispersion_with_inclination : ndarray
        Intrinsic line-of-sight velocity without pressure support [km/s].
    smeared : ndarray
        Beam-smeared rotation velocity [km/s].
    smeared_with_inclination : ndarray
        Beam-smeared line-of-sight velocity [km/s].
    velocity_dispersion : ndarray
        Total velocity dispersion including beam smearing and instrumental
        effects [km/s].
    V2baryon : ndarray
        Squared velocity contribution from baryonic components [km²/s²].
    V2h : ndarray
        Squared velocity contribution from dark matter halo [km²/s²].
    V2circ : ndarray
        Squared circular velocity [km²/s²].
    V2rot : ndarray
        Squared rotation velocity including pressure support [km²/s²].
    fdm : ndarray
        Dark matter fraction :math:`f_{\rm DM} = v_{\rm halo}^2 / v_{\rm circ}^2`.

    Notes
    -----
    

    where :math:`\sigma` is the velocity dispersion and the logarithmic slope
    is computed from the density profile.

    Beam smearing is implemented using a Gaussian convolution kernel that
    accounts for the finite spatial resolution of observations. The kernel
    is light-weighted using the combined light profiles of all baryonic
    components.

    References
    ----------
    Burkert, A., et al. 2010, ApJ, 725, 2324

    Examples
    --------
    Create a rotation curve for a galaxy with an exponential disk and NFW halo:

    >>> from RotCurves.baryons import FreemanDisk
    >>> from RotCurves.dm_halos import NFWHalo
    >>> disk = FreemanDisk(mass=1e10, r_s=2.0)
    >>> halo = NFWHalo(mass=1e12, concentration=10)
    >>> rc = RotationCurveObject(
    ...     edge=20.0, dx=0.1, Disk=disk, Halo=halo,
    ...     inclination=60, sigma_beam=0.5, include_beam_smearing=True
    ... )
    >>> v_obs = rc.smeared_with_inclination  # Line-of-sight velocity
    """
    def __init__(self,
                 galaxy=None,
                 edge=None,
                 dx=None,
                 rarray=None,
                 sigma_inst=0.,
                 oversample=1.,
                 oversample_edge=4.,
                 Halo=None,
                 Disk=None,
                 Ring=None,
                 Bulge=None,
                 sigma_dispersion=0.,
                 dispersion_function='const',
                 pressure_support="general",
                 inclination=None,
                 PA=0.,
                 sigma_beam=None,
                 FWHM_beam=None,
                 apply_2D=True,
                 include_beam_smearing=True,
                 printtime=False,
                 ndim=1.,
                 radial_velocity=0):
        
        # Initialize timing if requested
        if printtime:
            self.starttime = time.time()
        
        # Store all parameters directly
        # Mass components
        self.galaxy = galaxy
        self.halo = Halo
        self.disk = Disk
        self.ring = Ring
        self.bulge = Bulge
        
        # Configuration parameters
        self.inclination = inclination
        self.PA = PA
        self.ndim = ndim
        self.sigma0 = sigma_dispersion
        self.dispersion_function = dispersion_function
        self.pressure_support = pressure_support
        self.Vradial = radial_velocity
        
        # Beam and sampling parameters
        self.sigma_beam = sigma_beam
        self.FWHM_beam = FWHM_beam
        self.apply_2D = apply_2D
        self.include_beam_smearing = include_beam_smearing
        self.oversample = oversample
        self.oversample_edge = oversample_edge
        self.sigma_inst = sigma_inst
        
        # Process and validate parameters
        self._process_beam_conversion()
        self._process_galaxy_overrides()
        self._set_parameter_defaults()
        
        # Setup spatial grid
        self._setup_spatial_grid(rarray, edge, dx)
        
        # Setup derived beam parameters
        self._setup_derived_beam_parameters()
        
        # Keep the existing beam smearing calculation code unchanged
        ### only calculate intrinsic RC
        if not self.include_beam_smearing:
            if self.inclination is None:
                self.inclination = 90.
                logger.warning("Rotation Curve: no inclination set, assuming inc=90 for the calculation.")
            self.make_intrinsic_rotationCurve(self.R_majoraxis)

        ### beam-smeared RC
        else:
            if self.inclination is None:
                logger.warning("Rotation Curve: no inclination set, cannot perform beam smearing.")
            ### do a 1D rotation curve ###
            if not self.apply_2D:
                # define the 1D radial spaces
                self.sampling_edge = self.edge + self.oversample_edge * self.sigma_beam
                self.oversample_pixels = int(round(self.oversample_edge * self.sigma_beam / self.dx))
                self.sampling_rarray_1D = create_r_space(edge=self.sampling_edge, resolution=self.dx)

                # create intrinsic RC for the sampling range
                self.make_intrinsic_rotationCurve(self.sampling_rarray_1D)

                # create beam-smeared RC along the major axis
                self.smeared = self.apply_1D_beam_smearing(self.intrinsic)
                self.smeared_with_inclination = self.apply_1D_beam_smearing(self.intrinsic_with_inclination)
                self.smeared_no_dispersion = self.apply_1D_beam_smearing(self.intrinsic_no_dispersion)
                self.smeared_no_dispersion_with_inclination = self.apply_1D_beam_smearing(self.intrinsic_no_dispersion_with_inclination)

                # estimate velocity dispersion
                V_average = self.smeared_with_inclination
                V_sqaured = self.intrinsic_with_inclination ** 2
                V_sqaured_average = self.apply_1D_beam_smearing(V_sqaured)
                dispersion_squared = V_sqaured_average - V_average**2
                self.velocity_dispersion = np.sqrt(dispersion_squared + self.sigma_profile ** 2)

                # choose major axis only from the intrinsic curves
                N = len(self.intrinsic)
                self.intrinsic = self.intrinsic[self.oversample_pixels:N-self.oversample_pixels]
                self.intrinsic_with_inclination = self.intrinsic_with_inclination[self.oversample_pixels:N-self.oversample_pixels]
                self.intrinsic_no_dispersion = self.intrinsic_no_dispersion[self.oversample_pixels:N-self.oversample_pixels]
                self.intrinsic_no_dispersion_with_inclination = self.intrinsic_no_dispersion_with_inclination[self.oversample_pixels:N-self.oversample_pixels]

                if printtime:
                        print('time for 1D:', np.round(time.time() - self.starttime, 1))

            ### do a 2D rotation curve ###
            if apply_2D:
                geometrical_factor_elliptical = 1 / np.cos(np.deg2rad(self.inclination))

                self.sigma_beam_x = self.sigma_beam
                self.sigma_beam_y = self.sigma_beam * geometrical_factor_elliptical
                self.sigma_beam_pixels_x = self.sigma_beam_x / self.dx
                self.sigma_beam_pixels_y = self.sigma_beam_y / self.dx
                self.oversample_edge_pixels_x = int(np.ceil(self.oversample_edge * self.sigma_beam_pixels_x))
                self.oversample_edge_pixels_y = int(np.ceil(self.oversample_edge * self.sigma_beam_pixels_y))
                self.sampling_edge_x = self.edge + self.oversample_edge_pixels_x * self.dx
                self.sampling_edge_y = self.edge + self.oversample_edge_pixels_y * self.dx
                self.sampling_rarray_x = create_r_space(edge=self.sampling_edge_x, resolution=self.dx)
                self.sampling_rarray_y = create_r_space(edge=self.sampling_edge_y, resolution=self.dx)
                self.sampling_edge_2D = np.sqrt(self.sampling_edge_x**2 + self.sampling_edge_y**2)
                self.sampling_rarray_2D = create_r_space(edge=self.sampling_edge_2D, resolution=self.dx)

                # create the velocity array used to build the 2D grid from
                self.make_intrinsic_rotationCurve(self.sampling_rarray_2D)

                # create the beam-smeared rotation curve along the major axis
                self.smeared_with_inclination, self.smeared_light_profile = self.apply_2D_beam_smearing(
                    one_dimensional_vel=self.intrinsic_with_inclination,
                    one_dimensional_rarray=self.sampling_rarray_2D,
                    ndim=self.ndim
                )
                self.smeared = self.smeared_with_inclination / np.sin(np.deg2rad(self.inclination))
                V_average = self.smeared_with_inclination
                V_squared = self.intrinsic_with_inclination ** 2
                # V_squared_average, _ = self.apply_2D_beam_smearing(one_dimensional_vel=V_squared,
                #                                                 one_dimensional_rarray=self.sampling_rarray_2D,
                #                                                 ndim=self.ndim, for_dispersion=True)
                V_squared_average, _ = self.apply_2D_beam_smearing(one_dimensional_vel=self.intrinsic_with_inclination,
                                                                one_dimensional_rarray=self.sampling_rarray_2D,
                                                                ndim=self.ndim, for_dispersion=True)
                dispersion_squared = V_squared_average - V_average ** 2
                if self.ndim == 1:
                    dispersion_squared = dispersion_squared.flatten()

                # create the intrinsic curves in the original space
                self.make_intrinsic_rotationCurve(self.R_majoraxis)

                self.velocity_dispersion = np.sqrt(dispersion_squared + self.sigma_profile ** 2 + self.sigma_inst ** 2)

                if self.ndim == 2.:
                    self.smeared_with_inclination = rotate(self.smeared_with_inclination, angle=90-self.PA, reshape=False)
                    self.velocity_dispersion = rotate(self.velocity_dispersion, angle=90-self.PA, reshape=False)

                self.smeared_with_inclination = np.round(self.smeared_with_inclination, 4)
                self.smeared = np.round(self.smeared, 4)
                self.velocity_dispersion = np.round(self.velocity_dispersion, 4)

                if self.ndim == 1:
                    self.smeared_with_inclination = np.reshape(self.smeared_with_inclination, len(self.smeared_with_inclination))
                    self.smeared = np.reshape(self.smeared, len(self.smeared))
                    self.velocity_dispersion = np.reshape(self.velocity_dispersion, len(self.velocity_dispersion))

                if printtime:
                    print('time for 2D:', np. round(time.time() - self.starttime, 1))

    def _process_beam_conversion(self):
        """Convert FWHM to sigma if needed."""
        if self.sigma_beam is None and self.FWHM_beam is not None:
            self.sigma_beam = self.FWHM_beam / (2 * np.sqrt(2 * np.log(2)))

    def _process_galaxy_overrides(self):
        """Override parameters with values from galaxy object if provided."""
        if self.galaxy is not None:
            galaxy_attrs = ['dx', 'edge', 'oversample_edge', 'oversample', 'sigma_inst']
            for attr in galaxy_attrs:
                if hasattr(self.galaxy, attr):
                    setattr(self, attr, getattr(self.galaxy, attr))

    def _set_parameter_defaults(self):
        """Set default values for parameters that might be None."""
        if self.sigma_inst is None:
            self.sigma_inst = 0
        
        if self.oversample_edge is None:
            self.oversample_edge = 4

    def _setup_spatial_grid(self, rarray, edge, dx):
        """Setup the spatial grid for calculations."""
        if rarray is not None:
            self.R_majoraxis = rarray
            self.edge = np.max(rarray)
            if len(rarray) > 1:
                self.dx = np.diff(rarray)[0]
        elif edge is not None:
            self.edge = edge
            if dx is None:
                logger.info('dx not specified. Using default dx = 0.1 [kpc].')
                self.dx = 0.1
            else:
                self.dx = dx
        elif self.galaxy is None:
            raise ValueError("Must provide either 'rarray', 'edge', or 'galaxy' parameter")
        
        # Apply oversampling to pixel scale
        if hasattr(self, 'dx') and self.dx is not None:
            self.dx = self.dx / self.oversample
        
        # Create radial array if not already set from rarray
        if not hasattr(self, 'R_majoraxis'):
            if hasattr(self, 'edge') and hasattr(self, 'dx'):
                self.R_majoraxis = create_r_space(edge=self.edge, resolution=self.dx)
            else:
                raise ValueError("Cannot create radial array: missing 'edge' or 'dx' parameters")

    def _setup_derived_beam_parameters(self):
        """Setup parameters derived from beam settings."""
        if self.sigma_beam is not None and hasattr(self, 'dx') and self.dx is not None:
            self.sigma_beam_pixels = round(self.sigma_beam / self.dx)

    def get_dispersion_profile(self, R_array, functional_form='const'):
        r"""
        Calculate the velocity dispersion profile.

        Computes the radial profile of velocity dispersion :math:`\sigma(r)`
        based on the specified functional form. The dispersion is used to
        calculate pressure support corrections to the rotation curve.

        Parameters
        ----------
        R_array : array_like
            Radial array [kpc] at which to evaluate the dispersion profile.
        functional_form : str, optional
            Functional form for the dispersion profile. Options:
            - ``'const'``, ``'constant'``, or ``'flat'``: Constant dispersion
              :math:`\sigma(r) = \sigma_0`
            - ``'constant_h'``, ``'constant_height'``, or ``'const_h'``:
              Dispersion proportional to square root of surface density:
              :math:`\sigma(r) = \sigma_0 \sqrt{\Sigma(r) / \Sigma_0}`
            - ``'power_law'``: Power-law radial dependence:
              :math:`\sigma(r) = \sigma_0 / (1 + r / r_s)`
            Default is ``'const'``.

        Notes
        -----
        The dispersion profile is stored in ``self.sigma_profile``. The
        ``constant_h`` form assumes that the velocity dispersion scales with
        the square root of the surface density, which is appropriate for
        constant scale height disks.

        The ``power_law`` form is only available when a disk component is
        present, as it uses the disk scale radius :math:`r_s`.
        """
        # Define dispersion functions
        dispersion_functions = {
            ('const', 'constant', 'flat'): self._get_constant_dispersion,
            ('constant_h', 'constant_height', 'const_h'): self._get_surface_density_dispersion,
            ('power_law',): self._get_power_law_dispersion,
        }
        
        # Find matching function
        for forms, func in dispersion_functions.items():
            if functional_form in forms:
                self.dispersion_func = func
                break
        else:
            logger.warning(
                f"Rotation Curve: Pressure Suppport: Unknown dispersion function: {functional_form}."
                "Using constant dispersion.")
            self.dispersion_func = self._get_constant_dispersion
        
        self.sigma_profile = self.sigma0 * self.dispersion_func(R_array)

    def _get_constant_dispersion(self, r):
        """Helper for constant dispersion."""
        return np.ones_like(r)

    def _get_surface_density_dispersion(self, r):
        """Helper for surface density-based dispersion."""
        for component in [self.disk, self.ring]:
            if component is not None:
                return np.sqrt(component.surface_density_dimless(
                    component._normalized_radius(np.abs(r))
                ))
                break
        logger.warning(
            "Rotation Curve: Pressure Suppport: No disk or ring component found with dispersion function: constant_height."
            "Using constant dispersion.")
        return np.ones_like(r)

    def _get_power_law_dispersion(self, r):
        """Helper for power-law dispersion."""
        if self.disk is not None:
            return np.divide(1, 1 + (np.divide(np.abs(r), self.disk.r_s, 
                                              out=np.zeros_like(r), where=r!=0)))
        logger.warning(
            "Rotation Curve: Pressure Suppport: No disk component found with dispersion function: power_law."
            "Using constant dispersion.")
        return np.ones_like(r)

    def make_intrinsic_rotationCurve(self, R_array):
        r"""
        Calculate the intrinsic (unsmeared) rotation curve.

        Computes the rotation curve from all mass components (disk, ring, bulge,
        halo) and applies pressure support corrections. The method calculates:

        1. Individual velocity contributions from each component
        2. Combined baryonic and dark matter circular velocities
        3. Pressure support correction from gas velocity dispersion
        4. Final rotation velocity including all corrections

        Parameters
        ----------
        R_array : array_like
            Radial array [kpc] at which to evaluate the rotation curve.

        Notes
        -----
        The method computes several velocity components:

        - **Baryonic velocity**: :math:`v_{\rm baryon}^2 = v_{\rm disk}^2 +
          v_{\rm bulge}^2 + v_{\rm ring}^2`
        - **Circular velocity**: :math:`v_{\rm circ}^2 = v_{\rm baryon}^2 +
          v_{\rm halo}^2`
        - **Pressure support**: For the general (Burkert) method:
          :math:`v_{\sigma}^2 = 2\sigma^2 (\mathrm{d}\ln\Sigma / \mathrm{d}\ln r)`
        - **Rotation velocity**: :math:`v_{\rm rot}^2 = v_{\rm circ}^2 +
          v_{\sigma}^2`

        The pressure support correction uses the logarithmic slope of the
        density profile, which is computed analytically for disk and ring
        components. For exponential profiles, a simplified formula is used:

        .. math::

           v_{\sigma}^2 = 3.36 \frac{r}{r_e} \sigma^2,

        where :math:`r_e` is the effective radius.

        This method stores multiple velocity arrays:
        - ``intrinsic``: Rotation velocity with pressure support
        - ``intrinsic_no_dispersion``: Circular velocity without pressure support
        - ``intrinsic_with_inclination``: Line-of-sight velocity with pressure support
        - ``intrinsic_no_dispersion_with_inclination``: Line-of-sight velocity without pressure support

        The dark matter fraction :math:`f_{\rm DM} = v_{\rm halo}^2 / v_{\rm circ}^2`
        is also computed and stored in ``self.fdm``.
        """
        absR = np.abs(R_array)
        
        # Initialize all velocity arrays at once
        velocity_components = ['V2d', 'Vd', 'V2b', 'Vb', 'V2r', 'Vr', 'V2h', 'Vh']
        for attr in velocity_components:
            setattr(self, attr, np.zeros_like(R_array))
        
        # Calculate component velocities using a mapping
        components_map = {
            'disk': ('V2d', 'Vd'),
            'ring': ('V2r', 'Vr'), 
            'bulge': ('V2b', 'Vb'),
            'halo': ('V2h', 'Vh')
        }
        
        for comp_name, (v2_attr, v_attr) in components_map.items():
            component = getattr(self, comp_name)
            if component is not None:
                setattr(self, v2_attr, component.vcirc2(R_array))
                setattr(self, v_attr, component.vcirc(R_array))
        
        self.get_dispersion_profile(R_array, functional_form=self.dispersion_function)
        self.V2sigma = np.zeros_like(absR)
        ### Regular exponential profile (using re)
        if self.pressure_support in ['exponential', 'Exponential']:
            if self.disk is not None:
                re = self.disk.r_eff
            elif self.ring is not None:
                re = self.ring.rpeak
            else:
                logger.warning('PRESSURE SUPPORT: exponential: No disk or rings component found, assuming Re=1 kpc...')
                re = 1.
            self.V2sigma += 3.36 * (R_array / re) * self.sigma_profile ** 2

        ### General Burkert(2010) formula using analytical derivatives of density profiles
        elif self.pressure_support in ['general', 'General', 'Generalized', 'generalized', 'burkert', 'burkert10', 'Burkert', 'Burkert10']:
            for comp in [self.disk, self.ring]:
                if comp is not None:
                    if comp._is_massive():
                        # self.V2sigma += 2 * self.sigma_profile ** 2 * (absR/comp.r_s) * comp.dlnrho_dlnr(absR)
                        self.V2sigma = (self.V2sigma +
                                        2 * self.sigma_profile ** 2 * comp.dlnrho_dlnr(absR))

        self.V2sigma = np.nan_to_num(self.V2sigma)
        self.Vsigma = safe_sqrt(self.V2sigma, sign_array=R_array)
        # self.Vsigma = np.sqrt(np.abs(self.V2sigma)) * np.sign(self.V2sigma)

        self.V2baryon = self.V2d + self.V2b + self.V2r
        self.Vbaryon = safe_sqrt(self.V2baryon, sign_array=R_array)
        # self.Vbaryon = np.sqrt(np.maximum(0, self.V2baryon)) * np.sign(R_array)

        self.V2circ = self.V2baryon + self.V2h
        # self.Vcirc = np.sqrt(np.maximum(0, self.V2circ)) * np.sign(R_array)
        self.Vcirc = safe_sqrt(self.V2circ, sign_array=R_array)

        self.V2rot = self.V2circ + self.V2sigma
        # self.Vrot = np.sqrt(np.maximum(0, self.V2rot)) * np.sign(R_array)
        self.Vrot = safe_sqrt(self.V2rot, sign_array=R_array)

        ### final velocities
        self.intrinsic_no_dispersion = self.Vcirc
        self.intrinsic_no_dispersion_with_inclination = self.intrinsic_no_dispersion * np.sin(np.deg2rad(self.inclination))
        self.intrinsic = self.Vrot
        self.intrinsic_with_inclination = self.intrinsic * np.sin(np.deg2rad(self.inclination))
        self.velocity_dispersion = self.sigma_profile

        self.fdm = np.nan_to_num(self.V2h / self.V2circ, nan=0.0)

    def apply_1D_beam_smearing(self, velocity_array, truncate=4.0, mode="nearest"):
        r"""
        \texttt{Deprecated. Use apply_2D_beam_smearing instead.}
        Apply 1D Gaussian beam smearing along the major axis.

        Convolves the velocity array with a 1D Gaussian kernel to simulate
        the effect of finite spatial resolution. The kernel has standard
        deviation :math:`\sigma_{\rm beam}` in pixel units.

        Parameters
        ----------
        velocity_array : array_like
            Input velocity array [km/s] to be smeared. Should be sampled on
            the extended radial grid (``sampling_rarray_1D``).
        truncate : float, optional
            Truncation distance for the Gaussian kernel in units of standard
            deviation. The kernel is truncated at :math:`\pm \mathrm{truncate}
            \times \sigma_{\rm beam}`. Default is 4.0.
        mode : str, optional
            Edge handling mode for the convolution. Options are ``'nearest'``,
            ``'constant'``, ``'reflect'``, ``'mirror'``, or ``'wrap'``.
            Default is ``'nearest'``.

        Returns
        -------
        ndarray
            Beam-smeared velocity array [km/s] on the original radial grid
            (``R_majoraxis``). The output is cropped to remove the oversampled
            edge regions.

        Notes
        -----
        The beam smearing is applied using a 1D Gaussian filter:

        .. math::

           v_{\rm smeared}(r) = \int v(r') G(r - r'; \sigma_{\rm beam}) \, dr',

        where :math:`G` is a Gaussian kernel with standard deviation
        :math:`\sigma_{\rm beam}`. The convolution is performed using
        ``scipy.ndimage.gaussian_filter1d``.

        The input array should be sampled on an extended grid that includes
        ``oversample_edge * sigma_beam`` beyond the original edge to avoid
        edge effects. The output is cropped to match the original radial grid.
        """
        smeared = gaussian_filter1d(velocity_array, self.sigma_beam_pixels, truncate=truncate, mode=mode)
        N = len(smeared)
        return smeared[self.oversample_pixels: N - self.oversample_pixels]

    def apply_2D_beam_smearing(self, one_dimensional_vel, one_dimensional_rarray, for_dispersion=False, ndim=None):
        r"""
        Apply 2D Gaussian beam smearing with light weighting.

        Convolves a 2D velocity field with a Gaussian beam kernel, weighted by
        the combined light profiles of all baryonic components. The method
        projects the 1D rotation curve onto a 2D grid, applies inclination
        and radial velocity corrections, then convolves with an elliptical
        Gaussian kernel.

        Parameters
        ----------
        one_dimensional_vel : array_like
            Input 1D velocity array [km/s] as a function of radius. This is
            interpolated onto a 2D grid before beam smearing.
        one_dimensional_rarray : array_like
            Radial array [kpc] corresponding to ``one_dimensional_vel``.
        for_dispersion : bool, optional
            If ``True``, square the velocity before smearing (used for
            calculating velocity dispersion from beam smearing effects).
            Default is ``False``.
        ndim : float, optional
            Output dimensionality. ``1`` extracts the major axis only,
            ``2`` returns the full 2D velocity field. If ``None``, uses
            ``self.ndim``. Default is ``None``.

        Returns
        -------
        V_smeared : ndarray
            Beam-smeared velocity array [km/s]. Shape depends on ``ndim``:
            - If ``ndim=1``: 1D array along the major axis
            - If ``ndim=2``: 2D array of shape ``(len(R_majoraxis), len(R_majoraxis))``
        I_smeared : ndarray
            Beam-smeared light profile (same shape as ``V_smeared``). Used
            for visualization and normalization purposes.

        Notes
        -----
        The 2D beam smearing process involves several steps:

        1. **Grid construction**: Create a 2D Cartesian grid from the 1D radial
           array, accounting for the extended sampling region.

        2. **Velocity interpolation**: Interpolate the 1D rotation curve onto
           the 2D grid using cubic spline interpolation.

        3. **Projection**: Project the circular velocity onto the line of sight:
           :math:`v_{\rm obs} = v_{\rm rot} \cos\theta \sin i + v_{\rm radial}
           \sin\theta \sin i`, where :math:`\theta` is the azimuthal angle
           and :math:`i` is the inclination.

        4. **Light weighting**: Compute the combined light profile from all
           baryonic components (disk, ring, bulge) on the 2D grid.

        5. **Beam convolution**: Convolve with an elliptical Gaussian kernel:
           :math:`G(x, y) = \exp[-(x^2/(2\sigma_x^2) + y^2/(2\sigma_y^2))]`,
           where :math:`\sigma_y = \sigma_x / \cos i` accounts for the
           projection along the galactic plane due to the inclination.

        6. **Weighted averaging**: At each point, compute the light-weighted
           average velocity:
           :math:`v_{\rm smeared} = \sum (v \times I \times G) / \sum (I \times G)`

        The elliptical beam accounts for the fact that an inclined disk appears
        elongated along the minor axis. The kernel size is determined by
        ``sigma_beam`` and ``oversample_edge``.

        If no light profile is available (all components have zero mass), a
        uniform light distribution is assumed with a warning.
        """

        if ndim is None:
            ndim = self.ndim

        # build 2D grid from the size of the input 1D array
        xx, yy = np.meshgrid(self.sampling_rarray_x, self.sampling_rarray_y)
        rgrid = np.sqrt(xx ** 2 + yy ** 2)

        interpolator = CubicSpline(x=one_dimensional_rarray, y=one_dimensional_vel)
        Vgrid = interpolator(rgrid)

        costhetha_grid = np.divide(xx, rgrid, out=np.zeros_like(rgrid), where=rgrid!=0)
        sinthetha_grid = np.divide(yy, rgrid, out=np.zeros_like(rgrid), where=rgrid!=0)

        # project on LOS velocity
        Vgrid *= costhetha_grid

        # Add radial flow
        Vgrid += self.Vradial * np.sin(np.deg2rad(self.inclination)) * sinthetha_grid

        if for_dispersion:
            Vgrid = Vgrid**2

        # apply light weighting filter
        Igrid = np.zeros_like(rgrid)
        if self.disk is not None:
            Igrid += self.disk.light_profile(xx, yy)
        if self.ring is not None:
            Igrid += self.ring.light_profile(xx, yy)
        if self.bulge is not None:
            Igrid += self.bulge.light_profile(xx, yy)

        # Check that Igrid is non-zero
        if np.all(Igrid == 0):
            logger.warning('No light weighting used. Assuming constant light...')
            Igrid = np.ones_like(rgrid)

        # normalize Igrid in case of really low values
        # Igrid /= np.max(Igrid)

        kernel_x = np.arange(-self.oversample_edge_pixels_x, self.oversample_edge_pixels_x + 1)
        kernel_y = np.arange(-self.oversample_edge_pixels_y, self.oversample_edge_pixels_y + 1)
        kernel_xx, kernel_yy = np.meshgrid(kernel_x, kernel_y)

        gaussian_kernel = 1.
        gaussian_kernel *= np.exp(-(kernel_xx ** 2 / (2 * self.sigma_beam_pixels_x ** 2)))
        gaussian_kernel *= np.exp(-(kernel_yy ** 2 / (2 * self.sigma_beam_pixels_y ** 2)))
        gaussian_kernel /= np.sum(gaussian_kernel)

        if ndim == 1:
            majoraxis_idx = int((rgrid.shape[0] - 1) / 2)
            majoraxis = rgrid[majoraxis_idx]
            N = len(majoraxis)

            V_major_axis = []
            smeared_light_major_axis = []
            for idx in range(self.oversample_edge_pixels_x, N - self.oversample_edge_pixels_x):
                # y_min = majoraxis_idx - self.oversample_edge_pixels_y
                # y_max = majoraxis_idx + self.oversample_edge_pixels_y + 1
                # x_min = idx - self.oversample_edge_pixels_x
                # x_max = idx + self.oversample_edge_pixels_x + 1

                # Kernel shape
                kernel_h, kernel_w = gaussian_kernel.shape

                # When slicing from the grid, always extract a patch of the same shape
                half_h = kernel_h // 2
                half_w = kernel_w // 2

                # For a point at (y, x) in the grid:
                y_min = majoraxis_idx - half_h
                y_max = majoraxis_idx + half_h + 1
                x_min = idx - half_w
                x_max = idx + half_w + 1

                Vgrid_idx = Vgrid[y_min:y_max, x_min:x_max]
                Igrid_idx = Igrid[y_min:y_max, x_min:x_max]

                weights = Igrid_idx * gaussian_kernel
                numerator = np.nansum(Vgrid_idx * weights)
                denominator = np.nansum(weights)
                V_major_axis_idx = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
                V_major_axis.append(V_major_axis_idx)

                numerator = np.nansum(Igrid_idx * gaussian_kernel)
                denominator = np.nansum(gaussian_kernel)
                smeared_light_major_axis_idx = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
                smeared_light_major_axis.append(smeared_light_major_axis_idx)

            # make sure velocity array has no Nans
            V_major_axis = np.array(V_major_axis)
            V_major_axis = np.nan_to_num(V_major_axis)

            # normalize light profile
            smeared_light_major_axis = np.array(smeared_light_major_axis)
            smeared_light_major_axis = np.nan_to_num(smeared_light_major_axis)
            # smeared_light_major_axis /= np.max(smeared_light_major_axis)

            return V_major_axis, smeared_light_major_axis

        elif ndim == 2.:
            V2d_array = np.zeros(shape=(len(self.R_majoraxis), len(self.R_majoraxis)))

            for idx in range(self.oversample_edge_pixels_y, len(self.sampling_rarray_y) - self.oversample_edge_pixels_y):
                V2d_array_idx = np.zeros(shape=len(self.R_majoraxis))

                for j in range(self.oversample_edge_pixels_x, len(self.sampling_rarray_x) - self.oversample_edge_pixels_x):
                    y_min = idx - self.oversample_edge_pixels_y
                    y_max = idx + self.oversample_edge_pixels_y + 1
                    x_min = j - self.oversample_edge_pixels_x
                    x_max = j + self.oversample_edge_pixels_x + 1

                    Vgrid_idx = Vgrid[y_min:y_max, x_min:x_max]
                    Igrid_idx = Igrid[y_min:y_max, x_min:x_max]
                    V2d_array_idx_j = np.divide(np.sum(Vgrid_idx * Igrid_idx * gaussian_kernel), np.sum(Igrid_idx * gaussian_kernel), out=np.zeros(shape=(1)), where=np.sum(Igrid_idx*gaussian_kernel)!=0)
                    V2d_array_idx[j - self.oversample_edge_pixels_x] = (V2d_array_idx_j)

                V2d_array[idx - self.oversample_edge_pixels_y] = V2d_array_idx

                Igrid = np.asarray(Igrid)
                Igrid = np.nan_to_num(Igrid)

            return V2d_array, Igrid


def calculate_fraction_at_re(mass_components=None, reval=None):
    r"""
    Calculate the dark matter fraction at a specified radius.

    Computes the dark matter mass fraction :math:`f_{\rm DM}` at a given
    radius by comparing the dark matter and total circular velocity
    contributions:

    .. math::

       f_{\rm DM} = \frac{v_{\rm halo}^2}{v_{\rm circ}^2}.

    Parameters
    ----------
    mass_components : dict, optional
        Dictionary containing mass component profiles. Keys should include:
        - ``'halo'``: Dark matter halo profile (e.g., :class:`NFWHalo`)
        - ``'disk'``: Disk component (e.g., :class:`FreemanDisk`)
        - ``'ring'``: Ring component (e.g., :class:`GaussianRingProfile`)
        - ``'bulge'``: Bulge component (e.g., :class:`SersicProfile`)
        Values can be ``None`` if a component is not present.
    reval : float, optional
        Radius at which to evaluate the dark matter fraction [kpc]. If not
        provided, defaults to the effective radius of the disk or ring
        component. If no disk or ring is present, returns 1.0.

    Returns
    -------
    float
        Dark matter fraction :math:`f_{\rm DM}` at the specified radius.
        Returns 0.0 if no halo is present, and 1.0 if no baryonic components
        are found to estimate ``reval``.

    Notes
    -----
    The function creates a minimal :class:`RotationCurveObject` instance
    with beam smearing disabled to compute the velocity contributions at
    the specified radius. The dark matter fraction is computed from the
    squared velocities:

    .. math::

       f_{\rm DM} = \frac{v_{\rm halo}^2}{v_{\rm circ}^2}.

    This is equivalent to the mass fraction only in the limit where the
    mass distribution is spherically symmetric, but provides a useful
    diagnostic for rotation curve decomposition.

    Examples
    --------
    Calculate the dark matter fraction at the effective radius:

    >>> from RotCurves.baryons import FreemanDisk
    >>> from RotCurves.dm_halos import NFWHalo
    >>> disk = FreemanDisk(mass=1e10, r_s=2.0)
    >>> halo = NFWHalo(mass=1e12, concentration=10)
    >>> components = {'disk': disk, 'halo': halo, 'ring': None, 'bulge': None}
    >>> f_dm = calculate_fraction_at_re(mass_components=components, reval=5.0)
    """
    if reval is None:
        print('reval not specified. Assuming reval = r_eff_disk...')
        if mass_components['disk'] is not None:
            reval = mass_components['disk'].r_eff
        elif mass_components['ring'] is not None:
            reval = mass_components['ring'].r_eff
        else:
            print('No disk or ring component found to estimate reval. Returning 1 for fdm...')
            return 1.

    if mass_components['halo'] is None:
        print('Cant calculate DM fractions for a model with no halo!')
        return 0.
    else:
        rc = RotationCurveObject(rarray=[reval], Halo=mass_components['halo'], Disk=mass_components['disk'],
                                 Ring=mass_components['ring'], Bulge=mass_components['bulge'], sigma_dispersion=0.,
                                 pressure_support='general', apply_2D=False, include_beam_smearing=False)
        fraction = rc.V2h / (rc.V2h + rc.V2baryon)
        return fraction[0]
