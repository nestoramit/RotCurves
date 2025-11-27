import numpy as np
import logging
from scipy.special import gamma, gammainc, gammaincinv, i0, k0, i1, k1
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

from RotCurves.base_classes import SurfaceDensityProfile
from RotCurves.const import (
    load_noor_lookuptables,
    load_gaussian_tables,
)

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')

# load lookup tables
NoordermeerLookupTables = load_noor_lookuptables()
GaussianRingLookupTables, GaussianRingBTminLookupTables = load_gaussian_tables()


class SersicProfile(SurfaceDensityProfile):
    r"""
    Sérsic surface-density profile for galactic components (Sérsic 1968).

    This class implements a Sérsic profile, a generalization of the exponential disk profile. 
    The surface density follows:

    .. math::

       \Sigma(x) = \Sigma_0 \cdot \exp\left[-x^{1/n}\right],

    where :math:`x = r/r_s` is the dimensionless radius and :math:`n` is the
    Sérsic index. The profile reduces to an exponential disk when :math:`n=1`.

    The enclosed mass is:

    .. math::

       M(<x) = M_s \cdot \gamma(2n, x^{1/n}) ,

    where :math:`\gamma(a, z)` is the lower incomplete gamma function.

    The circular velocity is computed considering an Vrot thickness with axis ratio :math:`q_0`
    (Noordermeer et al. 2008). By default, lookup tables are used for the circular velocity calculation 
    for the closest (n, q0) values.

    Parameters
    ----------
    mass : float
        Total mass of the component [:math:`M_\odot`].
    r_eff : float, optional
        Effective (half-mass) radius [kpc]. If provided together with ``r_s``,
        ``r_s`` takes precedence. At least one of ``r_eff`` or ``r_s`` must be given.
    r_s : float, optional
        Scale radius [kpc]. If not given, it is derived from ``r_eff`` using
        :math:`r_s = r_\mathrm{eff} / b^n`, where :math:`b` is the Sérsic
        parameter. At least one of ``r_eff`` or ``r_s`` must be given.
    n : float, optional
        Sérsic index, determining the concentration of the profile. Higher values
        of :math:`n` correspond to more centrally concentrated profiles.
        :math:`n=1` gives an exponential disk, :math:`n=4` gives a de Vaucouleurs
        profile. Default is 1.
    q0 : float, optional
        Intrinsic axis ratio of the mass distribution (:math:`0 \leq q_0 \leq 1`).
        Used for thickened disk calculations. Default is 0 (infinitely thin disk).
    mass_to_light : float, optional
        Mass-to-light ratio used when generating a light profile from the mass
        profile (arbitrary but consistent units). Default is 1.
    lookup : bool, optional
        If ``True``, use lookup tables for circular velocity calculations
        (Noordermeer et al. 2008). Default is ``True``.

    Attributes
    ----------
    n : float
        Sérsic index.
    q0 : float
        Intrinsic axis ratio.
    lookup : bool
        Flag indicating whether lookup tables are used for circular velocity.
    vcirc_lookup_table : ndarray
        Lookup table for dimensionless circular velocity squared.

    Notes
    -----
    The Sérsic parameter :math:`b` is determined by solving
    :math:`\gamma(2n, b) = \Gamma(2n) / 2`, which ensures that half the mass
    is contained within the effective radius.

    The circular velocity calculation uses interpolation of pre-computed lookup
    tables when ``lookup=True``. The tables are parameterized by Sérsic index
    :math:`n` and axis ratio :math:`q_0`.

    References
    ----------
    Sérsic, J. L. 1968, Atlas de Galaxias Australes (Córdoba: Observatorio Astronómico)
    Noordermeer, E., et al. 2008, MNRAS, 385, 1359

    Examples
    --------
    Create a Sérsic profile with n=2 (intermediate between exponential and de Vaucouleurs):

    >>> prof = SersicProfile(mass=1e10, r_eff=5.0, n=2.0)
    >>> r = 10.0  # kpc
    >>> sigma = prof.surface_density(r)  # Σ(r)
    >>> v_c = prof.vcirc(r)              # circular velocity at r
    """
    def __init__(self, mass, r_eff=None, r_s=None, n=1, q0=0., mass_to_light=1., lookup=True):
        # Set the Sersic index n
        self.n = n

        # Call the parent class constructor
        super().__init__(mass=mass, r_eff=r_eff, r_s=r_s,
                         surface_density_function=self.surface_density_function, q0=q0,
                         mass_to_light=mass_to_light)


        # load lookuptables
        self.lookup = lookup
        # self.vcirc_lookup_table = None
        if self.lookup:
            self._lookup_table_values()
        #     self.vcirc_lookup_table = self._vcirc_lookup_table()

    def surface_density_function(self, x):
        r"""
        Dimensionless surface density :math:`f_{\Sigma}(x)` for the Sérsic profile:

        .. math::

           f_{\Sigma}(x) = \exp\left[-x^{1/n}\right],

        where :math:`x = r/r_s` is the dimensionless radius and :math:`n` is
        the Sérsic index.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless surface density :math:`f_{\Sigma}(x)`.
        """
        return np.exp(-x ** (1 / self.n))

    def sersic_b(self):
        r"""
        Calculate the Sérsic :math:`b` parameter.

        The Sérsic parameter :math:`b` is determined by solving the condition
        that half the mass is contained within the effective radius:

        .. math::

           \gamma(2n, b) = \frac{1}{2} \Gamma(2n),

        where :math:`\gamma(a, z)` is the lower incomplete gamma function.

        Returns
        -------
        float
            Sérsic parameter :math:`b`, which relates the effective radius to
            the scale radius via :math:`r_\mathrm{eff} = b^n r_s`.
        """
        return gammaincinv(2 * self.n, 0.5)

    def _calculate_effective_radius_from_scale(self):
        r"""
        Calculate the effective radius from the scale radius for a Sérsic profile.

        The effective radius is related to the scale radius by:

        .. math::

           r_\mathrm{eff} = b^n r_s,

        where :math:`b` is the Sérsic parameter.

        Returns
        -------
        float
            Effective radius :math:`r_\mathrm{eff}` [kpc].
        """
        return self.sersic_b()**self.n * self.r_s

    def _calculate_scale_radius_from_effective(self):
        r"""
        Calculate the scale radius from the effective radius for a Sérsic profile.

        The scale radius is related to the effective radius by:

        .. math::

           r_s = \frac{r_\mathrm{eff}}{b^n},

        where :math:`b` is the Sérsic parameter.

        Returns
        -------
        float
            Scale radius :math:`r_s` [kpc].
        """
        return self.sersic_b()**-self.n * self.r_eff

    def _scale_density(self):
        r"""
        Compute the surface-density normalization :math:`\Sigma_0` for a Sérsic profile.

        The scale density is:

        .. math::

           \Sigma_0 = \frac{M}{2\pi r_s^2 \Gamma(2n)},

        where :math:`\Gamma(2n)` is the complete gamma function, which accounts
        for the normalization of the Sérsic profile.

        Returns
        -------
        float
            Surface density normalization :math:`\Sigma_0` [:math:`M_\odot \, kpc^{-2}`].
        """
        correction_factor = 1 / gamma(2 * self.n)
        return self.mass / (2 * np.pi * self.r_s ** 2) * correction_factor

    def _scale_mass(self):
        r"""
        Compute the mass scaling factor :math:`M_s` for a Sérsic profile.

        The scale mass is:

        .. math::

           M_s = \frac{M}{\Gamma(2n)},

        where :math:`\Gamma(2n)` is the complete gamma function. This ensures
        that :math:`M(<x) = M_s f_M(<x)`.

        Returns
        -------
        float
            Mass scale factor :math:`M_s` [:math:`M_\odot`].
        """
        return self.mass / gamma(2 * self.n)

    def menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_M(<x)` for a Sérsic profile:

        .. math::

           f_M(<x) = \gamma(2n, x^{1/n}),

        where :math:`\gamma(a, z)` is the lower incomplete gamma function.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Returns
        -------
        ndarray
            Dimensionless enclosed mass :math:`f_M(<x)`, shape compatible with
            ``np.atleast_1d(x)``.
        """
        return gammainc(2 * self.n, x ** (1 / self.n)) * gamma(2 * self.n)

    def _lookup_table_values(self):
        self.n_table = self._n_table()
        self.q0_table = self._q0_table()

    def _n_table(self):
        n_list = np.asarray(
            sorted(set([x[0] for x in NoordermeerLookupTables.keys()]))
        )
        n_table = n_list[np.argmin(np.abs(n_list - self.n))]
        if np.abs(self.n - n_table) > 1e-2:
            logger.warning(
                f"SersicProfile index n: using lookup tables, taking %2.2f instead of %2.2f" % (n_table, self.n))
        return n_table

    def _q0_table(self):
        q0_list = np.asarray(
            sorted(set([x[1] for x in NoordermeerLookupTables.keys()]))
        )
        q0_table = q0_list[np.argmin(np.abs(q0_list - self.q0))]
        if np.abs(self.q0 - q0_table) > 1e-2:
            logger.warning(
                f"SersicProfile q0: using lookup tables, taking %2.2f instead of %2.2f" % (q0_table, self.q0))
        return q0_table

    def _vcirc_lookup_table(self):
        r"""
        Load the lookup table for dimensionless circular velocity squared.

        Retrieves the appropriate lookup table from Noordermeer et al. (2008)
        based on the Sérsic index :math:`n` and axis ratio :math:`q_0`. If the
        exact :math:`q_0` value is not available, the closest value is used
        with a warning.

        Returns
        -------
        ndarray
            Lookup table with shape ``(N, 2)``, where the first column contains
            dimensionless radii and the second column contains dimensionless
            squared circular velocities.
        """

        return NoordermeerLookupTables[self.n_table, self.q0_table]

    def vcirc2_dimless(self, x):
        r"""
        Dimensionless squared circular velocity :math:`f_v^2(x)` for a Sérsic profile.

        The circular velocity is computed following Noordermeer et al. (2008) for
        a thickened disk. When ``lookup=True``, the calculation uses pre-computed
        lookup tables that account for the finite thickness of the disk via the
        axis ratio :math:`q_0`.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless squared circular velocity :math:`f_v^2(x)`.

        Notes
        -----
        The lookup tables are stored in dimensionless units with respect to the
        effective radius, so a conversion factor is applied. The normalization
        constant :math:`C` accounts for the Sérsic profile parameters.

        References
        ----------
        Noordermeer, E., et al. 2008, MNRAS, 385, 1359
        """
        if self.lookup:
            vcirc_lookup_table = self._vcirc_lookup_table()
            interpolator = CubicSpline(
                x=vcirc_lookup_table[:, 0],
                y=vcirc_lookup_table[:, 1]
            )

            # TODO: the talbes are in x=r/reff, change to x=r/rs
            v2 = interpolator(x * self.r_s / self.r_eff)

            # TODO: added the constant C directly to single_noordermeer_calculation (remove after checks)
            # C = 2 * self.sersic_b()**(self.n+1) / (np.pi * self.n**2)
            C = 1

            return C * v2

        else:
            from RotCurves.calculate_tables import single_noordermeer_calculation
            x_eff = np.atleast_1d(x) * self.r_s / self.r_eff
            v2 = np.zeros_like(x_eff)
            for i in range(len(x_eff)):
                v2[i] = single_noordermeer_calculation(i, self.q0, self.n, x_eff)
            return v2[0] if np.isscalar(x) else v2

    def dlnrho_dlnr(self, r):
        r"""
        Logarithmic slope of the surface density profile at radius ``r``.

        Calculate the logarithmic slope :math:`\mathrm{d}\ln\Sigma / \mathrm{d}\ln r`
        of the Sérsic surface density profile. For a Sérsic profile:

        .. math::

           \frac{\mathrm{d}\ln\Sigma}{\mathrm{d}\ln r} = -\frac{1}{n} x^{1/n},

        where :math:`x = r/r_s` is the dimensionless radius.

        Parameters
        ----------
        r : array_like or float
            Radius [kpc].

        Returns
        -------
        ndarray or float
            Logarithmic slope :math:`\mathrm{d}\ln\Sigma / \mathrm{d}\ln r`
            evaluated at each input radius.

        Notes
        -----
        This quantity is commonly used in pressure-support and asymmetric-drift
        calculations (e.g., Burkert et al. 2010).

        References
        ----------
        Burkert, A., et al. 2016, ApJ, 826, 214
        """
        x = self._normalized_radius(r)
        return - 1/self.n * x**(1/self.n)


class FreemanDisk(SurfaceDensityProfile):
    r"""
    Exponential disk surface-density profile (Freeman 1970).

    This class implements an exponential disk profile, which is a special case
    of the Sérsic profile with :math:`n=1`. The surface density follows:

    .. math::

       \Sigma(x) = \Sigma_0 \cdot \exp(-x),

    where :math:`x = r/r_s` is the dimensionless radius.

    The enclosed mass is:

    .. math::

       M(<x) = M_s \cdot (1 - \exp(-x)(1 + x)).

    The circular velocity is computed analytically using modified Bessel functions
    for a razor-thin disk (Binney & Tremaine 2008).

    Parameters
    ----------
    mass : float
        Total mass of the disk [:math:`M_\odot`].
    r_eff : float, optional
        Effective (half-mass) radius [kpc]. If provided together with ``r_s``,
        ``r_s`` takes precedence. At least one of ``r_eff`` or ``r_s`` must be given.
    r_s : float, optional
        Scale radius (exponential scale length) [kpc]. If not given, it is derived
        from ``r_eff`` using :math:`r_s = r_\mathrm{eff} / 1.678`. At least one
        of ``r_eff`` or ``r_s`` must be given.
    mass_to_light : float, optional
        Mass-to-light ratio used when generating a light profile from the mass
        profile (arbitrary but consistent units). Default is 1.

    Notes
    -----
    The exponential disk is equivalent to a Sérsic profile with :math:`n=1`.
    The effective radius is related to the scale radius by
    :math:`r_\mathrm{eff} \approx 1.678 r_s`.

    The circular velocity calculation uses the analytical solution for a
    razor-thin exponential disk, which involves modified Bessel functions
    :math:`I_0`, :math:`I_1`, :math:`K_0`, and :math:`K_1`.

    References
    ----------
    Freeman, K. C. 1970, ApJ, 160, 811
    Binney, J., & Tremaine, S. 2008, Galactic Dynamics (2nd ed.; Princeton: Princeton Univ. Press)

    Examples
    --------
    Create an exponential disk:

    >>> disk = FreemanDisk(mass=1e10, r_s=2.0)
    >>> r = 5.0  # kpc
    >>> sigma = disk.surface_density(r)  # Σ(r)
    >>> v_c = disk.vcirc(r)               # circular velocity at r
    """
    def __init__(self, mass, r_eff=None, r_s=None, mass_to_light=1.):
        # Call the parent class constructor
        super().__init__(mass=mass, r_eff=r_eff, r_s=r_s,
                         q0=0., surface_density_function=self.surface_density_function,
                         mass_to_light=mass_to_light)

    def surface_density_function(self, x):
        r"""
        Dimensionless surface density :math:`f_{\Sigma}(x)` for an exponential disk:

        .. math::

           f_{\Sigma}(x) = \exp(-x),

        where :math:`x = r/r_s` is the dimensionless radius.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless surface density :math:`f_{\Sigma}(x)`.
        """
        return np.exp(-x)

    def _calculate_effective_radius_from_scale(self):
        r"""
        Calculate the effective radius from the scale radius for an exponential disk.

        The effective radius is related to the scale radius by:

        .. math::

           r_\mathrm{eff} = 1.678 r_s.

        Returns
        -------
        float
            Effective radius :math:`r_\mathrm{eff}` [kpc].
        """
        return self.r_s * 1.678347

    def _calculate_scale_radius_from_effective(self):
        r"""
        Calculate the scale radius from the effective radius for an exponential disk.

        The scale radius is related to the effective radius by:

        .. math::

           r_s = \frac{r_\mathrm{eff}}{1.678}.

        Returns
        -------
        float
            Scale radius :math:`r_s` [kpc].
        """
        return self.r_eff / 1.678347

    def _scale_density(self):
        r"""
        Compute the surface-density normalization :math:`\Sigma_0` for an exponential disk.

        The scale density is:

        .. math::

           \Sigma_0 = \frac{M}{2\pi r_s^2},

        where :math:`M` is the total mass and :math:`r_s` is the scale radius.

        Returns
        -------
        float
            Surface density normalization :math:`\Sigma_0` [:math:`M_\odot \, kpc^{-2}`].
        """
        return self.mass / (2 * np.pi * self.r_s ** 2)

    def _scale_mass(self):
        r"""
        Compute the mass scaling factor :math:`M_s` for an exponential disk.

        For an exponential disk, the scale mass equals the total mass:

        .. math::

           M_s = M.

        Returns
        -------
        float
            Mass scale factor :math:`M_s` [:math:`M_\odot`].
        """
        return self.mass

    def menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_M(<x)` for an exponential disk.

        The enclosed mass is computed using the analytical expression for an
        exponential disk:

        .. math::

           f_M(<x) = 1 - \exp(-x)(1 + x).

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Returns
        -------
        ndarray
            Dimensionless enclosed mass :math:`f_M(<x)`, shape compatible with
            ``np.atleast_1d(x)``.
        """
        return 1 - np.exp(-x) * (1 + x)

    def vcirc2_dimless(self, x):
        r"""
        Dimensionless squared circular velocity :math:`f_v^2(x)` for an exponential disk.

        The circular velocity is computed analytically for a razor-thin exponential
        disk using modified Bessel functions (Binney & Tremaine 2008):

        .. math::

           f_v^2(x) = 2y^2 \left[I_0(y) K_0(y) - I_1(y) K_1(y)\right],

        where :math:`y = x/2`, :math:`x = r/r_s` the dimensionless radius,
         and :math:`I_n`, :math:`K_n` are modified Bessel
        functions of the first and second kind, respectively.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless squared circular velocity :math:`f_v^2(x)`. Values
            that evaluate to NaN (typically at :math:`x=0`) are set to zero.

        Notes
        -----
        This is the analytical solution for a razor-thin exponential disk. The
        result is clipped to zero where the Bessel functions produce NaN values.

        References
        ----------
        Binney, J., & Tremaine, S. 2008, Galactic Dynamics (2nd ed.; Princeton: Princeton Univ. Press)
        """
        y = x/2
        res = 2 * y**2 * (i0(y) * k0(y) - i1(y) * k1(y))
        return np.nan_to_num(res, nan=0)

    def dlnrho_dlnr(self, r):
        r"""
        Logarithmic slope of the surface density profile at radius ``r``.

        Calculate the logarithmic slope :math:`\mathrm{d}\ln\Sigma / \mathrm{d}\ln r`
        of the exponential surface density profile. For an exponential disk:

        .. math::

           \frac{\mathrm{d}\ln\Sigma}{\mathrm{d}\ln r} = -x,

        where :math:`x = r/r_s` is the dimensionless radius.

        Parameters
        ----------
        r : array_like or float
            Radius [kpc].

        Returns
        -------
        ndarray or float
            Logarithmic slope :math:`\mathrm{d}\ln\Sigma / \mathrm{d}\ln r`
            evaluated at each input radius.

        Notes
        -----
        This quantity is commonly used in pressure-support and asymmetric-drift
        calculations (e.g., Burkert et al. 2010).

        References
        ----------
        Burkert, A., et al. 2010, ApJ, 725, 2324
        """
        x = self._normalized_radius(r)
        return - x


class GaussianRingProfile(SurfaceDensityProfile):
    r"""
    Gaussian ring surface-density profile.

    This class implements a Gaussian ring profile, where the surface density is
    a shifted Gaussian centered at the scale radius. The surface density follows:

    .. math::

       \Sigma}(x) = \Sigma_0 \cdot \exp\left[-A(x-1)^2\right],

    where :math:`x = r/r_s` is the dimensionless radius, :math:`r_s` is the peak
    of the Gaussian, and :math:`A = r_s^2/(2\sigma_\mathrm{ring}^2)` is a
    shape parameter related to the width of the ring.

    The ring is characterized by four related parameters:
    - :math:`r_s`: Peak (scale) radius of the ring
    - :math:`h`: Shape parameter, :math:`h = r_s / \mathrm{FWHM}_\mathrm{ring}`
    - :math:`\mathrm{FWHM}_\mathrm{ring}`: Full Width at Half Maximum
    - :math:`\sigma_\mathrm{ring}`: Standard deviation of the Gaussian

    Exactly two of these parameters must be provided; the other two are calculated
    automatically.

    Parameters
    ----------
    mass : float
        Total mass of the ring [:math:`M_\odot`].
    r_s : float, optional
        Scale (peak) radius where the ring density is maximum [kpc]. At least
        one of ``r_s`` or ``h`` must be provided.
    h : float, optional
        Shape parameter defined as :math:`h = r_s / \mathrm{FWHM}_\mathrm{ring}`.
        Higher values correspond to narrower rings. At least one of ``r_s`` or
        ``h`` must be provided.
    FWHM_ring : float, optional
        Full Width at Half Maximum of the Gaussian ring [kpc]. If not provided,
        it is calculated from the other parameters.
    sigma_ring : float, optional
        Standard deviation of the Gaussian ring [kpc]. If not provided, it is
        calculated from the other parameters via :math:`\sigma_\mathrm{ring} =
        \mathrm{FWHM}_\mathrm{ring} / (2\sqrt{2\ln 2})`.
    mass_to_light : float, optional
        Mass-to-light ratio used when generating a light profile from the mass
        profile (arbitrary but consistent units). Default is 1.
    lookup : bool, optional
        If ``True``, use lookup tables for circular velocity calculations.
        Default is ``True``.
    verbose : bool, optional
        If ``True``, enable verbose warnings (e.g., when using non-exact lookup
        table values). Default is ``False``.

    Attributes
    ----------
    h : float
        Shape parameter :math:`h = r_s / \mathrm{FWHM}_\mathrm{ring}`.
    r_s : float
        Scale (peak) radius [kpc].
    sigma_ring : float
        Standard deviation of the Gaussian ring [kpc].
    FWHM_ring : float
        Full Width at Half Maximum [kpc].
    A : float
        Shape parameter :math:`A = r_s^2/(2\sigma_\mathrm{ring}^2)`.
    lookup : bool
        Flag indicating whether lookup tables are used for circular velocity.
    vcirc_lookup_table : ndarray
        Lookup table for dimensionless circular velocity squared.
    btmin_lookup_table : ndarray
        Lookup table for minimum bulge-to-total mass ratio for stability.

    Notes
    -----
    The Gaussian ring profile is useful for modeling ring-like structures in
    galaxies, such as nuclear rings or resonance rings. The profile is normalized
    such that the total mass is :math:`M`.

    The circular velocity function shape is uniquely determined by the shape parameter ``h``.
    The calculation uses pre-computed lookup tables when ``lookup=True`` from 
    the closest ``h`` available. 

    Examples
    --------
    Create a Gaussian ring with peak radius 5 kpc and FWHM 2 kpc:

    >>> ring = GaussianRingProfile(mass=1e9, r_s=5.0, FWHM_ring=2.0)
    >>> r = 5.0  # kpc
    >>> sigma = ring.surface_density(r)  # Σ(r)
    >>> v_c = ring.vcirc(r)              # circular velocity at r
    """
    def __init__(self, mass, r_s=None, h=None, FWHM_ring=None, sigma_ring=None, mass_to_light=1.,
                 lookup=True, verbose=False):

        # Set the Gaussian ring parameters
        # h is defined as: h = r_s / FWHM_ring
        # check to see that given any two of the four parameters {r_s, h, FWHM_ring, sigma_ring},
        # the other two can be calculated
        if r_s is None and h is None:
            raise ValueError("Either r_s or h must be provided.")

        class _ParameterSolver:
            def __init__(self):
                # Map of combinations to calculation functions
                self.method_map = {
                    ('FWHM_ring', 'h'): self.calculate_from_fwhm_h,
                    ('FWHM_ring', 'r_s'): self.calculate_from_fwhm_rs,
                    ('h', 'r_s'): self.calculate_from_h_rs,
                    ('h', 'sigma_ring'): self.calculate_from_h_sigma,
                    ('r_s', 'sigma_ring'): self.calculate_from_rs_sigma,
                }

            def calculate(self, **kwargs):
                """
                Calculate the two missing parameters given two known ones.
                Example: calculate(A=3, B=4, C=None, D=None)
                """
                # Extract given and missing parameter names
                given = {k: v for k, v in kwargs.items() if v is not None}
                missing = [k for k, v in kwargs.items() if v is None]

                # Check if exactly two values are given
                if len(given) != 2 or len(missing) != 2:
                    raise ValueError("Exactly two parameters must be given and two must be None.")

                # Sort the given keys to match the method map
                given_keys = tuple(sorted(given.keys()))

                # Find the appropriate calculation function
                try:
                    func = self.method_map[given_keys]
                except KeyError:
                    raise ValueError(f"No calculation method for given pair: {given_keys}")

                # Calculate the missing values
                results = func(*given.values())

                # Build the complete result dictionary
                result_dict = {**given, **dict(zip(missing, results))}
                return result_dict

            # Calculation functions for each pair
            def calculate_from_fwhm_h(self, fwhm, h):
                scale = fwhm * h
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                return scale, sigma

            def calculate_from_fwhm_rs(self, fwhm, r_s):
                h = r_s / fwhm
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                return h, sigma

            def calculate_from_h_rs(self, h, r_s):
                fwhm = r_s / h
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                return fwhm, sigma

            def calculate_from_h_sigma(self, h, sigma):
                fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
                scale = fwhm * h
                return fwhm, scale

            def calculate_from_rs_sigma(self, r_s, sigma):
                fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
                h = r_s / fwhm
                return h, fwhm
        parameters = _ParameterSolver().calculate(h=h, FWHM_ring=FWHM_ring, r_s=r_s, sigma_ring=sigma_ring)
        self.h = parameters['h']
        self.r_s = parameters['r_s']
        self.sigma_ring = parameters['sigma_ring']
        self.FWHM_ring = parameters['FWHM_ring']
        self.A = self.r_s ** 2 / (2 * self.sigma_ring ** 2)
        # Call the parent class constructor
        super().__init__(mass=mass, r_s=self.r_s, surface_density_function=self.surface_density_function,
                         q0=0., mass_to_light=mass_to_light, verbose=verbose)

        # load lookuptables
        # TODO: update path
        self.lookup = lookup
        if self.lookup:
            self.h_table = self._h_table()

        # self.vcirc_lookup_table = self._vcirc_lookup_table()
        self.btmin_lookup_table = self._btmin_lookup_table()

    def surface_density_function(self, x):
        r"""
        Dimensionless surface density :math:`f_{\Sigma}(x)` for a Gaussian ring:

        .. math::

           f_{\Sigma}(x) = \exp\left[-A(x-1)^2\right],

        where :math:`x = r/r_s` is the dimensionless radius, :math:`r_s` is the 
        peak of the Gaussian and :math:`A = r_s^2/(2\sigma_\mathrm{ring}^2)` 
        is a shape parameter.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless surface density :math:`f_{\Sigma}(x)`.
        """
        return np.exp(-self.A * (x - 1) ** 2)

    def _scale_density(self):
        r"""
        Compute the surface-density normalization :math:`\Sigma_0` for a Gaussian ring.

        The scale density accounts for the normalization of the shifted Gaussian
        profile:

        .. math::

           \Sigma_0 = \frac{M}{2\pi r_s^2 C},

        where :math:`C` is a correction factor that accounts for the ring geometry:

        .. math::

           C = \frac{1}{2A}\left[\exp(-A) + \sqrt{\pi A}\left(1 + \Gamma(0.5, A)\right)\right],

        and :math:`\Gamma(0.5, A)` is the upper incomplete gamma function.

        Returns
        -------
        float
            Surface density normalization :math:`\Sigma_0` [:math:`M_\odot \, kpc^{-2}`].
        """
        A = self.A
        correction_factor = 1 / (2 * A) * (np.exp(-A) + np.sqrt(np.pi * A) * (1 + gammainc(0.5, A)))
        return self.mass / (2 * np.pi * self.r_s ** 2) / correction_factor

    def _scale_mass(self):
        r"""
        Compute the mass scaling factor :math:`M_s` for a Gaussian ring.

        The scale mass accounts for the normalization of the shifted Gaussian
        profile:

        .. math::

           M_s = \frac{M}{C},

        where :math:`C` is a correction factor that accounts for the ring geometry:

        .. math::

           C = \frac{1}{2A}\left[\exp(-A) + \sqrt{\pi A}\left(1 + \Gamma(0.5, A)\right)\right],

        Returns
        -------
        float
            Mass scale factor :math:`M_s` [:math:`M_\odot`].
        """
        A = self.A
        corr = 1 / (2 * A) * (np.exp(-A) + np.sqrt(np.pi * A) * (1 + gammainc(0.5, A)))
        return self.mass / corr

    def _h_table(self):
        h_list = np.asarray(
            sorted(set(list(GaussianRingLookupTables.keys())))
        )
        h_table = h_list[np.argmin(np.abs(h_list - self.h))]
        if np.abs(self.h - h_table) > 1e-2:
            logger.warning(
                f"GaussianRing shape parameter h: using lookup tables, "
                f"taking {h_table:2.2f} instead of {self.h:2.2f}")
        return h_table

    def _vcirc_lookup_table(self):
        r"""
        Load the lookup table for dimensionless circular velocity squared.

        Retrieves the appropriate lookup table based on the shape parameter
        :math:`h`. If the exact :math:`h` value is not available, the closest
        value is used with an optional warning.

        Returns
        -------
        ndarray
            Lookup table with shape ``(N, 2)``, where the first column contains
            dimensionless radii and the second column contains dimensionless
            squared circular velocities.
        """
        # h_list = GaussianRingLookupTables['h_list']
        # if self.h in h_list:
        #     closest_h = self.h
        # else:
        #     closest_h = h_list[np.argmin(np.abs(h_list - self.h))]
        #     if self.verbose:
        #         logger.warning(f'Gaussian Ring h: non-exact value, using {closest_h:2.3f} instead of {self.h:2.3f}')

        return GaussianRingLookupTables[self.h_table]

    def _btmin_lookup_table(self):
        r"""
        Load the lookup table for minimum bulge-to-total mass ratio.

        Retrieves the appropriate lookup table for the minimum bulge-to-total
        mass ratio (:math:`B/T_\mathrm{min}`) required to stabilize the ring
        against gravitational instabilities. The table is parameterized by the
        shape parameter :math:`h`.

        Returns
        -------
        ndarray
            Lookup table with shape ``(N, 2)``, where the first column contains
            scale radii and the second column contains minimum :math:`B/T` values.
        """
        h_list = GaussianRingBTminLookupTables['h_list']
        if self.h in h_list:
            closest_h = self.h
        else:
            closest_h = h_list[np.argmin(np.abs(h_list - self.h))]
            if self.verbose:
                logger.warning(f'Gaussian Ring h: non-exact value, using {closest_h:2.3f} instead of {self.h:2.3f}')

        return GaussianRingBTminLookupTables[closest_h]

    def min_stabilizing_mass(self):
        r"""
        Calculate the minimum bulge-to-total mass ratio for ring stability.

        Determines the minimum bulge-to-total mass ratio (:math:`B/T_\mathrm{min}`)
        required to stabilize the Gaussian ring against gravitational instabilities,
        such that system is in centrifugal equilibrium.
        This is done by finding the smallest :math:`B/T` such that the total
        circular velocity squared (ring + bulge) is positive everywhere.

        When ``lookup=True``, the value is interpolated from pre-computed lookup
        tables. Otherwise, it is calculated numerically by testing different
        :math:`B/T` values with a de Vaucouleurs bulge (:math:`n=4`).

        Returns
        -------
        float
            Minimum bulge-to-total mass ratio :math:`B/T_\mathrm{min}` required
            for stability. If no stable solution is found, returns 0.99.

        Notes
        -----
        The calculation assumes a de Vaucouleurs bulge profile (:math:`n=4`,
        :math:`q_0=1`) with effective radius 1 kpc. The stability criterion is
        that the total circular velocity squared must be positive at all radii
        from :math:`0.01 r_s` to :math:`2 r_s`.
        """
        if self.lookup:
            interpolator = CubicSpline(x=self.btmin_lookup_table[: ,0],
                                       y=self.btmin_lookup_table[:, 1])
            BT_min = interpolator(self.r_s)

        else:
            logger.warning('Gaussian Ring BT min: No lookuptable found. Calculating ...')

            R_array = np.logspace(-2, np.log10(2), num=51) * self.r_s
            N = int(1e3)
            i = 0
            for BT in np.logspace(-3, 0, num=N):
                bulge_mass = self.mass * BT / (1 - BT)
                bulge = SersicProfile(mass=bulge_mass, r_eff=1.0, n=4.0, q0=1.0)

                vcirc2_new = self.vcirc2(R_array) + bulge.vcirc2(R_array)
                if all(vcirc2_new > 0):
                    BT_min = np.ceil(BT * 1e4) * 1e-4
                    logger.warning(f'Gaussian Ring BT min: Found BTmin = {BT_min:%.4f} for invh = {self.h:%.2f} ...')
                    break
                if i == N-1:
                    logger.warning(r"Couldn't find central stabilizing mass for the given Gaussian ring distribution.")
                    BT_min = 0.99
                i += 1

        return BT_min

    def menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_M(<x)` for a Gaussian ring.

        The enclosed mass is computed using the analytical expression for a
        shifted Gaussian profile:

        .. math::

           f_M(<x) = \frac{1}{2A}\left[\exp(-A) - \exp(-A(x-1)^2)\right]
                    + \frac{1}{2\sqrt{A}} 
                    \left[\gamma(0.5, A) + \gamma(0.5, A(x-1)^2)\cdot\mathrm{sign}(x-1)\right]\Gamma(0.5),

        where :math:`A = r_s^2/(2\sigma_\mathrm{ring}^2)` 
        and :math:`\gamma(a, z)` is the lower incomplete gamma function.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Returns
        -------
        ndarray
            Dimensionless enclosed mass :math:`f_M(<x)`, shape compatible with
            ``np.atleast_1d(x)``.
        """
        A = self.A
        Ax = A * (x - 1) ** 2

        p1 = 1 / (2 * A) * (np.exp(-A) - np.exp(-Ax))
        p2 = 1 / (2 * np.sqrt(A)) * (gammainc(0.5, A) + gammainc(0.5, Ax) * np.sign(x - 1)) * gamma(0.5)
        return p1 + p2

    def vcirc2_dimless(self, x):
        r"""
        Dimensionless squared circular velocity :math:`f_v^2(x)` for a Gaussian ring.

        The circular velocity is computed using pre-computed lookup tables when
        ``lookup=True``, or by numerical integration when ``lookup=False``.

        When using lookup tables, the dimensionless squared circular velocity is:

        .. math::

           f_v^2(x) = C \cdot f_{v,\mathrm{table}}^2(x),

        where :math:`C = 4A/\pi` is a normalization constant and
        :math:`f_{v,\mathrm{table}}^2` is interpolated from the lookup table.

        When ``lookup=False``, the calculation uses numerical integration of the
        gravitational potential for an axisymmetric mass distribution.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless squared circular velocity :math:`f_v^2(x)`.

        Notes
        -----
        The squared circular velocity :math:`f_v^2` can be negative in the inner region, 
        in a region inside :math:`x<1`.
        The lookup tables are parameterized by the shape parameter :math:`h`.
        The normalization constant :math:`C` accounts for the ring geometry.
        """
        if self.lookup:
            # Use the lookup table for the Gaussian ring
            # TODO: something is wrong with the lookup table, it is not working. calculate it again.
            vcirc_lookup_table = self._vcirc_lookup_table()
            interpolator = CubicSpline(
                x=vcirc_lookup_table[:, 0],
                y=vcirc_lookup_table[:, 1]
            )
            v2 = interpolator(x)

        else:
            from RotCurves.calculate_tables import single_GaussianRing_integral_v2
            isscalar = np.isscalar(x)
            x = np.atleast_1d(x)
            v2 = np.zeros_like(x)
            for i in range(len(x)):
                v2[i] = single_GaussianRing_integral_v2(i, self.h, x)
            return v2[0] if isscalar else v2
            # Iprime = lambda t: quad(lambda s: t * (s**2 - t**2)**-0.5 * (1-s) * self.surface_density_dimless(s), t, np.inf)[0]
            # func = lambda x: quad(lambda t: -Iprime(t) * t * (x**2 - t**2)**-0.5, 0, x)[0]
            # v2 = [func(xi) for xi in x]
            # v2 = np.asarray(v2)

        # TODO: Added the constant to single_GaussianRing_integral (remove after checks)
        C = 1.
        # C = 4 * self.A / np.pi
        return C * v2

    def dlnrho_dlnr(self, r):
        r"""
        Logarithmic slope of the surface density profile at radius ``r``.

        Calculate the logarithmic slope :math:`\mathrm{d}\ln\Sigma / \mathrm{d}\ln r`
        of the Gaussian ring surface density profile. For a Gaussian ring:

        .. math::

           \frac{\mathrm{d}\ln\Sigma}{\mathrm{d}\ln r} = -A x (x - 1),

        where :math:`x = r/r_s` is the dimensionless radius and
        :math:`A = r_s^2/(2\sigma_\mathrm{ring}^2)`.

        Parameters
        ----------
        r : array_like or float
            Radius [kpc].

        Returns
        -------
        ndarray or float
            Logarithmic slope :math:`\mathrm{d}\ln\Sigma / \mathrm{d}\ln r`
            evaluated at each input radius.

        Notes
        -----
        This quantity is commonly used in pressure-support and asymmetric-drift
        calculations (e.g., Burkert et al. 2010).

        References
        ----------
        Burkert, A., et al. 2010, ApJ, 725, 2324
        """
        x = self._normalized_radius(r)
        return - 1/2 * 2 * self.A * x * (x - 1)


class LightGaussianRingProfile(GaussianRingProfile):
    r"""
    Light-only Gaussian ring profile (zero mass).

    This class creates a Gaussian ring profile with zero mass, useful for
    generating light profiles without associated mass. All methods from
    :class:`GaussianRingProfile` are inherited, but the mass is set to zero.
    It contributed to the total flux but has no kinematic effect.

    Parameters
    ----------
    r_s : float, optional
        Scale (peak) radius where the ring density is maximum [kpc]. At least
        one of ``r_s`` or ``h`` must be provided.
    h : float, optional
        Shape parameter defined as :math:`h = r_s / \mathrm{FWHM}_\mathrm{ring}`.
        At least one of ``r_s`` or ``h`` must be provided.
    FWHM_ring : float, optional
        Full Width at Half Maximum of the Gaussian ring [kpc].
    sigma_ring : float, optional
        Standard deviation of the Gaussian ring [kpc].
    lookup : bool, optional
        If ``True``, use lookup tables for circular velocity calculations.
        Default is ``True``.

    Notes
    -----
    This class is primarily used for generating light profiles that can be
    scaled independently of mass profiles.
    """
    def __init__(self, r_s=None, h=None, FWHM_ring=None, sigma_ring=None, lookup=True):
        # Call the parent class constructor
        mass = 0.
        super().__init__(mass=mass, r_s=r_s, h=h, FWHM_ring=FWHM_ring, sigma_ring=sigma_ring, lookup=lookup)


class LightSersicProfile(SersicProfile):
    r"""
    Light-only Sérsic profile (zero mass).

    This class creates a Sérsic profile with zero mass, useful for generating
    light profiles without associated mass. All methods from
    :class:`SersicProfile` are inherited, but the mass is set to zero.
    It contributed to the total flux but has no kinematic effect.
    
    Parameters
    ----------
    r_eff : float, optional
        Effective (half-light) radius [kpc]. If provided together with ``r_s``,
        ``r_s`` takes precedence. At least one of ``r_eff`` or ``r_s`` must be given.
    r_s : float, optional
        Scale radius [kpc]. If not given, it is derived from ``r_eff``.
        At least one of ``r_eff`` or ``r_s`` must be given.
    n : float, optional
        Sérsic index, determining the concentration of the profile. Default is 1.
    lookup : bool, optional
        If ``True``, use lookup tables for circular velocity calculations.
        Default is ``True``.

    Notes
    -----
    This class is primarily used for generating light profiles that can be
    scaled independently of mass profiles.
    """
    def __init__(self, r_eff=None, r_s=None, n=1., lookup=True):
        # Call the parent class constructor
        mass = 0.
        super().__init__(mass=mass, r_eff=r_eff, r_s=r_s, n=n, lookup=lookup)


class LightFreemanDiskProfile(FreemanDisk):
    r"""
    Light-only exponential disk profile (zero mass).

    This class creates an exponential disk profile with zero mass, useful for
    generating light profiles without associated mass. All methods from
    :class:`FreemanDisk` are inherited, but the mass is set to zero.

    Parameters
    ----------
    r_eff : float, optional
        Effective (half-light) radius [kpc]. If provided together with ``r_s``,
        ``r_s`` takes precedence. At least one of ``r_eff`` or ``r_s`` must be given.
    r_s : float, optional
        Scale radius (exponential scale length) [kpc]. If not given, it is derived
        from ``r_eff``. At least one of ``r_eff`` or ``r_s`` must be given.

    Notes
    -----
    This class is primarily used for generating light profiles that can be
    scaled independently of mass profiles.
    """
    def __init__(self, r_eff=None, r_s=None):
        # Call the parent class constructor
        mass = 0.
        super().__init__(mass=mass, r_eff=r_eff, r_s=r_s)
