from time import time_ns
import numpy as np
import astropy.constants as c
from astropy.cosmology import Planck18
# from scipy.special import j0, j1, k0
# from scipy.integrate import quad
import warnings
import logging

from RotCurves.base_utils import integrate_quad_list, solve_numerical_using_brentq

# Define constants as global variables
G_CONST = c.G.to('kpc km2 / (s2 Msun)').value

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')


class SurfaceDensityProfile:
    r"""
    Axisymmetric surface-density profile following any smooth profile function.

    The class provides helpers for **dimensionless** functions and their **physical** scale values.
    We define the **dimensionless radius** as:

    .. math::

        x = r / r_s

    and the following **dimensionless** functions:

    .. math::

       f_{\Sigma}(x) \equiv \frac{\Sigma(x)}{\Sigma_0},

    .. math::

       f_{M}(<x) \equiv \int_{0}^{x} t \, f_{\Sigma}(t)\, dt,

    .. math::

       f^2_{v}(x) \equiv -x \int_{0}^{\infty}  dk J_1(k r_s x)\, k\, \left[ \int_0^{\infty} J_0(ku) f_{\Sigma}(u) u du \right],

    The corresponding **physical** quantities are:

    .. math::

       \Sigma(r) = \Sigma_0 \, f_{\Sigma}(r/r_s), \qquad
       M(<r)     = M_s \, f_{M}(<r/r_s), \qquad
       v_c(r)   = v_s \, f_{v}(r/r_s)

    And the scale values are:

    .. math::

       \Sigma_0 \equiv \frac{M}{2\pi r_s^2\, f_{M}(<\infty)}, \qquad
       M_s \equiv 2 \pi \Sigma_0 r_s^2, \qquad
       v_s \equiv \sqrt{\frac{G \, M_s}{r_s}}

    The default profile uses an exponential surface density profile (Freeman 1970):

    .. math::

       f_{\Sigma}(x) = e^{-x}.

    Parameters
    ----------
    mass : float
        Total mass associated with the component (e.g., solar masses).
    r_eff : float, optional
        Effective (half-mass or half-light) radius in the same length units as
        ``r_s`` (e.g., kpc). If provided together with ``r_s``, ``r_s`` takes
        precedence. At least one of ``r_eff`` or ``r_s`` must be given.
    r_s : float, optional
        Scale radius (e.g., exponential disk scale length) in length units
        (e.g., kpc). If not given, it is derived from ``r_eff``.
    surface_density_function : callable, optional
        Callable returning the **dimensionless** surface density
        :math:`f_{\Sigma}(x)` as a function of :math:`x=r/r_s`. If ``None``,
        the default exponential profile is used, :math:`f_{\Sigma}(x)=\exp(-x)`.
    q0 : float, optional
        Intrinsic axis ratio of the mass distribution (0 ≤ q0 ≤ 1).
        Reserved for use in projected-light utilities. Default is 0.
    mass_to_light : float, optional
        Mass-to-light ratio used when generating a light profile from the mass
        profile (arbitrary but consistent units). Default is 1.
    verbose : bool, optional
        If True, enable verbose behavior (e.g., warnings, logging). Default is False.

    Attributes
    ----------
    mass : float
        Total mass of the component.
    r_eff : float
        Effective radius (kpc).
    r_s : float
        Scale radius (kpc). Guaranteed to be set after initialization.
    q0 : float
        Intrinsic axis ratio.
    mass_to_light : float
        Mass-to-light ratio used for light-profile scaling.
    surface_density_function : callable
        Function returning :math:`f_{\Sigma}(x)`.
    scale_density : float
        Surface-density normalization :math:`\Sigma_0` (:math:`M_\odot \, kpc^{-2}`).
    scale_mass : float
        Mass scaling factor :math:`M_s` so that :math:`M(<x) = M_s f_{M}(x)` (:math:`M_\odot`).
    scale_velocity : float
        Velocity scaling :math:`v_s` so that :math:`v_c(r) = v_s\, f_{v}(x)` (:math:`km \, s^{-1})`.

    Notes
    -----
    By default, the profile is an exponential disk (Freeman 1970). Other profiles
    can be implemented by providing a custom ``surface_density_function``.
    The gravitational constant ``G_CONST`` is in units of :math:`kpc \, km^2 \, s^{-2} \, M_\odot^{-1}`.

    Examples
    --------
    Create an exponential disk of mass 1e10 Msun and scale length 2 kpc:

    >>> prof = SurfaceDensityProfile(mass=1e10, r_s=2.0)
    >>> r = 8.0  # kpc
    >>> sigma = prof.surface_density(r)         # Σ(r)
    >>> m_r = prof.menc(r)                      # M(<r)
    >>> v_c = prof.vcirc(r)                     # circular velocity at r
    """

    def __init__(self, mass, r_eff=None, r_s=None, surface_density_function=None,
                 q0=0., mass_to_light=1., verbose=False):
        self.mass = mass
        self.r_eff = r_eff
        self.r_s = r_s
        self.q0 = q0
        self.mass_to_light = mass_to_light
        self.verbose = verbose

        # Set the default surface density function to an exponential "Freeman Disk" if not provided (e.g., Freeman+1970)
        if surface_density_function is None:
            self.surface_density_function = self._default_surface_density_function
        else:
            self.surface_density_function = surface_density_function

        # Handle cases where both radii are given
        if self.r_eff is not None and self.r_s is not None:
            warnings.warn("Both r_eff and r_s are given. Using r_s.", UserWarning)
        elif self.r_s is None and self.r_eff is not None:
            self.r_s = self._calculate_scale_radius_from_effective()
        elif self.r_eff is None and self.r_s is not None:
            self.r_eff = self._calculate_effective_radius_from_scale()
        else:
            raise ValueError("Either r_s or r_eff must be provided.")

        self.scale_density = self._scale_density()
        self.scale_mass = self._scale_mass()
        self.scale_velocity = self._scale_velocity()

    def _default_surface_density_function(self, r):
        r"""
        Return the default dimensionless surface density :math:`f_{\Sigma}(x)`.

        Notes
        -----
        The default is the exponential (Freeman) disk:

        .. math::

           f_{\Sigma}(x) = e^{-x},

        with :math:`x = r / r_s`.
        """
        x = self._normalized_radius(r)
        return np.exp(-x)

    def _normalized_radius(self, r):
        """Return the dimensionless radius :math:`x = |r| / r_s`."""
        return np.abs(r) / self.r_s

    def _calculate_scale_radius_from_effective(self):
        """
        Infer :math:`r_s` from :math:`r_\mathrm{eff}` by solving the half-mass
        condition in dimensionless form.

        Notes
        -----
        We solve for :math:`r_s` such that

        .. math::

           f_{M}\!\left(<\frac{r_\mathrm{eff}}{r_s}\right)
           \;=\; \tfrac{1}{2}\, f_{M}(<\infty).

        Returns
        -------
        float
            Estimated scale radius :math:`r_s` in the same units as ``r_eff``.
        """
        func = lambda r_s: self.menc_dimless(self.r_eff / r_s) - 0.5 * self.menc_dimless(np.inf)
        return solve_numerical_using_brentq(func, p0=self.r_eff)

    def _calculate_effective_radius_from_scale(self):
        """
        Infer :math:`r_\mathrm{eff}` from :math:`r_s` by solving the half-mass
        condition in dimensionless form.

        Notes
        -----
        We solve for :math:`r_\mathrm{eff}` such that

        .. math::

           f_{M}\!\left(<\frac{r_\mathrm{eff}}{r_s}\right)
           \;=\; \tfrac{1}{2}\, f_{M}(<\infty).

        Returns
        -------
        float
            Estimated effective radius :math:`r_\mathrm{eff}` in the same units as ``r_s``.
        """
        func = lambda r_eff: self.menc_dimless(r_eff / self.r_s) - 0.5 * self.menc_dimless(np.inf)
        # func = lambda r_eff: self.menc_dimless(r_eff / self.r_s) - 0.5 * self.menc_dimless(3*self.r_s)
        return solve_numerical_using_brentq(func, p0=self.r_s)

    def _scale_density(self):
        r"""
        Compute the surface-density normalization :math:`\Sigma_0`.

        Notes
        -----
        Using :math:`M_\mathrm{tot} = \mathrm{mass}`, we enforce

        .. math::

           M_\mathrm{tot}
           \;=\; \left(2\pi \Sigma_0 r_s^2\right)\, f_{M}(<\infty),

        hence

        .. math::

           \Sigma_0 \;=\; \frac{M_\mathrm{tot}}{2\pi r_s^2\, f_{M}(<\infty)}.

        Returns
        -------
        float
            Scale surface density :math:`\Sigma_0` (mass / length\ :sup:`2`).
        """
        return self.mass / (2 * np.pi * self.r_s ** 2 * self.menc_dimless(np.inf))[0]

    def _scale_intensity(self):
        r"""
        Compute the light-profile normalization from :math:`\Sigma_0` and M/L.

        Notes
        -----
        The intensity normalization :math:`I_0` satisfies

        .. math::

           I_0 \;=\; \left(\mathrm{M/L}\right)\, \Sigma_0.

        Returns
        -------
        float
            Intensity normalization (arbitrary but consistent units).
        """
        if self.mass == 0.:
            return 1.
        else:
            return self.scale_density * self.mass_to_light

    def _scale_mass(self):
        r"""
        Compute the mass scaling factor :math:`2\pi \Sigma_0 r_s^2`.

        Notes
        -----
        This is the multiplicative factor converting :math:`f_{M}(<x)` to the
        physical enclosed mass :math:`M(<r)`.

        Returns
        -------
        float
            Mass scale factor.
        """
        return 2 * np.pi * self.scale_density * self.r_s**2

    def _scale_velocity(self):
        r"""
        Compute the velocity scaling :math:`v_s`.

        Notes
        -----
        From the definition

        .. math::

           v_s \;\equiv\; \sqrt{\frac{G \, M_s}{r_s}},

        which implies :math:`v_c(r) = v_s\, f_v(r/r_s)`.

        Returns
        -------
        float
            Velocity scale :math:`v_s` (units consistent with ``G_CONST`` and ``r_s``).
        """
        return np.sqrt(G_CONST * self.scale_mass / self.r_s)

    def _is_massive(self):
        """
        Return whether the component carries positive mass.

        Returns
        -------
        bool
            True if ``mass > 0``.
        """
        return self.mass > 0.

    def surface_density_dimless(self, x):
        r"""
        Dimensionless surface density :math:`f_{\Sigma}(x)` at dimensionless radius :math:`x`.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            :math:`f_{\Sigma}(x)`.
        """
        return self.surface_density_function(x)

    def surface_density(self, r):
        r"""
        Physical surface density :math:`\Sigma(r) = \Sigma_0 \, f_{\Sigma}(r/r_s)`.

        Parameters
        ----------
        r : array_like or float
            Radius (length units; e.g., kpc).

        Returns
        -------
        ndarray or float
            Surface density in mass / length\ :sup:`2`.
        """
        x = self._normalized_radius(r)
        return self.surface_density_dimless(x) * self.scale_density

    def menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_{M}(<x)`.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Notes
        -----
        By definition,
        .. math::

           f_{M}(<x) \;=\; \int_{0}^{x} t \, f_{\Sigma}(t)\, dt.

        Returns
        -------
        ndarray
            :math:`f_{M}(<x)`. Always returned as a 1-D NumPy array; for scalar
            ``x`` the shape is ``(1,)`` to match ``integrate_quad_list``.
        """
        mass = integrate_quad_list(lambda t: t * self.surface_density_dimless(t), 0, x)
        return mass

    def menc(self, r):
        r"""
        Physical enclosed mass :math:`M(<r)`.

        Parameters
        ----------
        r : array_like or float
            Radius (length units; e.g., kpc).

        Notes
        -----
        Using :math:`M(<r) = \left(2\pi \Sigma_0 r_s^2\right) f_M(<r/r_s)`.

        Returns
        -------
        ndarray or float
            Enclosed mass in the same units as ``mass``.
        """
        x = self._normalized_radius(r)
        return self.menc_dimless(x) * self.scale_mass

    # def circular_velocity_dimless(self, x):
    #     r"""
    #     Dimensionless circular velocity :math:`f_{v}(x)`.
    #
    #     Parameters
    #     ----------
    #     x : array_like
    #         Dimensionless radius values.
    #
    #     Returns
    #     -------
    #     ndarray
    #         :math:`f_{v}(x)`.
    #     """
    #     ...

    def vcirc2_dimless(self, x):
        r"""
        Dimensionless squared circular velocity :math:`f_{v}^2(x)`.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius.

        Notes
        -----
        From the definition of :math:`f_v(x)`,

        .. math::

           f_{v}^2(x) \;=\; -x \int_{0}^{\infty}  dk J_1(k r_s x)\, k\, \left[ \int_0^{\infty} J_0(ku) f_{\Sigma}(u) u du \right]

        Returns
        -------
        ndarray or float
            :math:`f_{v}^2(x)`.
        """
        return np.divide(self.menc_dimless(x), x, out=np.zeros_like(x), where=x!=0)

    def vcirc2(self, r):
        r"""
        Physical squared circular velocity :math:`v_c^2(r)`.

        Parameters
        ----------
        r : array_like or float
            Radius (length units; e.g., kpc).

        Notes
        -----
        Using :math:`v_c(r) = v_s\, f_v(r/r_s)`, we have

        .. math::

           v_c^2(r) \;=\; v_s^2 \, f_v^2(r/r_s).

        Returns
        -------
        ndarray or float
            :math:`v_c^2(r)` in (velocity)\ :sup:`2` units.
        """
        x = self._normalized_radius(r)
        return self.vcirc2_dimless(x) * self.scale_velocity**2

    def vcirc_dimless(self, x):
        r"""
        Dimensionless circular velocity :math:`f_{v}(x)`.

        Negative values of :math:`f_{v}^2` are clipped to zero to avoid NaNs.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius.

        Returns
        -------
        ndarray or float
            :math:`f_{v}(x)`.
        """
        return np.sqrt(np.maximum(self.vcirc2_dimless(x), 0))

    def vcirc(self, r):
        r"""
        Physical circular velocity :math:`v_c(r)`.

        Parameters
        ----------
        r : array_like or float
            Radius (length units; e.g., kpc).

        Notes
        -----
        With :math:`v_c(r) = v_s\, f_v(r/r_s)` and :math:`v_s` defined above.

        Returns
        -------
        ndarray or float
            Circular velocity in units consistent with ``G_CONST`` and ``r_s``.
        """
        x = self._normalized_radius(r)
        return self.vcirc_dimless(x) * self.scale_velocity

    def light_profile(self, r, r2=None):
        r"""
        Projected light profile proportional to :math:`\Sigma(r)` and M/L.

        Parameters
        ----------
        r : array_like or float
            Radius along the major axis (length units).
        r2 : array_like or float, optional
            Radius along the minor/second axis for elliptical profiles. If
            provided, the profile is evaluated at :math:`\sqrt{x^2 + x_2^2}`
            in dimensionless units.

        Notes
        -----
        The profile is

        .. math::

           I(r) = \left(\mathrm{M/L}\right)\, \Sigma_0 \, f_{\Sigma}(x),

        where :math:`x = r/r_s` (or :math:`x = \sqrt{x^2 + x_2^2}` if ``r2`` is given).

        Returns
        -------
        ndarray or float
            Intensity profile in arbitrary but consistent units.
        """
        scale_intensity = self._scale_intensity()
        x = self._normalized_radius(r)
        if r2 is None:
            lprof = self.surface_density_dimless(x)
        else:
            x2 = self._normalized_radius(r2)
            lprof = self.surface_density_dimless(np.sqrt(x**2 + x2**2))

        return scale_intensity * lprof

    def dlnrho_dlnr(self, r):
        r"""
        Logarithmic slope of the (surface) density profile at radius :math:`r`:
        :math:`\mathrm{d}\ln\Sigma / \mathrm{d}\ln r`.

        Parameters
        ----------
        r : array_like
            Radius (length units; e.g., kpc).

        Returns
        -------
        ndarray
            The local logarithmic slope evaluated at each input radius.

        Notes
        -----
        This quantity is commonly used in pressure-support and asymmetric-drift
        calculations (e.g., Burkert et al. 2010). The numerical gradient is
        taken with respect to :math:`\ln r`; values at or extremely close to
        :math:`r = 0` may be ill-defined. Expects ``r`` to be a monotonically
        increasing 1D array.
        """
        x = self._normalized_radius(r)
        dlnrho_dlnr = np.gradient(np.log(self.surface_density_dimless(x)), np.log(x))
        return dlnrho_dlnr


class DarkMatterHaloProfile:

    def __init__(self, mass, z=None, r_vir=None, r_s=None, concentration=None, scale_density=None,
                 virial_overdensity=200, density_function=None, adiabatic_contraction=False):
        self.z = z
        self.mass = mass
        self.r_vir = r_vir
        self.r_s = r_s
        self.c = concentration
        self.scale_density = scale_density
        self.virial_overdensity = virial_overdensity
        self.density_function = density_function
        self.adiabatic_contraction = adiabatic_contraction

        # Set the default redshift to 0 if not provided
        if z is None:
            self.z = 0

        # Set the critical density of the Universe at redshift z
        self.rho_crit = Planck18.critical_density(self.z).to('Msun / kpc3').value

        # Set the default density function to NFW if not provided (e.g., Navarro+1995)
        if density_function is None:
            self.density_function = self._default_density_function

        # Calculate missing parameters from the given ones
        # parameters: c, r_s, r_vir, scale_density, virial_overdensity
        class _ParameterSolver:
            def __init__(self, outer_self):
                self.mass = outer_self.mass
                self.rho_crit = outer_self.rho_crit
                self.menc_dimless = outer_self._menc_dimless
                # Map of combinations to calculation functions
                self.method_map = {
                    ('c', 'r_s'): self.calculate_from_c_rs,
                    ('c', 'r_vir'): self.calculate_from_c_rvir,
                    ('c', 'scale_density'): self.calculate_from_c_scale_density,
                    ('c', 'virial_overdensity'): self.calculate_from_c_virial_overdensity,
                    ('r_vir', 'r_s'): self.calculate_from_rvir_rs,
                    ('r_vir', 'scale_density'): self.calculate_from_rvir_scale_density,
                    ('r_s', 'scale_density'): self.calculate_from_rs_scale_density,
                    ('r_s','virial_overdensity'): self.calculate_from_rs_virial_overdensity,
                    ('scale_density', 'virial_overdensity'): self.calculate_from_scale_density_virial_overdensity,
                    }

            def calculate(self, **kwargs):
                """
                Calculate the three missing parameters given two known ones.
                """
                # Extract given and missing parameter names
                given = {k: v for k, v in kwargs.items() if v is not None}
                missing = [k for k, v in kwargs.items() if v is None]

                # Check if exactly two values are given
                if len(given) != 2 or len(missing) != 3:
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
            def calculate_from_c_rs(self, c, r_s):
                r_vir = c * r_s
                virial_overdensity = self.mass / (4/3 * np.pi * r_vir**3 * self.rho_crit)
                scale_density = c**3 / self.menc_dimless(c) * virial_overdensity * self.rho_crit
                return r_vir, scale_density, virial_overdensity

            def calculate_from_c_rvir(self, c, r_vir):
                r_s = r_vir / c
                virial_overdensity = self.mass / (4/3 * np.pi * r_vir**3 * self.rho_crit)
                scale_density = c**3 / self.menc_dimless(c) * virial_overdensity * self.rho_crit
                return r_s, scale_density, virial_overdensity

            def calculate_from_c_scale_density(self, c, scale_density):
                virial_overdensity = scale_density/self.rho_crit * self.menc_dimless(c)/c**3
                r_vir = (3 * self.mass / (4 * np.pi * virial_overdensity * self.rho_crit))**(1/3)
                r_s = r_vir / c
                return r_s, r_vir, virial_overdensity

            def calculate_from_c_virial_overdensity(self, c, virial_overdensity):
                r_vir = (3 * self.mass / (4 * np.pi * virial_overdensity * self.rho_crit))**(1/3)
                r_s = r_vir / c
                scale_density = c**3 / self.menc_dimless(c) * virial_overdensity * self.rho_crit
                return r_s, r_vir, scale_density

            def calculate_from_rvir_rs(self, r_vir, r_s):
                c = r_vir / r_s
                virial_overdensity = self.mass / (4/3 * np.pi * r_vir**3 * self.rho_crit)
                scale_density = c**3 / self.menc_dimless(c) * virial_overdensity * self.rho_crit
                return c, scale_density, virial_overdensity

            def calculate_from_rvir_scale_density(self, r_vir, scale_density):
                virial_overdensity = self.mass / (4 / 3 * np.pi * r_vir ** 3 * self.rho_crit)
                c = solve_numerical_using_brentq(lambda c: scale_density - c**3/self.menc_dimless(c)*virial_overdensity*self.rho_crit, p0=5)
                r_s = r_vir / c
                return c, r_s, virial_overdensity

            def calculate_from_rs_scale_density(self, r_s, scale_density):
                func = lambda c: scale_density - c**3/self.menc_dimless(c) * (self.mass / (4/3*np.pi * (c*r_s)**3))
                c = solve_numerical_using_brentq(func, p0=5)
                r_vir = c * r_s
                virial_overdensity = self.mass / (4/3 * np.pi * r_vir**3 * self.rho_crit)
                return c, r_vir, virial_overdensity

            def calculate_from_rs_virial_overdensity(self, r_s, virial_overdensity):
                r_vir = (3 * self.mass / (4 * np.pi * virial_overdensity * self.rho_crit))**(1/3)
                c = r_vir / r_s
                scale_density = c**3 / self.menc_dimless(c) * virial_overdensity * self.rho_crit
                return c, r_vir, scale_density

            def calculate_from_scale_density_virial_overdensity(self, scale_density, virial_overdensity):
                r_vir = (3 * self.mass / (4 * np.pi * virial_overdensity * self.rho_crit))**(1/3)
                c = solve_numerical_using_brentq(lambda c: scale_density - c**3/self.menc_dimless(c) * virial_overdensity * self.rho_crit, p0=5)
                r_s = r_vir / c
                return c, r_s, r_vir
        parameters = _ParameterSolver(self).calculate(c=concentration, r_s=r_s, r_vir=r_vir, scale_density=scale_density,
                                                      virial_overdensity=virial_overdensity)
        self.c = float(parameters['c'])
        self.r_s = float(parameters['r_s'])
        self.r_vir = float(parameters['r_vir'])
        self.scale_density = float(parameters['scale_density'])
        self.virial_overdensity = float(parameters['virial_overdensity'])

        # Calculate the scale density, mass, and velocity
        self.scale_mass = self._scale_mass()
        self.scale_velocity = self._scale_velocity()

    def _default_density_function(self, x):
        return x**-1 * (1 + x)**-2

    def _normalized_radius(self, r):
        """Return the dimensionless radius :math:`x = |r| / r_s`."""
        return np.abs(r) / self.r_s

    def _scale_mass(self):
        return 4/3 * np.pi * self.scale_density * self.r_s ** 3

    def _scale_velocity(self):
       return np.sqrt(G_CONST * self.scale_mass / self.r_s)

    def _density_dimless(self, x):
        return self.density_function(x)

    def density(self, r):
        x = self._normalized_radius(r)
        return self._density_dimless(x) * self.scale_density

    def _menc_dimless(self, x):
        mass = integrate_quad_list(lambda t: 3 * t ** 2 * self._density_dimless(t), 0, x)
        return mass

    def menc(self, r):
        x = self._normalized_radius(r)
        return self._menc_dimless(x) * self.scale_mass

    def _vcirc2_dimless(self, x):
        return np.divide(self._menc_dimless(x), x, out=np.zeros_like(x), where=x != 0)

    def vcirc2(self, r):
        x = self._normalized_radius(r)
        return self._vcirc2_dimless(x) * self.scale_velocity ** 2

    def _vcirc_dimless(self, x):
        return np.sqrt(np.maximum(self._vcirc2_dimless(x), 0))

    def vcirc(self, r):
        x = self._normalized_radius(r)
        return self._vcirc_dimless(x) * self.scale_velocity




























# class SurfaceDensityProfile_bak:
#     def __init__(self, mass, r_eff=None, r_s=None, surface_density_function=None,
#                  q0=0., mass_to_light=1., verbose=False):
#         self.mass = mass
#         self.r_eff = r_eff
#         self.r_s = r_s
#         self.q0 = q0
#         self.mass_to_light = mass_to_light
#         self.verbose = verbose
#
#         # Set the default surface density function to an exponential "Freeman Disk" if not provided (e.g., Freeman+1970)
#         if surface_density_function is None:
#             self.surface_density_function = self._default_surface_density_function
#         else:
#             self.surface_density_function = surface_density_function
#
#         # Handle cases where both radii are given
#         if self.r_eff is not None and self.r_s is not None:
#             warnings.warn("Both r_eff and r_s are given. Using r_s.", UserWarning)
#         elif self.r_s is None and self.r_eff is not None:
#             self.r_s = self._calculate_scale_radius_from_effective()
#         elif self.r_eff is None and self.r_s is not None:
#             self.r_eff = self._calculate_effective_radius_from_scale()
#         else:
#             raise ValueError("Either r_s or r_eff must be provided.")
#
#         self.scale_density = self._scale_density()
#         self.scale_mass = self._scale_mass()
#         self.scale_velocity = self._scale_velocity()
#
#     def _default_surface_density_function(self, r):
#         # Default exponential Freeman Disk surface density profile
#         x = self._calculate_normalized_radius(r)
#         return np.exp(-x)
#
#     def _calculate_normalized_radius(self, r):
#         # Normalized radius x = r / r_s
#         return np.abs(r) / self.r_s
#
#     def _calculate_scale_radius_from_effective(self):
#         # Placeholder formula to convert effective radius to scale radius
#         func = lambda r_s: self.menc_dimless(self.r_eff / r_s) - 0.5 * self.menc_dimless(np.inf)
#         return solve_numerical_using_brentq(func, p0=self.r_eff)
#
#     def _calculate_effective_radius_from_scale(self):
#         # Placeholder formula to convert scale radius to effective radius
#         func = lambda r_eff: self.menc_dimless(r_eff / self.r_s) - 0.5 * self.menc_dimless(np.inf)
#         # func = lambda r_eff: self.menc_dimless(r_eff / self.r_s) - 0.5 * self.menc_dimless(3*self.r_s)
#         return solve_numerical_using_brentq(func, p0=self.r_s)
#
#     def _scale_density(self):
#         # Placeholder for scale density calculation
#         return self.mass / (2 * np.pi * self.r_s ** 2 * self.menc_dimless(np.inf))[0]
#
#     def _scale_intensity(self):
#         # Placeholder for scale intensity calculation
#         if self.mass == 0.:
#             return 1.
#         else:
#             return self.scale_density * self.mass_to_light
#
#     def _scale_mass(self):
#         # Placeholder for scale mass calculation
#         return 2 * np.pi * self.scale_density * self.r_s**2
#
#     def _scale_velocity(self):
#         # Use the global G_CONST
#         return np.sqrt(G_CONST * self.scale_mass / self.r_s)
#
#     def _is_massive(self):
#         # Check if the mass is massive
#         return self.mass > 0.
#
#     def surface_density_dimless(self, x):
#         return self.surface_density_function(x)
#
#     def surface_density(self, r):
#         x = self._calculate_normalized_radius(r)
#         return self.surface_density_dimless(x) * self.scale_density
#
#     def menc_dimless(self, x):
#         # Integration using scipy.quad
#         mass = integrate_quad_list(lambda t: t * self.surface_density_dimless(t), 0, x)
#         return mass
#
#     def menc(self, r):
#         x = self._calculate_normalized_radius(r)
#         return self.menc_dimless(x) * self.scale_mass
#
#     def vcirc2_dimless(self, x):
#         # Velocity calculation using enclosed mass
#         return np.divide(self.menc_dimless(x), x, out=np.zeros_like(x), where=x!=0)
#
#     def vcirc2(self, r):
#         x = self._calculate_normalized_radius(r)
#         return self.vcirc2_dimless(x) * self.scale_velocity**2
#
#     def vcirc_dimless(self, x):
#         """
#         Calculate the circular velocity for an array of dimensionless radii `x`.
#         Negative vcirc2 values are set to zero to avoid NaNs.
#         """
#
#         return np.sqrt(np.maximum(self.vcirc2_dimless(x), 0))
#
#     def vcirc(self, r):
#         x = self._calculate_normalized_radius(r)
#         return self.vcirc_dimless(x) * self.scale_velocity
#
#     def light_profile(self, r, r2=None):
#         scale_intensity = self._scale_intensity()
#         x = self._calculate_normalized_radius(r)
#         if r2 is None:
#             lprof = self.surface_density_dimless(x)
#         else:
#             x2 = self._calculate_normalized_radius(r2)
#             lprof = self.surface_density_dimless(np.sqrt(x**2 + x2**2))
#
#         return scale_intensity * lprof
#
#     def dlnrho_dlnr(self, r):
#         """
#         Calculate the logarithmic density slope at radius r.
#         Used in calculations of the pressure support (e.g., Burkert+2010)
#         """
#         x = self._calculate_normalized_radius(r)
#         dlnrho_dlnr = np.gradient(np.log(self.surface_density_dimless(x)), np.log(x))
#         return dlnrho_dlnr