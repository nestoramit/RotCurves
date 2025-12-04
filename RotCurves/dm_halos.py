import numpy as np
from time import time_ns

from RotCurves.base_classes import DarkMatterHaloProfile
from scipy.special import hyp2f1, gamma, gammainc

def halo_selector(component_type, **kwargs):
    """
    Factory function to create dark matter halo profile instances based on the specified type.

    Parameters
    ----------
    component_type : str
        Type of dark matter halo profile to create. Options are:
        'NFW', 'alpha-NFW', 'Burkert', 'Einasto', 'Dekel-Zhao'.
    **kwargs : dict
        Additional keyword arguments to pass to the halo profile constructor.

    Returns
    -------
    DarkMatterHaloProfile
        An instance of the specified dark matter halo profile.

    Raises
    ------
    ValueError
        If an unsupported halo type is provided.

    Examples
    --------
    Create an NFW halo:

    >>> halo = halo_selector('NFW',mass=1e12,concentration=10,z=0)

    Create a Burkert halo:

    >>> halo = halo_selector('Burkert',mass=1e12,concentration=10,z=0)
    """
    component_type = component_type.lower()
    if component_type in ['nfw']:
        return NFWHalo(**kwargs)
    elif component_type in ['alpha-nfw', 'alpha_nfw', 'alphanfw']:
        return alhpaNFWHalo(**kwargs)
    elif component_type in ['burkert']:
        return BurkertHalo(**kwargs)
    elif component_type in ['einasto']:
        return EinastoHalo(**kwargs)
    elif component_type in ['dekel-zhao', 'dekel_zhao', 'dekelzhao', 'dz']:
        return DekelZhaoHalo(**kwargs)
    else:
        raise ValueError(f"Unsupported halo type: {component_type}. Supported types are: "
                         "'NFW', 'alpha-NFW', 'Burkert', 'Einasto', 'Dekel-Zhao'.")

class NFWHalo(DarkMatterHaloProfile):
    r"""
    Navarro-Frenk-White (NFW) dark matter halo profile (Navarro et al. 1995).

    This class implements the NFW density profile, a widely used and well-studied
    model for dark matter halos in cosmological simulations.
    The density profile follows:

    .. math::

       \rho(x) = \frac{\rho_0}{x (1 + x)^2}

    where :math:`x = r/r_s` is the dimensionless radius and :math:`r_s` is the
    scale radius. The concentration parameter is defined as:
    
    .. math::

       c \equiv \frac{r_{\rm vir}}{r_s}

    and :math:`r_{\rm vir}` is the virial radius.

    The profile has a cuspy inner region with :math:`\rho \propto r^{-1}`
    and an outer region with :math:`\rho \propto r^{-3}`.

    The dimensionless enclosed mass is:

    .. math::

       f_M(<x) = 3\left[\ln(1+x) - \frac{x}{1+x}\right],

    and the scale mass is:

    .. math::

       M_s = \frac{4 \pi}{3} \rho_0 r_s^3 = \frac{M}{3\left[\ln(1+c) - \frac{c}{1+c}\right]}

    Parameters
    ----------
    z : float, optional
        Redshift used to set :math:`\rho_{\rm crit}(z)`. Default is 0.0.
    mass : float, optional
        Halo mass :math:`M` [:math:`M_\odot`], interpreted as the mass within
        :math:`r_{\rm vir}` given ``virial_overdensity``. Default is :math:`10^{12} M_\odot`.
    concentration : float, optional
        Concentration parameter :math:`c \equiv r_{\rm vir}/r_s`. Default is 10.
    virial_overdensity : float, optional
        Virial overdensity :math:`\Delta_{\rm vir}` relative to critical density.
        Default is 200.
    r_vir : float, optional
        Virial radius :math:`r_{\rm vir}` [kpc]. If not provided, it is calculated
        from mass and virial overdensity.
    r_s : float, optional
        Scale radius :math:`r_s` [kpc]. If not provided, it is calculated from
        concentration and virial radius.
    scale_density : float, optional
        Density normalization :math:`\rho_s` [:math:`M_\odot\,\mathrm{kpc}^{-3}`].
        If not provided, it is calculated from the other parameters.
    adiabatic_contraction : bool, optional
        Placeholder flag for adiabatic-contraction adjustments (not applied in
        current implementation). Default is ``False``.

    Notes
    -----
    

    References
    ----------
    Navarro, J. F., Frenk, C. S., & White, S. D. M. 1995, MNRAS, 275, 720

    Examples
    --------
    Create an NFW halo with mass :math:`10^{12} M_\odot` and concentration 10:

    >>> halo = NFWHalo(mass=1e12, concentration=10, z=0)
    >>> r = 10.0  # kpc
    >>> rho = halo.density(r)      # ρ(r)
    >>> m_r = halo.menc(r)         # M(<r)
    >>> v_c = halo.vcirc(r)        # circular velocity at r
    """
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200.,
                 r_vir=None, r_s=None, scale_density=None, adiabatic_contraction=False):
        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function,
                         adiabatic_contraction=adiabatic_contraction)

    def density_function(self, x):
        r"""
        Dimensionless density profile :math:`f_\rho(x)` for the NFW profile:

        .. math::

           f_\rho(x) = \frac{1}{x(1+x)^2},

        where :math:`x = r/r_s` is the dimensionless radius.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless density :math:`f_\rho(x)`.
        """
        return x**-1 * (1 + x)**-2

    def _menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_M(<x)` for the NFW profile:

        .. math::

           f_M(<x) = 3\left[\ln(1+x) - \frac{x}{1+x}\right].

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Returns
        -------
        ndarray
            Dimensionless enclosed mass :math:`f_M(<x)`.
        """
        return 3 * (np.log(1 + x) - x / (1 + x))

class alhpaNFWHalo(DarkMatterHaloProfile):
    r"""
    Generalized NFW (alpha-NFW) dark matter halo profile with variable inner slope, and fixed outer slope (= 3).

    This class implements a generalization of the NFW profile where the inner
    density slope :math:`\alpha` is a free parameter. 
    
    The density profile follows:

    .. math::

       \rho(x) = \frac{\rho_0}{x^{\alpha} (1 + x)^{(3-\alpha)}},

    where :math:`x = r/r_s` is the dimensionless radius, :math:`\alpha` is the
    inner slope parameter, and :math:`r_s` is the scale radius. When
    :math:`\alpha = 1`, this reduces to the standard NFW profile.

    The dimensionless enclosed mass is:

    .. math::

       f_M(<x) = \frac{3}{3-\alpha} x^{3-\alpha}
                {}_2F_1(3-\alpha, 3-\alpha; 4-\alpha; -x),

    where :math:`{}_2F_1` is the hypergeometric function.

    Parameters
    ----------
    z : float, optional
        Redshift used to set :math:`\rho_{\rm crit}(z)`. Default is 0.0.
    mass : float, optional
        Halo mass :math:`M` [:math:`M_\odot`], interpreted as the mass within
        :math:`r_{\rm vir}` given ``virial_overdensity``. Default is :math:`10^{12} M_\odot`.
    concentration : float, optional
        Concentration parameter :math:`c \equiv r_{\rm vir}/r_s`. Default is 10.
    virial_overdensity : float, optional
        Virial overdensity :math:`\Delta_{\rm vir}` relative to critical density.
        Default is 200.
    alpha : float, optional
        Inner slope parameter :math:`\alpha` of the density profile. The inner
        region has :math:`\rho \propto r^{-\alpha}`. Default is 1.0 (NFW).
    r_vir : float, optional
        Virial radius :math:`r_{\rm vir}` [kpc]. If not provided, it is calculated
        from mass and virial overdensity.
    r_s : float, optional
        Scale radius :math:`r_s` [kpc]. If not provided, it is calculated from
        concentration and virial radius.
    scale_density : float, optional
        Density normalization :math:`\rho_s` [:math:`M_\odot\,\mathrm{kpc}^{-3}`].
        If not provided, it is calculated from the other parameters.
    adiabatic_contraction : bool, optional
        Placeholder flag for adiabatic-contraction adjustments (not applied in
        current implementation). Default is ``False``.

    Notes
    -----
    The generalized NFW profile allows for different inner density slopes, which
    can be useful for modeling halos with cores (:math:`\alpha < 1`) or steeper
    cusps (:math:`\alpha > 1`). The outer slope remains :math:`-3` as in the
    standard NFW profile.

    References
    ----------
    Zhao, D. H., et al. 2003, MNRAS, 339, 12

    Examples
    --------
    Create a generalized NFW halo with inner slope :math:`\alpha = 0.5`:

    >>> halo = alhpaNFWHalo(mass=1e12, concentration=10, alpha=0.5, z=0)
    >>> r = 10.0  # kpc
    >>> rho = halo.density(r)      # ρ(r)
    """
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200., alpha=1.0,
                 r_vir=None, r_s=None, scale_density=None, adiabatic_contraction=False):
        self.alpha = alpha
        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function,
                         adiabatic_contraction=adiabatic_contraction)

    def density_function(self, x):
        r"""
        Dimensionless density profile :math:`f_\rho(x)` for the generalized NFW profile:

        .. math::

           f_\rho(x) = x^{-\alpha}(1+x)^{-(3-\alpha)},

        where :math:`x = r/r_s` is the dimensionless radius and :math:`\alpha` is
        the inner slope parameter.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless density :math:`f_\rho(x)`.
        """
        return x**-self.alpha * (1 + x)**-(3-self.alpha)

    def _menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_M(<x)` for the generalized NFW profile:

        .. math::

           f_M(<x) = \frac{3}{3-\alpha} x^{3-\alpha}
                    {}_2F_1(3-\alpha, 3-\alpha; 4-\alpha; -x),

        where :math:`{}_2F_1` is the hypergeometric function.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Returns
        -------
        ndarray
            Dimensionless enclosed mass :math:`f_M(<x)`.
        """
        return 3/(3-self.alpha) * (x**(3-self.alpha) * hyp2f1(3-self.alpha, 3-self.alpha, 4-self.alpha, -x))

class BurkertHalo(DarkMatterHaloProfile):
    r"""
    Burkert dark matter halo profile with a constant-density core (Burkert 1995).

    This class implements the Burkert profile, which features a constant-density
    core in the inner region, and an outer slope of -3. 
    
    The density profile follows:

    .. math::

       \rho(x) = \frac{\rho_0}{(1+x)(1+x^2)},

    where :math:`x = r/r_s` is the dimensionless radius and :math:`r_s` is the
    scale radius. The profile has a flat core (:math:`\rho \approx \rho_0`)
    for :math:`x \ll 1` and falls as :math:`\rho \propto r^{-3}` for :math:`x \gg 1`.

    The dimensionless enclosed mass is:

    .. math::

       f_M(<x) = \frac{3}{2}\left[\frac{1}{2}\ln(1+x^2) + \ln(1+x) - \arctan(x)\right].

    Parameters
    ----------
    z : float, optional
        Redshift used to set :math:`\rho_{\rm crit}(z)`. Default is 0.0.
    mass : float, optional
        Halo mass :math:`M` [:math:`M_\odot`], interpreted as the mass within
        :math:`r_{\rm vir}` given ``virial_overdensity``. Default is :math:`10^{12} M_\odot`.
    concentration : float, optional
        Concentration parameter :math:`c \equiv r_{\rm vir}/r_s`. Default is 10.
    virial_overdensity : float, optional
        Virial overdensity :math:`\Delta_{\rm vir}` relative to critical density.
        Default is 200.
    alpha : float, optional
        Unused parameter (kept for interface compatibility). Default is 1.0.
    r_vir : float, optional
        Virial radius :math:`r_{\rm vir}` [kpc]. If not provided, it is calculated
        from mass and virial overdensity.
    r_s : float, optional
        Scale radius :math:`r_s` [kpc]. If not provided, it is calculated from
        concentration and virial radius.
    scale_density : float, optional
        Density normalization :math:`\rho_s` [:math:`M_\odot\,\mathrm{kpc}^{-3}`].
        If not provided, it is calculated from the other parameters.
    adiabatic_contraction : bool, optional
        Placeholder flag for adiabatic-contraction adjustments (not applied in
        current implementation). Default is ``False``.

    Notes
    -----
    The Burkert profile is particularly useful for modeling dark matter halos
    in dwarf galaxies, where observations suggest the presence of constant-density
    cores rather than cusps. The core radius is approximately equal to the scale
    radius :math:`r_s`.

    References
    ----------
    Burkert, A. 1995, ApJ, 447, L25

    Examples
    --------
    Create a Burkert halo with mass :math:`10^{12} M_\odot` and concentration 10:

    >>> halo = BurkertHalo(mass=1e12, concentration=10, z=0)
    >>> r = 10.0  # kpc
    >>> rho = halo.density(r)      # ρ(r)
    >>> m_r = halo.menc(r)         # M(<r)
    """
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200., alpha=1.0,
                 r_vir=None, r_s=None, scale_density=None, adiabatic_contraction=False):
        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function,
                         adiabatic_contraction=adiabatic_contraction)

    def density_function(self, x):
        r"""
        Dimensionless density profile :math:`f_\rho(x)` for the Burkert profile:

        .. math::

           f_\rho(x) = \frac{1}{(1+x)(1+x^2)},

        where :math:`x = r/r_s` is the dimensionless radius.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless density :math:`f_\rho(x)`.
        """
        return ((1+x)*(1 + x**2))**-1

    def _menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_M(<x)` for the Burkert profile:

        .. math::

           f_M(<x) = \frac{3}{2}\left[\frac{1}{2}\ln(1+x^2) + \ln(1+x) - \arctan(x)\right].

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Returns
        -------
        ndarray
            Dimensionless enclosed mass :math:`f_M(<x)`.
        """
        return 3/2 * (1/2*np.log(1+x**2) + np.log(1+x) - np.arctan(x))

class EinastoHalo(DarkMatterHaloProfile):
    r"""
    Einasto dark matter halo profile (Einasto 1965), with a variable slope parametrization.

    This class implements the Einasto profile, a widely flexible profile 
    that can be used to model a wide range of dark matter halos, 
    including both cores and cusps. 
    
    The density profile follows:

    .. math::

       \rho(x) = \rho_0 \cdot \exp\left(-x^{1/n}\right),

    where :math:`x = r/r_s` is the dimensionless radius, :math:`r_s` is the scale
    radius, and :math:`n` is the Einasto index that controls the shape of the
    profile. Higher values of :math:`n` correspond to more centrally concentrated
    profiles.

    The dimensionless enclosed mass is:

    .. math::

       f_M(<x) = 3n \cdot gamma(3n, x^{1/n}),

    where :math:`\gamma(a, z)` is the lower incomplete gamma function.

    Parameters
    ----------
    z : float, optional
        Redshift used to set :math:`\rho_{\rm crit}(z)`. Default is 0.0.
    mass : float, optional
        Halo mass :math:`M` [:math:`M_\odot`], interpreted as the mass within
        :math:`r_{\rm vir}` given ``virial_overdensity``. Default is :math:`10^{12} M_\odot`.
    concentration : float, optional
        Concentration parameter :math:`c \equiv r_{\rm vir}/r_s`. Default is 10.
    virial_overdensity : float, optional
        Virial overdensity :math:`\Delta_{\rm vir}` relative to critical density.
        Default is 200.
    n : float, optional
        Einasto index :math:`n` that controls the shape of the profile. Higher
        values correspond to more centrally concentrated profiles. Typical values
        range from 0.5 to 10. Default is 1.0.
    r_vir : float, optional
        Virial radius :math:`r_{\rm vir}` [kpc]. If not provided, it is calculated
        from mass and virial overdensity.
    r_s : float, optional
        Scale radius :math:`r_s` [kpc]. If not provided, it is calculated from
        concentration and virial radius.
    scale_density : float, optional
        Density normalization :math:`\rho_s` [:math:`M_\odot\,\mathrm{kpc}^{-3}`].
        If not provided, it is calculated from the other parameters.
    adiabatic_contraction : bool, optional
        Placeholder flag for adiabatic-contraction adjustments (not applied in
        current implementation). Default is ``False``.

    Notes
    -----
    The Einasto profile provides a better fit to simulated dark matter halos
    than the NFW profile, especially in the outer regions. The logarithmic slope
    of the density profile varies smoothly with radius, unlike the NFW profile
    which has a fixed outer slope.

    The Einasto index :math:`n` is typically correlated with halo mass, with
    higher-mass halos having larger values of :math:`n`.

    References
    ----------
    Einasto, J. 1965, Trudy Astrofizicheskogo Instituta Alma-Ata, 5, 87

    Examples
    --------
    Create an Einasto halo with Einasto index :math:`n = 2`:

    >>> halo = EinastoHalo(mass=1e12, concentration=10, n=2.0, z=0)
    >>> r = 10.0  # kpc
    >>> rho = halo.density(r)      # ρ(r)
    >>> m_r = halo.menc(r)         # M(<r)
    """
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200., n=1.0,
                 r_vir=None, r_s=None, scale_density=None, adiabatic_contraction=False):
        self.n = n

        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function,
                         adiabatic_contraction=adiabatic_contraction)

    def density_function(self, x):
        r"""
        Dimensionless density profile :math:`f_\rho(x)` for the Einasto profile:

        .. math::

           f_\rho(x) = \exp\left(-x^{1/n}\right),

        where :math:`x = r/r_s` is the dimensionless radius and :math:`n` is
        the Einasto index.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless density :math:`f_\rho(x)`.
        """
        return np.exp(-x**(1/self.n))

    def _menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_M(<x)` for the Einasto profile:

        .. math::

           f_M(<x) = 3n\,\gamma(3n, x^{1/n})\,\Gamma(3n),

        where :math:`\gamma(a, z)` is the lower incomplete gamma function and
        :math:`\Gamma(a)` is the complete gamma function.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Returns
        -------
        ndarray
            Dimensionless enclosed mass :math:`f_M(<x)`.
        """
        return 3 * self.n * gammainc(3*self.n, x**(1/self.n)) * gamma(3*self.n)

class DekelZhaoHalo(DarkMatterHaloProfile):
    r"""
    Dekel-Zhao dark matter halo profile (Dekel et al. 2017; Freundlich et al. 2020).

    This class implements the Dekel-Zhao profile, a flexible double-power-law
    profile that can accurately describe dark matter halos with variable inner
    and outer slopes. 
    
    The density profile follows:

    .. math::

       \rho(x) = \rho_0 \cdot 
                 \frac{3-\alpha}{\alpha}\left[1 + \frac{3-g}{3-\alpha}x^{1/b}\right]
                 \left[x^\alpha(1+x^{1/b})^{1+b(g-\alpha)}\right]^{-1},

    where :math:`x = r/r_s` is the dimensionless radius, :math:`r_s` is the scale
    radius, :math:`\alpha` is the inner slope, :math:`g` is the outer slope, and
    :math:`b` controls the sharpness of the transition between inner and outer
    regions.

    The profile has an inner region with :math:`\rho \propto r^{-\alpha}` and an
    outer region with :math:`\rho \propto r^{-g}`, with a smooth transition
    controlled by :math:`b`.

    Parameters
    ----------
    z : float, optional
        Redshift used to set :math:`\rho_{\rm crit}(z)`. Default is 0.0.
    mass : float, optional
        Halo mass :math:`M` [:math:`M_\odot`], interpreted as the mass within
        :math:`r_{\rm vir}` given ``virial_overdensity``. Default is :math:`10^{12} M_\odot`.
    concentration : float, optional
        Concentration parameter :math:`c \equiv r_{\rm vir}/r_s`. Default is 10.
    virial_overdensity : float, optional
        Virial overdensity :math:`\Delta_{\rm vir}` relative to critical density.
        Default is 200.
    alpha : float, optional
        Inner slope parameter :math:`\alpha` of the density profile. The inner
        region has :math:`\rho \propto r^{-\alpha}`. Default is 1.0.
    gamma : float, optional
        Outer slope parameter :math:`gamma` of the density profile. The outer region
        has :math:`\rho \propto r^{-gamma}`. Default is 3.5.
    beta : float, optional
        Transition parameter :math:`beta` that controls the sharpness of the transition
        between inner and outer regions. Higher values correspond to sharper
        transitions. Default is 2.0.
    r_vir : float, optional
        Virial radius :math:`r_{\rm vir}` [kpc]. If not provided, it is calculated
        from mass and virial overdensity.
    r_s : float, optional
        Scale radius :math:`r_s` [kpc]. If not provided, it is calculated from
        concentration and virial radius.
    scale_density : float, optional
        Density normalization :math:`\rho_s` [:math:`M_\odot\,\mathrm{kpc}^{-3}`].
        If not provided, it is calculated from the other parameters.
    adiabatic_contraction : bool, optional
        Placeholder flag for adiabatic-contraction adjustments (not applied in
        current implementation). Default is ``False``.

    Notes
    -----
    The Dekel-Zhao profile is a generalization of several commonly used profiles:
    - When :math:`\alpha = 1`, :math:`gamma = 3`, and :math:`beta = 1`, it reduces to NFW
    - When :math:`\alpha = 0`, it produces a cored profile
    - When :math:`beta \to \infty`, it approaches a double power-law with a sharp break

    The profile provides excellent fits to dark matter halos in cosmological
    simulations and can accommodate a wide range of density profiles.

    References
    ----------
    Dekel, A., et al. 2017, MNRAS, 468, 1005
    Freundlich, J., et al. 2020, MNRAS, 499, 2912

    Examples
    --------
    Create a Dekel-Zhao halo with inner slope :math:`\alpha = 0.5` and outer slope :math:`gamma = 3.5`:

    >>> halo = DekelZhaoHalo(mass=1e12, concentration=10, alpha=0.5, gamma=3.5, beta=2.0, z=0)
    >>> r = 10.0  # kpc
    >>> rho = halo.density(r)      # ρ(r)
    >>> m_r = halo.menc(r)         # M(<r)
    """
    def __init__(self, 
            z=0.0, 
            mass=1e12, 
            concentration=10, 
            virial_overdensity=200., 
            alpha=1., 
            gamma=3.5, 
            beta=2.,
            r_vir=None, 
            r_s=None, 
            scale_density=None, 
            adiabatic_contraction=False
        ):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function,
                         adiabatic_contraction=adiabatic_contraction)

    def density_function(self, x):
        r"""
        Dimensionless density profile :math:`f_\rho(x)` for the Dekel-Zhao profile:

        .. math::

           f_\rho(x) = \frac{3-\alpha}{\alpha}\left[1 + \frac{3-gamma}{3-\alpha}x^{1/beta}\right]
                       \left[x^\alpha(1+x^{1/beta})^{1+beta(gamma-\alpha)}\right]^{-1},

        where :math:`x = r/r_s` is the dimensionless radius, :math:`\alpha` is
        the inner slope, :math:`gamma` is the outer slope, and :math:`beta` controls
        the transition sharpness.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`.

        Returns
        -------
        ndarray or float
            Dimensionless density :math:`f_\rho(x)`.
        """
        p1 = (3-self.alpha)/self.alpha * (1 + (3-self.gamma)/(3-self.alpha)*x**(1/self.beta))
        p2 = (x**self.alpha * (1 + x**(1/self.beta))**(1+self.beta*(self.gamma-self.alpha)) )**-1
        return p1*p2

    def _menc_dimless(self, x):
        r"""
        Dimensionless enclosed mass :math:`f_M(<x)` for the Dekel-Zhao profile:

        .. math::

           f_M(<x) = x^3 \left[x^\alpha(1+x^{1/beta})^{1+beta(gamma-\alpha)}\right]^{-1}.

        Parameters
        ----------
        x : array_like or float
            Dimensionless radius :math:`x = r/r_s`. ``np.inf`` is allowed.

        Returns
        -------
        ndarray
            Dimensionless enclosed mass :math:`f_M(<x)`.
        """
        return x**3 * (x**self.alpha * (1 + x**(1/self.beta))**(1+self.beta*(self.gamma-self.alpha)) )**-1

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    now = time_ns()
    m = 3e10
    halo_nfw = NFWHalo(z=0, mass=m, concentration=10, virial_overdensity=200)
    halo_genalpha = alhpaNFWHalo(z=0, mass=m, concentration=10, virial_overdensity=200, alpha=0.)
    burkert = BurkertHalo(z=0, mass=m, concentration=10, virial_overdensity=200)
    dekel = DekelZhaoHalo(z=0, mass=m, concentration=10, virial_overdensity=200, alpha=1.0)
    print(f"{(time_ns() - now) * 1e-9:0.4f} sec")

    now = time_ns()
    r = np.linspace(0, halo_nfw.r_vir, num=100)
    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    axes[0].loglog(r, halo_nfw.density(r), label="nfw")
    axes[0].loglog(r, halo_genalpha.density(r), label="gen", ls='--')
    axes[0].loglog(r, burkert.density(r), label="gen", ls=':')
    axes[0].loglog(r, dekel.density(r), label="gen", ls='-.')
    axes[1].loglog(r, halo_nfw.menc(r), label="nfw")
    axes[1].loglog(r, halo_genalpha.menc(r), label="gen", ls='--')
    axes[1].loglog(r, burkert.menc(r), label="gen", ls=':')
    axes[1].loglog(r, dekel.menc(r), label="gen", ls='-.')
    axes[2].loglog(r, halo_nfw.vcirc(r), label="nfw")
    axes[2].loglog(r, halo_genalpha.vcirc(r), label="gen", ls='--')
    axes[2].loglog(r, burkert.vcirc(r), label="gen", ls=':')
    axes[2].loglog(r, dekel.vcirc(r), label="gen", ls='-.')
    print(f"{(time_ns() - now) * 1e-9:0.4f} sec")
    fig.legend()
    plt.show()

