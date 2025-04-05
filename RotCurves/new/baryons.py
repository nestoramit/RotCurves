import os
import numpy as np
import logging
from scipy.special import gamma, gammainc, gammaincinv, i0, k0, i1, k1
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import astropy.constants as c
import astropy.units as u
import matplotlib.pyplot as plt

from RotCurves.new.base_utils import integrate_quad_list
from RotCurves.utils import load_gaussian_tables, M_solar
from base_classes import SurfaceDensityProfile

# Define constants as global variables
G_CONST = c.G.to('kpc km2 / (s2 Msun)').value

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')


def load_noor_lookuptable(dir_path):
    tables = {}
    q_list = np.asarray(list(set([float(x[x.find('_q') + 2:x.find('.npy')]) for x in os.listdir(dir_path) if x.find('npy') > 0])))
    n_list = np.asarray(list(set([float(x[x.find('_n') + 2:x.find('_q')]) for x in os.listdir(dir_path) if x.find('npy') > 0])))
    tables['q_list'] = q_list
    tables['n_list'] = n_list

    for n in n_list:
        tables[n] = {}
        for q in q_list:
            try:
                tables[n][q] = np.load(os.path.join(dir_path, "noor_n%2.2f_q%2.2f.npy" % (n, q)))
            except:
                pass
    return tables

def load_gaussian_tables(dir_path):
    tables = {}
    # TODO: switch from invh to h
    invh_list = np.asarray(list(
        set([float(x[x.find('_invh') + 6:x.find('.csv')]) for x in os.listdir(dir_path) if
             x.find('csv') > 0])))
    tables['h_list'] = invh_list

    for invh in invh_list:
        tables[invh] = np.load(os.path.join(dir_path, "Gauss_invh_%2.2f.npy" % invh))
    return tables


# TODO: change directory when taking out of "new" folder
noor_lookuptables_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lookup_tables/Noordermeer_lookup_tables"
GaussianRing_lookuptables_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lookup_tables/GaussianRing_lookup_tables"
NoordermeerLookupTables = load_noor_lookuptable(noor_lookuptables_dir)
GaussianRingLookupTables = load_gaussian_tables(GaussianRing_lookuptables_dir)


class SersicProfile(SurfaceDensityProfile):
    def __init__(self, mass, effective_radius=None, scale_radius=None, n=1, q0=0.):
        """
        Initializes a SersicProfile instance, inheriting from SurfaceDensityProfile and modifying
        the enclosed mass calculation.

        :param mass: Total mass of the galaxy.
        :param effective_radius: Effective radius (optional).
        :param scale_radius: Scale radius (optional).
        :param surface_density_function: Custom surface density function (optional).
        :param n: The Sersic index, determining the concentration of the profile.
        :param q0: intrinsic axis ratio (optional, default is 0).
        """
        # Set the Sersic index n
        self.n = n

        # Call the parent class constructor
        super().__init__(mass=mass,
                         effective_radius=effective_radius,
                         scale_radius=scale_radius,
                         surface_density_function=self.surface_density_function,
                         q0=q0)


        # load lookuptables
        self.integral_results = NoordermeerLookupTables[self.n][self._closest_q_table()]

    def surface_density_function(self, x):
        """
        The surface density follows a Sersic law (Sérsic+1968):
        Σ(r) = Σ0 * exp[-x^(1/n)]
        for x = r / r_s.
        """

        return np.exp(-x ** (1 / self.n))

    def sersic_b(self):
        """
        Calculate the Sersic b parameter using the inverse gamma function.
        """
        return gammaincinv(2 * self.n, 0.5)

    def _calculate_effective_radius_from_scale(self):
        return self.sersic_b()**self.n * self.scale_radius

    def _calculate_scale_radius_from_effective(self):
        return self.sersic_b()**-self.n * self.effective_radius

    def scale_density(self):
        corr = gamma(2 * self.n)
        return self.mass / (2 * np.pi * self.scale_radius**2) / corr

    def scale_mass(self):
        return self.mass / gamma(2 * self.n)

    def menc_dimless(self, x):
        """
        The enclosed mass is computed using the Sersic profile equation:
        f_M(<x) = gamma(2n, x^(1/n))
        """
        return gammainc(2 * self.n, x ** (1 / self.n)) * gamma(2 * self.n)

    def _closest_q_table(self):
        q0_list = NoordermeerLookupTables['q_list']
        if self.q0 in q0_list:
            return self.q0
        else:
            q0_closest = q0_list[np.argmin(np.abs(q0_list - self.q0))]
            logger.warning('Sersic disk q: non-exact value, using %2.3f instead of %2.3f' % (q0_closest, self.q0))
            return q0_closest

    def vcirc2_dimless(self, x):
        """
        The circular velocity is computed following Noordermeer+2008 for a thickened disk
        Taken from the lookuptables under "/lookup_tables/Noordermeer_lookup_tables"
        """

        interpolator = CubicSpline(x=self.integral_results[:, 0], y=self.integral_results[:, 1])

        # TODO: the talbes are in x=r/reff, change to x=r/rs
        v2 = interpolator(x*self.scale_radius/self.effective_radius)

        # TODO: the constant C should be included in the tables and removed from here
        C = 2 * self.sersic_b()**(self.n+1) / (np.pi * self.n**2)
        return C * v2

    def dlnrho_dlnr(self, r):
        """
        Calculate the logarithmic slope of the density profile at radius r.
        Used in calculations of the pressure support (e.g., Burkert+2010)
        """
        x = self._calculate_normalized_radius(r)
        return - 1/self.n * x**(1/self.n)


class FreemanDisk(SurfaceDensityProfile):
    def __init__(self, mass, effective_radius=None, scale_radius=None):
        """
        Initializes a FreemanDisk instance, inheriting from SurfaceDensityProfile and modifying
        the enclosed mass calculation.

        :param mass: Total mass of the galaxy.
        :param effective_radius: Effective radius (optional).
        :param scale_radius: Scale radius (optional).
        :param surface_density_function: Custom surface density function (optional).
        """

        # Call the parent class constructor
        super().__init__(mass=mass,
                         effective_radius=effective_radius,
                         scale_radius=scale_radius,
                         surface_density_function=self.surface_density_function)

    def surface_density_function(self, x):
        """
        The surface density follows an Exponential law (Freeman+1970):
        Σ(r) = Σ0 * exp[-x]
        for x = r / r_s.
        """

        return np.exp(-x)


    def _calculate_effective_radius_from_scale(self):
        return self.scale_radius * 1.678

    def _calculate_scale_radius_from_effective(self):
        return self.effective_radius / 1.678

    def scale_density(self):
        return self.mass / (2 * np.pi * self.scale_radius**2)

    def scale_mass(self):
        return self.mass

    def menc_dimless(self, x):
        """
            The enclosed mass is computed using the Freeman disk equation:
            f_M(<x) = 1 - exp(-x) * (1 + x)
        """
        return 1 - np.exp(-x) * (1 + x)

    def vcirc2_dimless(self, x):
        """
        The circular velocity is computed following Noordermeer+2008 for a thickened disk
        Taken from the lookuptables under "/lookup_tables/Noordermeer_lookup_tables"
        """

        y = x/2
        res = 2 * y**2 * (i0(y) * k0(y) - i1(y) * k1(y))
        return np.nan_to_num(res, nan=0)

    def dlnrho_dlnr(self, r):
        """
        Calculate the logarithmic slope of the density profile at radius r.
        Used in calculations of the pressure support (e.g., Burkert+2010)
        """
        x = self._calculate_normalized_radius(r)
        return - x


class GaussianRingProfile(SurfaceDensityProfile):
    def __init__(self, mass, scale_radius=None, h=None, FWHM_ring=None, sigma_ring=None, mass_to_light=1.,
                 lookup=True):
        """
        Initializes a GaussianRing instance, inheriting from SurfaceDensityProfile and modifying
        the enclosed mass calculation.

        :param mass: Total mass of the ring.
        :param h: Shape parameter, h = r_s / FWHM_ring (optional).
        :param scale_radius: Scale (peak) radius (optional).
        :param FWHM_ring: Full Width at Half Maximum of the Gaussian ring.
        :param sigma_ring: Standard deviation of the Gaussian ring.
        :param mass_to_light: Mass-to-light ratio (optional).
        """

        # Set the Gaussian ring parameters
        # h is defined as: h = r_s / FWHM_ring
        # check to see that given any two of the four parameters {r_s, h, FWHM_ring, sigma_ring},
        # the other two can be calculated
        if scale_radius is None and h is None:
            raise ValueError("Either r_s or h must be provided.")

        class _ParameterSolver:
            def __init__(self):
                # Map of combinations to calculation functions
                self.method_map = {
                    ('FWHM_ring', 'h'): self.calculate_from_fwhm_h,
                    ('FWHM_ring', 'r_s'): self.calculate_from_fwhm_scale,
                    ('h', 'r_s'): self.calculate_from_h_scale,
                    ('h', 'sigma_ring'): self.calculate_from_h_sigma,
                    ('r_s', 'sigma_ring'): self.calculate_from_scale_sigma,
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

            def calculate_from_fwhm_scale(self, fwhm, scale):
                h = scale / fwhm
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                return h, sigma

            def calculate_from_h_scale(self, h, scale):
                fwhm = scale / h
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                return fwhm, sigma

            def calculate_from_h_sigma(self, h, sigma):
                fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
                scale = fwhm * h
                return fwhm, scale

            def calculate_from_scale_sigma(self, scale, sigma):
                fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
                h = scale / fwhm
                return h, fwhm
        parameters = _ParameterSolver().calculate(h=h, FWHM_ring=FWHM_ring, scale_radius=scale_radius, sigma_ring=sigma_ring)
        self.h = parameters['h']
        self.scale_radius = parameters['r_s']
        self.sigma_ring = parameters['sigma_ring']
        self.FWHM_ring = parameters['FWHM_ring']
        self.A = self.scale_radius**2 / (2 * self.sigma_ring**2)

        # Call the parent class constructor
        super().__init__(mass=mass,
                         scale_radius=scale_radius,
                         surface_density_function=self.surface_density_function,
                         mass_to_light=mass_to_light)


        # load lookuptables
        # TODO: update path
        self.lookup = lookup
        self.integral_results = GaussianRingLookupTables[self._closest_h_table()]

    def surface_density_function(self, x):
        """
        The surface density follows a shifted Gaussian:
        Σ(r) = Σ0 * exp[-A*(x-1)^2]
        for x = r / r_s.
        """

        return np.exp(-self.A * (x-1)**2)

    def scale_density(self):
        A = self.A
        corr = 1 / (2 * A) * (np.exp(-A) + np.sqrt(np.pi * A) * (1 + gammainc(0.5, A)))
        return self.mass / (2 * np.pi * self.scale_radius**2) / corr

    def scale_mass(self):
        A = self.A
        corr = 1 / (2 * A) * (np.exp(-A) + np.sqrt(np.pi * A) * (1 + gammainc(0.5, A)))
        return self.mass / corr

    # def menc_dimless_paper(self, x):
    #     A = self.A
    #     Ax = A * (x-1)**2
    #
    #     p1 = np.exp(-A/2)/A * np.exp(-1/2*Ax)
    #     p2 = np.sinh(1/2 * (Ax - A))
    #     p3 = 1 / (2*np.sqrt(A)) * (gammainc(0.5, A) + gammainc(0.5, Ax*np.sign(x-1))) * gamma(0.5)
    #     return p1*p2 + p3

    def menc_dimless(self, x):
        A = self.A
        Ax = A * (x - 1) ** 2

        p1 = 1 / (2 * A) * (np.exp(-A) - np.exp(-Ax))
        p2 = 1 / (2 * np.sqrt(A)) * (gammainc(0.5, A) + gammainc(0.5, Ax) * np.sign(x - 1)) * gamma(0.5)
        return p1 + p2

    def _closest_h_table(self):
        h_list = GaussianRingLookupTables['h_list']

        if self.h in h_list:
            closest_h = self.h
        else:
            closest_h = h_list[np.argmin(np.abs(h_list - self.h))]
            logger.warning(f'Gaussian Ring h: non-exact value, using {closest_h:2.3f} instead of {self.h:%2.3f}')

        return closest_h

    def vcirc2_dimless(self, x):
        """
        Taken from the lookuptables under "/lookup_tables/GaussianRing_lookup_tables"
        """

        if self.lookup:
            # Use the lookup table for the Gaussian ring
            # TODO: something is wrong with the lookup table, it is not working. calculate it again.
            interpolator = CubicSpline(x=self.integral_results[:, 0], y=self.integral_results[:, 1])
            v2 = interpolator(x)

        else:
            Iprime = lambda t: quad(lambda s: t * (s**2 - t**2)**-0.5 * (1-s) * self.surface_density_dimless(s), t, np.inf)[0]
            func = lambda x: quad(lambda t: -Iprime(t) * t * (x**2 - t**2)**-0.5, 0, x)[0]
            v2 = [func(xi) for xi in x]
            v2 = np.asarray(v2)

        # TODO: the constant C should be included in the tables and removed from here
        # C = 1.
        C = 4 * self.A / np.pi
        return C * v2

    def dlnrho_dlnr(self, r):
        x = self._calculate_normalized_radius(r)
        return - 2 * self.A * x * (x - 1)


# Test if a FreemanDisk and SersicProfile with n=1 are the same
def test_freeman_sersic():
    M = 1e11
    Reff = 5.
    n = 1.0
    q0 = 0

    r = np.linspace(0, 20, num=100)
    sersic = SersicProfile(mass=M, effective_radius=Reff, n=n, q0=q0)
    freeman = FreemanDisk(mass=M, effective_radius=Reff)

    rtol = 1e-3
    atol = 0.

    assert np.allclose(sersic.surface_density(r), freeman.surface_density(r), rtol=rtol, atol=atol, equal_nan=True)
    assert np.allclose(sersic.menc(r), freeman.menc(r), rtol=rtol, atol=atol, equal_nan=True)
    assert np.allclose(sersic.vcirc(r), freeman.vcirc(r), rtol=rtol, atol=atol, equal_nan=True)

    print("Freeman and Sersic n=1 profiles are equal within the tolerance limits.")


if __name__ == '__main__':
    from rotationcurves.models.models import GaussianRingObject

    M = 1e10
    Rpeak = 5.
    h = 1.
    kpc = c.kpc.to(u.m).value
    M_solar = c.M_sun.to(u.kg).value  # kg
    G = c.G.to(u.m ** 3 / u.kg / u.s ** 2).value  # m^3 kg^-1 s^-2

    r = np.linspace(0, 10*Rpeak, num=100)

    freeman = FreemanDisk(mass=M, scale_radius=Rpeak)
    ring_new = GaussianRingProfile(mass=M, scale_radius=Rpeak, h=h, lookup=False)
    ring_old = GaussianRingObject(mass=M, rpeak=Rpeak, ring_FWHM=Rpeak/h, use_lookuptable=True)
    ring_old.get_profiles(r)
    ring_old.get_RC(r)

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    axes[0].plot(r, ring_new.surface_density(r), label="Surface Density")
    axes[0].plot(r, ring_old.density_profile, ls='--', label="Surface Density old")
    axes[1].plot(r, ring_new.menc(r), label="Enclosed Mass")
    axes[1].plot(r, ring_old.mass_profile*ring_old.mass/ring_old.M0, ls='--', label="Enclosed Mass old")
    axes[2].plot(r, ring_new.vcirc(r), label="Circular Velocity")
    axes[2].plot(r, ring_old.V/np.sqrt(G)*1e-3, ls='--', label="Circular Velocity old")
    axes[2].plot(r, freeman.vcirc(r), ls='-.', label="Freeman Disk")
    plt.legend()
    plt.show()
