import os
import numpy as np
import logging
from scipy.special import gamma, gammainc, gammaincinv
from scipy.interpolate import CubicSpline
import astropy.constants as c

from RotCurves.utils import load_noordermeer_tables
from base_classes import SurfaceDensityProfile

# Define constants as global variables
G_CONST = c.G.to('kpc km2 / (s2 Msun)').value

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
# TODO: change directory when taking out of "new" folder
noor_lookuptables_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/lookup_tables/Noordermeer_lookup_tables"
NoordermeerLookupTables = load_noor_lookuptable(noor_lookuptables_dir)


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
        self.integral_results = NoordermeerLookupTables[self.n][self.closest_q_table()]

    def surface_density_function(self, x):
        """
        The surface density follows a Sersic law (Sérsic+1968):
        Σ(r) = Σ0 * exp[-x^(1/n)]
        for x = r / scale_radius.
        """

        return np.exp(-x ** (1 / self.n))

    def sersic_b(self):
        """
        Calculate the Sersic b parameter using the inverse gamma function.
        """
        return gammaincinv(2 * self.n, 0.5)

    def calculate_effective_radius_from_scale(self):
        return self.sersic_b()**self.n * self.scale_radius

    def calculate_scale_radius_from_effective(self):
        return self.sersic_b()**-self.n * self.effective_radius

    def scale_density(self):
        return self.mass / (2 * np.pi * self.scale_radius**2 * gamma(2 * self.n))

    def scale_mass(self):
        return self.mass / gamma(2 * self.n)

    def menc_dimless(self, x):
        """
        The enclosed mass is computed using the Sersic profile equation:
        f_M(<x) = gamma(2n, x^(1/n))
        """
        return gammainc(2 * self.n, x ** (1 / self.n)) * gamma(2 * self.n)

    def closest_q_table(self):
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.special import i0, i1, k0, k1

    M = 1e11
    Reff = 5.
    n = 1.0
    q0 = 0

    r = np.linspace(0, 10, num=100)
    x = r / (Reff/1.68)
    sersic = SersicProfile(mass=M, effective_radius=Reff, n=n, q0=q0)

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    axes[0].plot(r, sersic.surface_density(r), label="Surface Density")
    axes[1].plot(r, sersic.menc(r), label="Enclosed Mass")
    axes[2].plot(r, sersic.vcirc(r), label="Circular Velocity")

    Vfreeman = 0.5 * x**2 * (i0(x/2) * k0(x/2) - i1(x/2) * k1(x/2)) * sersic.scale_velocity**2
    axes[2].plot(r, np.sqrt(Vfreeman), label="Freeman disk")
    plt.legend()

    plt.figure()
    plt.plot(r, sersic.vcirc2(r) / Vfreeman)
    plt.show()