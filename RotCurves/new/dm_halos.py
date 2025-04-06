import numpy as np
from time import time_ns

from astropy.constants.codata2018 import alpha

from base_classes import DarkMatterHaloProfile
from scipy.special import hyp2f1, gamma, gammainc

class NFWHalo(DarkMatterHaloProfile):
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200.,
                 r_vir=None, r_s=None, scale_density=None):
        """
        Initialize an NFW halo profile.

        Parameters:
        - M: Mass of the halo in solar masses.
        - c: Concentration parameter.
        - z: Redshift (default is 0.0).
        """

        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function)

    def density_function(self, x):
        return x**-1 * (1 + x)**-2

    def _menc_dimless(self, x):
        return 3 * (np.log(1 + x) - x / (1 + x))


class alhpaNFWHalo(DarkMatterHaloProfile):
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200., alpha=1.0,
                 r_vir=None, r_s=None, scale_density=None):
        """
        Initialize an NFW halo profile.

        Parameters:
        - M: Mass of the halo in solar masses.
        - c: Concentration parameter.
        - z: Redshift (default is 0.0).
        - alpha: inner slope for the density profile (default is 1.0).
        """

        self.alpha = alpha
        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function)

    def density_function(self, x):
        return x**-self.alpha * (1 + x)**-(3-self.alpha)

    def _menc_dimless(self, x):
        return 3/(3-self.alpha) * (x**(3-self.alpha) * hyp2f1(3-self.alpha, 3-self.alpha, 4-self.alpha, -x))

class BurkertHalo(DarkMatterHaloProfile):
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200., alpha=1.0,
                 r_vir=None, r_s=None, scale_density=None):
        """
        Initialize a Burkert halo profile (Burkert+1995).

        Parameters:
        - M: Mass of the halo in solar masses.
        - c: Concentration parameter.
        - z: Redshift (default is 0.0).
        """

        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function)

    def density_function(self, x):
        return ((1+x)*(1 + x**2))**-1

    def _menc_dimless(self, x):
        return 3/2 * (1/2*np.log(1+x**2) + np.log(1+x) - np.arctan(x))

class EinastoHalo(DarkMatterHaloProfile):
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200., n=1.0,
                 r_vir=None, r_s=None, scale_density=None):
        """
        Initialize a Burkert halo profile (Burkert+1995).

        Parameters:
        - M: Mass of the halo in solar masses.
        - c: Concentration parameter.
        - z: Redshift (default is 0.0).
        - n: Einasto parameter (default is 1.0).
        """

        self.n = n

        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function)

    def density_function(self, x):
        return np.exp(-x**(1/self.n))

    def _menc_dimless(self, x):
        return 3 * self.n * gammainc(3*self.n, x**(1/self.n)) * gamma(3*self.n)

class DekelZhaoHalo(DarkMatterHaloProfile):
    def __init__(self, z=0.0, mass=1e12, concentration=10, virial_overdensity=200., alpha=1., g=3.5, b=2.,
                 r_vir=None, r_s=None, scale_density=None):
        """
        Initialize a Dekel-Zhao halo profile (Freundlich+2020).

        Parameters:
        - M: Mass of the halo in solar masses.
        - c: Concentration parameter.
        - z: Redshift (default is 0.0).
        - alpha: inner slope for the density profile (default is 1.0).
        - g: outer slope for the density profile (default is 3.5).
        - b: transition parameter (default is 2.0).
        """

        self.alpha = alpha
        self.g = g
        self.b = b

        super().__init__(mass=mass, concentration=concentration, z=z, virial_overdensity=virial_overdensity,
                         r_vir=r_vir, r_s=r_s, scale_density=scale_density,
                         density_function=self.density_function)

    def density_function(self, x):
        p1 = (3-self.alpha)/self.alpha * (1 + (3-self.g)/(3-self.alpha)*x**(1/self.b))
        p2 = (x**self.alpha * (1 + x**(1/self.b))**(1+self.b*(self.g-self.alpha)) )**-1
        return p1*p2

    def _menc_dimless(self, x):
        return x**3 * (x**self.alpha * (1 + x**(1/self.b))**(1+self.b*(self.g-self.alpha)) )**-1

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

