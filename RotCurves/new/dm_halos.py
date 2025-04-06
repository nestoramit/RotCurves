import numpy as np
from time import time_ns
from base_classes import DarkMatterHaloProfile

class NFW_Halo(DarkMatterHaloProfile):
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

    def _vcirc2_dimless(self, x):
        return 3 * (np.log(1 + x) - x / (1 + x)) / x



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    now = time_ns()
    halo_nfw = NFW_Halo(z=0, mass=1e12, concentration=10, virial_overdensity=200)
    halo_gen = DarkMatterHaloProfile(z=0, mass=1e12, concentration=10, virial_overdensity=200)
    print(f"{(time_ns() - now) * 1e-9:0.4f} sec")

    now = time_ns()
    r = np.linspace(0, halo_nfw.r_vir, num=100)
    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    axes[0].loglog(r, halo_nfw.density(r), label="nfw")
    axes[0].loglog(r, halo_gen.density(r), label="gen")
    axes[1].semilogy(r, halo_nfw.menc(r), label="nfw")
    axes[1].semilogy(r, halo_gen.menc(r), label="gen")
    axes[2].plot(r, halo_nfw.vcirc(r), label="nfw")
    axes[2].plot(r, halo_gen.vcirc(r), label="gen")
    print(f"{(time_ns() - now) * 1e-9:0.4f} sec")
    fig.legend()
    plt.show()

