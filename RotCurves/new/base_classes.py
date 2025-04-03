from time import time_ns

import numpy as np
import astropy.constants as c
from scipy.special import j0, j1, k0
from scipy.integrate import quad
import warnings
from base_utils import integrate_quad_list, solve_numerical_using_brentq


# Define constants as global variables
G_CONST = c.G.to('kpc km2 / (s2 Msun)').value

class SurfaceDensityProfile:
    def __init__(self, mass, effective_radius=None, scale_radius=None, surface_density_function=None,
                 q0=0., mass_to_light=1.):
        self.mass = mass
        self.effective_radius = effective_radius
        self.scale_radius = scale_radius
        self.q0 = q0
        self.mass_to_light = mass_to_light

        # Set the default surface density function to an exponential "Freeman Disk" if not provided (e.g., Freeman+1970)
        if surface_density_function is None:
            self.surface_density_function = self._default_surface_density_function
        else:
            self.surface_density_function = surface_density_function

        # Handle cases where both radii are given
        if self.effective_radius is not None and self.scale_radius is not None:
            warnings.warn("Both effective_radius and scale_radius are given. Using scale_radius.", UserWarning)
        elif self.scale_radius is None and self.effective_radius is not None:
            self.scale_radius = self._calculate_scale_radius_from_effective()
        elif self.effective_radius is None and self.scale_radius is not None:
            self.effective_radius = self._calculate_effective_radius_from_scale()
        else:
            raise ValueError("Either scale_radius or effective_radius must be provided.")

        self.scale_density = self.scale_density()
        self.scale_mass = self.scale_mass()
        self.scale_velocity = self.scale_velocity()

    def _default_surface_density_function(self, r):
        # Default exponential Freeman Disk surface density profile
        x = self._calculate_normalized_radius(r)
        return np.exp(-x)

    def _calculate_normalized_radius(self, r):
        # Normalized radius x = r / scale_radius
        return np.abs(r) / self.scale_radius

    def _calculate_scale_radius_from_effective(self):
        # Placeholder formula to convert effective radius to scale radius
        func = lambda r_s: self.menc_dimless(self.effective_radius / r_s) - 0.5 * self.menc_dimless(np.inf)
        return solve_numerical_using_brentq(func, p0=self.effective_radius)

    def _calculate_effective_radius_from_scale(self):
        # Placeholder formula to convert scale radius to effective radius
        func = lambda r_eff: self.menc_dimless(r_eff / self.scale_radius) - 0.5 * self.menc_dimless(np.inf)
        # func = lambda r_eff: self.menc_dimless(r_eff / self.scale_radius) - 0.5 * self.menc_dimless(3*self.scale_radius)
        return solve_numerical_using_brentq(func, p0=self.scale_radius)

    def scale_density(self):
        # Placeholder for scale density calculation
        return self.mass / (2 * np.pi * self.scale_radius ** 2 * self.menc_dimless(np.inf))[0]

    def scale_mass(self):
        # Placeholder for scale mass calculation
        return 2 * np.pi * self.scale_density * self.scale_radius**2

    def scale_velocity(self):
        # Use the global G_CONST
        return np.sqrt(G_CONST * self.scale_mass / self.scale_radius)

    def surface_density_dimless(self, x):
        return self.surface_density_function(x)

    def surface_density(self, r):
        x = self._calculate_normalized_radius(r)
        return self.surface_density_dimless(x) * self.scale_density

    def menc_dimless(self, x):
        # Integration using scipy.quad
        mass = integrate_quad_list(lambda t: t * self.surface_density_dimless(t), 0, x)
        return mass

    def menc(self, r):
        x = self._calculate_normalized_radius(r)
        return self.menc_dimless(x) * self.scale_mass

    # def circular_velocity_dimless(self, x):
    #     """
    #     Calculate the circular velocity for an array of dimensionless radii `x`.
    #
    #     Parameters:
    #     - x : np.array
    #         The dimensionless radius values for which to calculate the circular velocity.
    #
    #     Returns:
    #     - np.array : The corresponding circular velocities.
    #     """
    #
    #     # Define the S_func as an inline function to calculate the integral
    #     def S_func(k):
    #         return -quad(lambda u: j0(k * u) * u * self.surface_density_dimless(u), 0, np.inf)[0]
    #
    #     # Define the main function for circular velocity calculation
    #     def func(t):
    #         return -t * quad(lambda k: j1(k * t * self.scale_radius) * k * S_func(k), 0, np.inf)[0]
    #
    #     # Apply the function element-wise to the input array `x`
    #     return np.sqrt(np.asarray([func(xi) for xi in x]))

    def vcirc2_dimless(self, x):
        # Velocity calculation using enclosed mass
        return np.divide(self.menc_dimless(x), x, out=np.zeros_like(x), where=x!=0)

    def vcirc2(self, r):
        x = self._calculate_normalized_radius(r)
        return self.vcirc2_dimless(x) * self.scale_velocity**2

    def vcirc_dimless(self, x):
        """
        Calculate the circular velocity for an array of dimensionless radii `x`.
        Negative vcirc2 values are set to zero to avoid NaNs.
        """

        return np.sqrt(np.clip(self.vcirc2_dimless(x), 0, None))


    def vcirc(self, r):
        x = self._calculate_normalized_radius(r)
        return self.vcirc_dimless(x) * self.scale_velocity

    def light_profile(self, r):
        x = self._calculate_normalized_radius(r)
        return self.mass_to_light * self.surface_density_dimless(x)

    def dlnrho_dlnr(self, r):
        """
        Calculate the logarithmic density slope at radius r.
        Used in calculations of the pressure support (e.g., Burkert+2010)
        """
        x = self._calculate_normalized_radius(r)
        dlnrho_dlnr = np.gradient(np.log(self.surface_density_dimless(x)), np.log(x))
        return dlnrho_dlnr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    func_exp = lambda x: np.exp(-x)

    disk = SurfaceDensityProfile(mass=1e11, effective_radius=5, surface_density_function=func_exp)

    now = time_ns()
    r = np.linspace(0, 30, num=100)
    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    axes[0].plot(r, disk.surface_density(r), label="Surface Density")
    axes[1].plot(r, disk.menc(r), label="Enclosed Mass")
    axes[2].plot(r, disk.vcirc2(r), label="Circular Velocity")
    print(f"{(time_ns() - now)*1e-9:0.4f} sec")
    fig.legend()
    plt.show()