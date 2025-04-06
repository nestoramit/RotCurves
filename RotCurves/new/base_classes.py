from time import time_ns

import numpy as np
import astropy.constants as c
from astropy.cosmology import Planck18
from scipy.special import j0, j1, k0
from scipy.integrate import quad
import warnings
from base_utils import integrate_quad_list, solve_numerical_using_brentq
import logging

# Define constants as global variables
G_CONST = c.G.to('kpc km2 / (s2 Msun)').value

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')

class SurfaceDensityProfile:
    def __init__(self, mass, r_eff=None, r_s=None, surface_density_function=None,
                 q0=0., mass_to_light=1.):
        self.mass = mass
        self.r_eff = r_eff
        self.r_s = r_s
        self.q0 = q0
        self.mass_to_light = mass_to_light
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
        # Default exponential Freeman Disk surface density profile
        x = self._calculate_normalized_radius(r)
        return np.exp(-x)

    def _calculate_normalized_radius(self, r):
        # Normalized radius x = r / r_s
        return np.abs(r) / self.r_s

    def _calculate_scale_radius_from_effective(self):
        # Placeholder formula to convert effective radius to scale radius
        func = lambda r_s: self.menc_dimless(self.r_eff / r_s) - 0.5 * self.menc_dimless(np.inf)
        return solve_numerical_using_brentq(func, p0=self.r_eff)

    def _calculate_effective_radius_from_scale(self):
        # Placeholder formula to convert scale radius to effective radius
        func = lambda r_eff: self.menc_dimless(r_eff / self.r_s) - 0.5 * self.menc_dimless(np.inf)
        # func = lambda r_eff: self.menc_dimless(r_eff / self.r_s) - 0.5 * self.menc_dimless(3*self.r_s)
        return solve_numerical_using_brentq(func, p0=self.r_s)

    def _scale_density(self):
        # Placeholder for scale density calculation
        return self.mass / (2 * np.pi * self.r_s ** 2 * self.menc_dimless(np.inf))[0]

    def _scale_intensity(self):
        # Placeholder for scale intensity calculation
        if self.mass == 0.:
            return 1.
        else:
            return self.scale_density * self.mass_to_light

    def _scale_mass(self):
        # Placeholder for scale mass calculation
        return 2 * np.pi * self.scale_density * self.r_s**2

    def _scale_velocity(self):
        # Use the global G_CONST
        return np.sqrt(G_CONST * self.scale_mass / self.r_s)

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
    #         return -t * quad(lambda k: j1(k * t * self.r_s) * k * S_func(k), 0, np.inf)[0]
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

        return np.sqrt(np.maximum(self.vcirc2_dimless(x), 0))

    def vcirc(self, r):
        x = self._calculate_normalized_radius(r)
        return self.vcirc_dimless(x) * self.scale_velocity

    def light_profile(self, r):
        scale_intensity = self._scale_intensity()
        x = self._calculate_normalized_radius(r)
        return scale_intensity * self.surface_density_dimless(x)

    def dlnrho_dlnr(self, r):
        """
        Calculate the logarithmic density slope at radius r.
        Used in calculations of the pressure support (e.g., Burkert+2010)
        """
        x = self._calculate_normalized_radius(r)
        dlnrho_dlnr = np.gradient(np.log(self.surface_density_dimless(x)), np.log(x))
        return dlnrho_dlnr


class DarkMatterHaloProfile:
    def __init__(self, mass, z=None, r_vir=None, r_s=None, concentration=None, scale_density=None, virial_overdensity=200, density_function=None):
        self.z = z
        self.mass = mass
        self.r_vir = r_vir
        self.r_s = r_s
        self.c = concentration
        self.scale_density = scale_density
        self.virial_overdensity = virial_overdensity
        self.density_function = density_function

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
        # Default NFW density profile
        return x**-1 * (1 + x)**-2

    def _calculate_normalized_radius(self, r):
        # Normalized radius x = r / r_s
        return np.abs(r) / self.r_s

    def _scale_mass(self):
        # Placeholder for scale mass calculation
        return 4/3 * np.pi * self.scale_density * self.r_s ** 3

    def _scale_velocity(self):
        # Use the global G_CONST
        return np.sqrt(G_CONST * self.scale_mass / self.r_s)

    def _density_dimless(self, x):
        return self.density_function(x)

    def density(self, r):
        x = self._calculate_normalized_radius(r)
        return self._density_dimless(x) * self.scale_density

    def _menc_dimless(self, x):
        # Integration using scipy.quad
        mass = integrate_quad_list(lambda t: 3 * t ** 2 * self._density_dimless(t), 0, x)
        return mass

    def menc(self, r):
        x = self._calculate_normalized_radius(r)
        return self._menc_dimless(x) * self.scale_mass

    def _vcirc2_dimless(self, x):
        # Velocity calculation using enclosed mass
        return np.divide(self._menc_dimless(x), x, out=np.zeros_like(x), where=x != 0)

    def vcirc2(self, r):
        x = self._calculate_normalized_radius(r)
        return self._vcirc2_dimless(x) * self.scale_velocity ** 2

    def _vcirc_dimless(self, x):
        """
        Calculate the circular velocity for an array of dimensionless radii `x`.
        Negative vcirc2 values are set to zero to avoid NaNs.
        """

        return np.sqrt(np.maximum(self._vcirc2_dimless(x), 0))

    def vcirc(self, r):
        x = self._calculate_normalized_radius(r)
        return self._vcirc_dimless(x) * self.scale_velocity


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     lightring =