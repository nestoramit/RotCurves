import numpy as np
import pandas as pd
import os
from scipy import interpolate as scp_interp
from scipy import integrate as scp_integrate
from utils import *
from calc_lookup_tables import create_GaussianRing_lookuptable
from calc_lookup_tables import create_noordermeer_lookuptable
from calc_lookup_tables import solve_noordermeer_integral

class SersicObject:
    def __init__(self, mass, re, sersic_n=1., q=0.2, light_weighting=True, apply_2D=True, ring_absorption_coeff=None, running_in_cluster=False):

        self.verbose = False # for debugging

        self.mass = mass
        self.re = re
        self.n = sersic_n
        if self.n is None:
            print('Sersic index not specified. Assuming n=1...')
            self.n = 1.
        self.q = q
        self.invq = np.round(1/q, 1)
        self.b = self._sersic_b(sersic_n)
        if self.n == 1:
            self.rd = re / 1.68
        else:
            self.rd = self.re / np.power(self.b, self.n)
        self.Sig0 = self.mass / (2 * pi * self.rd ** 2 * self.n * scp_functions.gamma(2 * self.n))

        self.light_weighting = light_weighting
        self.mass_to_light = 1.
        self.apply_2D = apply_2D
        self.running_in_cluster = running_in_cluster
        self.ring_absorption_coeff = ring_absorption_coeff

        # if not self.running_in_cluster:
        #     self.integral_results_folder = os.path.join(main_path, "code", "github-rotationcurves", "lookup_tables", "Noordermeer_lookup_tables")
        # else:
        #     self.integral_results_folder = "/mnt/sdceph/users/ycohen/Nestor/inputs/Noordermeer_lookup_tables//"
        # self.integral_results_path_finder(self.q_table())
        # self.integral_results = np.load(self.integral_results_path)

        self.integral_results = Noordermeer_lookup_tables[self.n][self.q_table()]

    def _is_massive(self):
        if np.log10(self.mass / M_solar) > 5.:
            return True
        else:
            return False

    def _sersic_b(self, n):
        return sersic_b(n)

    def integral_results_path_finder(self, q):
        self.integral_results_path = self.integral_results_folder + "/noor_n%2.2f_q%2.2f.npy" % (self.n, q)

    def q_table(self):
        # q_list = np.asarray(list(set([float(x[x.find('_q') + 2:x.find('.npy')]) for x in os.listdir(self.integral_results_folder) if x.find('npy') > 0])))
        q_list = Noordermeer_lookup_tables['q_list']
        if self.q in q_list:
            q_table = self.q
        else:
            q_table = q_list[np.argmin(np.abs(q_list - self.q))]
            if self.verbose:
                logger.warning('Sersic disk q: non-exact value, using %2.3f instead of %2.3f' % (q_table, self.q))
            if np.abs(q_table - self.q) > 1.:
                if self.verbose:
                    logger.warning('Sersic disk q: No close q value. Creating new lookup table for q = %2.2f' % self.q)
                create_noordermeer_lookuptable(q_range=[np.round(self.q, 2)], n_range=[np.round(self.q, 2)],
                                                         running_in_cluster=self.running_in_cluster)

        return q_table

    def density_function(self, r):
        return self.Sig0 * np.exp(-np.power(np.divide(np.abs(r), self.rd), 1/self.n))

    def density_prime_function(self, r):
        return -self.Sig0 / (self.n * np.abs(r)) * np.power(np.divide(np.abs(r), self.rd), 1/self.n) * np.exp(-np.power(np.divide(np.abs(r), self.rd), 1/self.n))

    def density_prime_to_density_function(self, r):
        return -np.power(np.divide(np.abs(r), self.rd), 1/self.n) * np.divide(1, self.n * np.abs(r), out=np.zeros_like(r), where=r!=0)

    def mass_function(self, r):
        return self.mass * scp_functions.gammainc(2 * self.n, np.power(np.abs(r) / self.rd, 1 / self.n))

    def observed_density_function(self, r):
        if self.ring_absorption_coeff is not None:
            self.observed_density_function = \
                self.Sig0 * \
                np.abs(np.exp(-np.power(np.divide(np.abs(r), self.rd), 1/self.n)) -
                       np.exp(-self.ring_absorption_coeff * np.power(np.abs(r) / self.rd, 1 / self.n)))
        else:
            self.observed_density_function = self.density_function(r)

    def light_profile(self, x, y=None):
        if self.apply_2D:
            # light_profile = np.exp(-np.power(np.divide(x**2 + y**2, self.rd**2), 1))
            light_profile = np.exp(-np.power(np.divide(x**2 + y**2, self.rd**2), 1 / (self.n)))
            if self.ring_absorption_coeff is not None:
                light_profile += -np.exp(-self.ring_absorption_coeff * np.power(np.divide(x**2 + y**2, self.rd**2), 1 / (self.n)))
        else:
            light_profile = np.exp(-np.power(np.divide(np.abs(x), self.rd), 1 / self.n))
            if self.ring_absorption_coeff is not None:
                light_profile += -np.exp(-self.ring_absorption_coeff * np.power(np.abs(x) / self.rd, 1 / self.n))
        I0 = self.Sig0 * self.mass_to_light

        return light_profile * I0

    def get_RC(self, R_array, use_lookuptable=True):
        self.V2 = self.V_Noordermeer(R=R_array, M=self.mass, Rhalf=self.re, q=self.q, n=self.n, use_lookuptable=use_lookuptable)
        self.V = np.sqrt(np.maximum(self.V2, 0)) * np.sign(R_array)
        self.V = np.nan_to_num(self.V)

    def get_local_density(self, R_array):
        self.dSig_dr = lambda r: - 1 / (self.n * r) * np.power(np.divide(r, self.rd), 1 / self.n) * self.density_function(r)
        self.local_density_func = lambda r: - 1 / pi * scp_integrate.quad(lambda x: self.dSig_dr(x) / np.sqrt(x**2 - r**2), a=r, b=1e3 * self.re)[0]

        self.local_density = []
        for r in R_array:
            value = self.local_density_func(np.abs(r))
            self.local_density.append(value)
        self.local_density = np.asarray((self.local_density))

    def V_Noordermeer(self, R, M=None, Rhalf=None, q=None, n=None, use_lookuptable=True):
        R = np.asarray(R)
        absR = np.abs(R)

        if M is None:
            M = self.mass
        if Rhalf is None:
            Rhalf = self.re
        if q is None:
            q = self.q
        if n is None:
            n = self.n

        if use_lookuptable:
            interpolator = scp_interp.CubicSpline(x=self.integral_results[:, 0], y=self.integral_results[:, 1])
            Vn2 = interpolator(absR/Rhalf)
        else:
            if isinstance(absR, float) or isinstance(absR, int):
                Vn2 = solve_noordermeer_integral(q, n, absR / Rhalf)
            else:
                Vn2 = []
                for r in absR:
                        Vn2.append(solve_noordermeer_integral(q, n, r / Rhalf))

        C = 2 * (G * M / Rhalf) * np.power(self.b, 2 * n + 1) / (pi * n * n * scp_functions.gamma(2 * n))
        return C * Vn2

class GaussianRingObject:
    def __init__(self, mass=None, rpeak=None, ring_FWHM=None, light_weighting=True,
                 running_in_cluster=False, apply_2D=True, verbose=False, use_lookuptable=True):
        if mass is None:
            raise TypeError('Gaussian ring: mass cannot be None!')
        if (rpeak is None) and (ring_FWHM is None):
            raise TypeError('Gaussian ring: Must specify either rpeak & FWHM')

        self.mass = mass
        self.light_weighting = light_weighting
        self.mass_to_light = 1.
        self.apply_2D = apply_2D
        self.running_in_cluster = running_in_cluster
        self.verbose = verbose
        self.use_lookuptable = use_lookuptable
        self.A = 4 * np.log(2)

        self.rpeak = rpeak
        self.FWHM = ring_FWHM
        self.sigma_ring = self.FWHM / FWHM2sig
        self.invh = self.rpeak / self.FWHM
        # self.invh_table = self.invh_table()

        # if not self.running_in_cluster:
        #     self.integral_results_folder = os.path.join(main_path, "code", "github-rotationcurves", "lookup_tables", "GaussianRing_lookup_tables")
        # else:
        #     self.integral_results_folder = "/mnt/sdceph/users/ycohen/Nestor/inputs/GaussianRing_lookup_tables//"
        # self.integral_results_path_finder(self.invh_table())
        # self.integral_results = np.load(self.integral_results_path)

        self.integral_results = GaussianRing_lookup_tables[self.invh_table()]

        self.M0 = self.mass / self.mass_function_dimless(np.inf)
        self.Sig0 = self.M0 / (2 * np.pi * self.rpeak ** 2)
        self.pot0 = 4 * G * self.Sig0 * self.rpeak
        self.V0 = np.sqrt(self.pot0)

        self.V2 = None
        self.V = None

    def _is_massive(self):
        if np.log10(self.mass / M_solar) > 5.:
            return True
        else:
            return False

    # def integral_results_path_finder(self, invh):
    #     self.integral_results_path = self.integral_results_folder + "/Gauss_invh_%2.2f.npy" % invh

    def invh_table(self, bt_min=False):
        # invh_list = np.asarray([float(x[x.find('invh_') + 5:x.find('.npy')]) for x in os.listdir(self.integral_results_folder) if x.find('npy') > 0])
        if not bt_min:
            invh_list = GaussianRing_lookup_tables['invh_list']
        else:
            invh_list = GaussianRing_BTmin_lookup_tables['invh_list']

        if self.invh in invh_list:
            invh_table = self.invh
        else:
            invh_table = invh_list[np.argmin(np.abs(invh_list - self.invh))]
            if self.verbose:
                logger.warning('Gaussian Ring invh: non-exact value, using %2.3f instead of %2.3f' % (invh_table, self.invh))
            if np.abs(invh_table - self.invh) > 0.3:
                logger.warning('Gaussian Ring invh: No close invh value. Creating new lookup table for invh = %2.3f' % self.invh)
                create_GaussianRing_lookuptable(invh_range=[np.round(self.invh, 2)], running_in_cluster=self.running_in_cluster)

        return invh_table


    def density_function_dimless(self, x):
        return np.exp(-self.A * self.invh ** 2 * np.power(x - 1., 2.))

    def mass_function_dimless(self, x):
        return integrate_an_array(lambda t: t * self.density_function_dimless(t), 0, x)

    def density_prime_function_dimless(self, x):
        return - 2 * self.A * self.invh**2 * (x - 1) * self.density_function_dimless(x)

    def I_function(self, a):
        return scp_integrate.quad(
            lambda x: x * self.density_function_dimless(x) / np.sqrt(x ** 2 - a ** 2),
            a, np.inf)[0]

    def Iprime_function(self, a):
        return scp_integrate.quad(
            lambda x: a * self.density_prime_function_dimless(np.sqrt(x ** 2 + a ** 2)) / np.sqrt(x ** 2 + a ** 2),
            0, np.inf)[0]

    def potential_function(self, x):
        integral = scp_integrate.quad(
            lambda a: np.arcsin(np.minimum(2 * a / ((a + x) + np.abs(a - x)), 1.)) * self.Iprime_function(a), 0,
            np.inf)[0]

        return integral

    def V2_function(self, x):
        integral = scp_integrate.quad(lambda t: - self.Iprime_function(t) * t / np.sqrt(x ** 2 - t ** 2), 0, x)[0]

        return integral

    def density_prime_to_density_function(self, r):
        return - 2 * self.A * self.invh ** 2 * (np.abs(r) / self.rpeak - 1) / self.rpeak

    def re(self):
        xe = solve_numerical_using_brentq(
            lambda xe: self.mass_function_dimless(xe) - 0.5 * self.mass_function_dimless(np.inf), p0=1.)
        return xe * self.rpeak

    def get_profiles(self, R_array, use_lookuptable=True):
        x_array = R_array / self.rpeak
        self.density_profile = self.Sig0 * np.vectorize(self.density_function_dimless)(x_array)

        if use_lookuptable:
            interpolator = scp_interp.CubicSpline(x=self.integral_results[:, 0], y=self.integral_results[:, 2])
            self.mass_profile = self.M0 * interpolator(x_array)
        else:
            self.mass_profile = self.M0 * np.vectorize(self.mass_function_dimless)(x_array)

    def get_RC(self, R_array):
        starttime = time.time_ns()
        self.solve_velocity_integral(R_array=R_array, use_lookuptable=self.use_lookuptable)

        self.V2 = self.V0 ** 2 * self.V2_dimless
        self.V = self.V0 * self.V_dimless * np.sign(R_array)
        # ### TODO: ADDED BECAUSE OF A MISTAKE, DELETE AFTER CORRECTION IN THE LOOKUP TABLES
        # if self.use_lookuptable:
        #     self.V2 = self.V0 ** 2 * self.V2_dimless * self.invh
        #     self.V = self.V0 * self.V_dimless * np.sign(R_array) * np.sqrt(self.invh)
        # else:
        #     self.V2 = self.V0 ** 2 * self.V2_dimless
        #     self.V = self.V0 * self.V_dimless * np.sign(R_array)
        # ### TODO: UP TO HERE

        if self.verbose:
            print('get RC runtime: %s ms' % np.round((time.time_ns() - starttime) * 1e-6, 0))

    def light_profile(self, x, y=None):
        if self.apply_2D:
            light_profile = np.exp(- np.divide(np.power(np.sqrt(x**2 + y**2) - self.rpeak, 2), 2 * self.sigma_ring ** 2))
        else:
            light_profile = self.density_function_dimless(x / self.rpeak)
        I0 = self.Sig0 * self.mass_to_light

        return light_profile * I0

    def solve_velocity_integral(self, R_array, use_lookuptable=True):
        if use_lookuptable:
            X_array = np.abs(R_array) / self.rpeak
            interpolator = scp_interp.CubicSpline(x=self.integral_results[:, 0],
                                                  y=self.integral_results[:, 1])
            self.V2_dimless = interpolator(X_array)
        else:
            X_array = np.abs(R_array) / self.rpeak
            self.V2_dimless = np.vectorize(self.V2_function)(X_array)

        self.V_dimless = np.sqrt(np.maximum(self.V2_dimless, 0))

    def get_potential(self, R_array):
        starttime = time.time_ns()
        self.solve_potential_integral(R_array=R_array)

        self.potential = self.pot0 * self.potential_dimless
        if self.verbose:
            print('get potential runtime: %s ms' % np.round((time.time_ns() - starttime) * 1e-6, 0))

    def solve_potential_integral(self, R_array):
        X = np.abs(R_array) / self.rpeak
        self.potential_dimless = np.vectorize(self.potential_function)(X)

    def find_minimal_bulge(self, bulge_re=kpc, bulge_sersicn=4.0, bulge_q=1.0, use_lookuptable=True):
        # invh_table = self.invh_table()
        # if (invh_table in GaussianRing_BTmin_lookup_tables['invh_list']) and use_lookuptable:
        if use_lookuptable:
            GaussianRing_BTmin_lookuptable = GaussianRing_BTmin_lookup_tables[self.invh_table(bt_min=True)]
            interpolator = scp_interp.CubicSpline(x=GaussianRing_BTmin_lookuptable[: ,0],
                                                  y=GaussianRing_BTmin_lookuptable[:, 1])
            BT_min = interpolator(self.rpeak/kpc)

        else:
            logger.warning('Gaussian Ring BT min: No lookuptable found. Calculating ...')

            starttime = time.time_ns()
            R_array = np.logspace(-2, np.log10(2), num=51) * self.rpeak
            self.get_RC(R_array)

            N = int(1e3)
            i = 0
            for BT in np.logspace(-3, 0, num=N):
                bulge_mass = self.mass * BT / (1 - BT)
                bulge = SersicObject(mass=bulge_mass, re=bulge_re, sersic_n=bulge_sersicn, q=bulge_q,
                                     running_in_cluster=self.running_in_cluster)
                bulge.get_RC(R_array)
                V2_new = self.V2 + bulge.V2
                if all(V2_new > 0):
                    BT_min = np.ceil(BT * 1e4) * 1e-4
                    logger.warning('Gaussian Ring BT min: Found BTmin = %.4f for invh = %.2f ...' % (BT_min, self.invh))
                    break
                if i == N-1:
                    logger.warning(r"Couldn't find central stabilizing mass for the given Gaussian ring distribution.")
                    BT_min = 0.99
                i += 1
            if self.verbose:
                print('find minbulge runtime: %s ms' % np.round((time.time_ns() - starttime) * 1e-6, 0))

        self.BT_min = BT_min

    def find_rmin(self):
        starttime = time.time_ns()
        if not os.path.exists(self.integral_results_path):
            print('No lookup table for invh=%s. Creating table...' % np.round(self.invh, 1))
            create_GaussianRing_lookuptable(invh_range=[np.round(self.invh, 2)],
                                                      running_in_cluster=self.running_in_cluster)
            self.integral_results_df = pd.read_csv(self.integral_results_path, index_col=[0], header=[0]).squeeze("columns")
        x_min_idx = np.argwhere(self.integral_results_df['V2'].values == next(i for i in self.integral_results_df['V2'] if i >= 0))[0]
        x_min = self.integral_results_df.index.values[x_min_idx][0]

        self.x_min = x_min
        self.r_min = self.x_min * self.rpeak
        if self.verbose:
            print('find rmin runtime: %s ms' % np.round((time.time_ns() - starttime) * 1e-6, 0))

# class FreemanObject:
#     def __init__(self, mass, re):
#         self.mass = mass
#         self.re = re
#         self.rd = re / 1.68
#
#         self.density_function = lambda x: self.mass / (2*pi * self.rd ** 2) * np.exp(-(x / self.rd))
#         self.mass_function = lambda x: self.mass * (1 - (1 + x / self.rd) * np.exp(-(x / self.rd)))
#
#     def get_RC(self, R_array):
#         A = 2 * G * self.mass / self.rd
#         y = np.abs(R_array) / (2 * self.rd)
#         self.V = np.sqrt(A * y ** 2 * (scp_functions.iv(0, y) * scp_functions.kn(0, y) - scp_functions.iv(1, y) * scp_functions.kn(1, y))) * np.sign(R_array)
#         self.V = np.nan_to_num(self.V)

class HaloObject:
    def __init__(self, profile, mass, concentration_parameter, redshift, alpha=1., beta=3., gamma=1., contracted=False):
        self.type = profile
        if self.type is None:
            print('halo type not specified. assuming NFW profile...')
            self.type = 'NFW'
        self.contracted = contracted
        self.mass = mass
        self.z = redshift
        self.cosmology_parameters = cosmology(self.z)
        self.c = concentration_parameter
        self.rvir = (self.mass / ((4*pi/3)*200*self.cosmology_parameters.rho_c)) ** (1/3)
        self.rs = self.rvir / self.c
        self.rho_vir = 200 * self.cosmology_parameters.rho_c
        self.V_vir = np.sqrt(G * self.mass / self.rvir)

        if self.type in ['NFW', 'nfw']:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
        elif self.type in ['Burkert', 'burkert']:
            self.alpha = None
            self.beta = None
            self.gamma = None
        elif self.type in ['dekel', 'Einasto']:
            self.alpha = alpha
            self.beta = None
            self.gamma = None
        elif self.type in ['Lucky 13', 'Lucky13', 'lucky 13', 'lucky13', 'L13', 'l13']:
            self.alpha = None
            self.beta = None
            self.gamma = None
        elif self.type in ['Dekel', 'dekel', 'Dekel-Zhao', 'dekel-zhao', 'Dekel-Zhao', 'DZ', 'dz']:
            self.alpha = alpha
            self.beta = 3.5
            self.gamma = 2.
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        self.overdensity = 200 * (self.c ** 3) / self.get_fM(self.c)
        self.characteristic_density = self.cosmology_parameters.rho_c * self.overdensity

    def density_function(self, r):
        return self.characteristic_density * self.get_frho(np.abs(r) / self.rs)

    def mass_function(self, r):
        return 4 * pi / 3 * (self.rs ** 3) * self.characteristic_density * self.get_fM(np.abs(r) / self.rs)

    def update_powers(self, alpha, beta, gamma):
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma

    def get_fM(self, x):
        """
        :param x: r / rs [dimless]
        :param halo_profile: string: NFW or burkert
        :param alpha: inner slope
        :param beta: outer slope
        :param gamma: transition parameter
        :return: value for fM
        """

        fM = None

        if self.type in ['NFW', 'nfw'] and self.alpha == 1. and self.beta == 3. and self.gamma == 1.:
            fM = 3 * (np.log(1 + x) - x / (1 + x))

        elif self.type in ['NFW', 'nfw'] and self.gamma == 1.:
            fM = 3 / (3-self.alpha) * (x ** (3-self.alpha) * scp_functions.hyp2f1(3-self.alpha, self.beta-self.alpha, 4-self.alpha, -x))

        elif self.type in ['NFW', 'nfw']:
            fM = 3 / (3-self.alpha) * (x ** (3-self.alpha) * scp_functions.hyp2f1(self.gamma*(3-self.alpha), self.beta-self.alpha, 4-self.alpha, -(x**(1/self.gamma))))

        elif self.type in ['Burkert', 'burkert']:
            fM = 3/2 * (np.log(1 + x**2) / 2 + np.log(1 + x) - np.arctan(x))

        elif self.type in ['Einasto', 'dekel'] and self.alpha == 1.:
            fM = 3/8 * np.exp(2) * scp_functions.gammainc(3, 2 * x) * scp_functions.gamma(3)

        elif self.type in ['Einasto', 'dekel']:
            fM = 3/2 * np.power(2/self.alpha, 1 - 3/self.alpha) * np.exp(2/self.alpha) * scp_functions.gammainc(3/self.alpha, 2/self.alpha * np.power(x, self.alpha)) * scp_functions.gamma(3/self.alpha)

        elif self.type in ['Dekel', 'dekel', 'Dekel-Zhao', 'dekel-zhao', 'Dekel-Zhao', 'DZ']:
            fM = x ** 3 * (1 + np.sqrt(x)) * self.get_frho(x)

        elif self.type in ['Lucky 13', 'Lucky13', 'lucky 13', 'lucky13', 'L13', 'l13']:
            fM = 3 * (np.log(1 + x) - np.divide(x * (3*x + 2), 2 * np.power(1 + x, 2)))

        else:
            f = lambda s: 3 * s **2 * self.get_frho(s)
            if isinstance(x, (int, float, np.int64)):
                fM = scp_integrate.quad(f, 0, x)[0]
            else:
                fM = []
                for x_idx in x:
                    fM.append(scp_integrate.quad(f, 0, x_idx)[0])
                fM = np.array(fM)

        return fM

    def get_frho(self, x):
        """

        :param x: r / rs (dimless)
        :param halo_profile: halo profile type (string)
        :param alpha: inner slope
        :param beta: outer slope
        :param gamma: transition parameter

        :return: density function
        """
        frho = None

        if self.type in ['NFW', 'alphaNFW', 'Dekel', 'dekel', 'Dekel-Zhao', 'dekel-zhao', 'Dekel-Zhao', 'DZ', 'dz']:
            frho = np.divide(1, np.power(x, self.alpha) * np.power(1 + np.power(x, 1/self.gamma), self.gamma * (self.beta - self.alpha)), out=np.zeros_like(x), where=x!=0)

        elif self.type in ["burkert", "Burkert"]:
            frho = 1 / ((1 + x) * (1 + np.power(x, 2)))

        elif self.type in ['Einasto', 'dekel']:
            frho = np.exp(-2/self.alpha * (np.power(x, self.alpha) - 1))

        elif self.type in ['Lucky 13', 'Lucky13', 'lucky 13', 'lucky13', 'L13', 'l13']:
            frho = np.power(1 + x, -3)

        if frho is None:
            raise Exception("Error in halo name!")

        return frho

    def get_mass_profile(self, R_array, disk=None, bulge=None, contracted=None):
        if contracted is None:
            contracted = self.contracted

        if not contracted:
            self.mass_profile = self.mass_function(R_array)
            self.density_profile = self.density_function(R_array)
        else:
            if disk is None or bulge is None:
                print("can't perform adiabatic contraction on the halo if disk and bulge are None!")
                print("\nContinuing with original halo...")
            else:
                Mhalo_contracted = []
                R_not_contracted = []
                contraction_coefficient = []

                proto_halo_mass = self.mass + disk.mass + bulge.mass
                proto_halo = HaloObject(self.type, mass=proto_halo_mass, concentration_parameter=self.c,
                                        redshift=self.z, alpha=self.alpha, beta=self.beta, gamma=self.gamma, contracted=False)

                md = disk.mass / proto_halo.mass
                mb = bulge.mass / proto_halo.mass
                mdm = 1 - md - mb

                disk.get_RC(R_array)
                bulge.get_RC(R_array)

                for idx, rf in enumerate(abs(R_array)):

                    if rf == 0:
                        contraction_coefficient.append(1.)
                        R_not_contracted.append(0.)
                        Mhalo_contracted.append(0.)
                    else:

                        # func = lambda t: (rf ** 2) * (disk.V2[idx] + bulge.V2[idx]) - G * proto_halo.mass_function(t) * (t - mdm * rf)
                        # ri = help.solve_numerical_using_brentq(func, p0=rf, p1_oom=4)
                        # eta = rf / ri

                        func = lambda eta: G * proto_halo.mass_function(rf / eta) / (rf / eta) * (1 - eta * (1 - md - mb)) - eta**2 * (disk.V2[idx] + bulge.V2[idx])
                        eta = solve_numerical_using_brentq(func, p0=1., p1_oom=4)
                        ri = rf / eta

                        Mhalo_contracted_idx = proto_halo.mass_function(ri) * mdm
                        if Mhalo_contracted_idx < 0:
                            raise Exception("Halo mass cannot be negative!")

                        contraction_coefficient.append(eta)
                        R_not_contracted.append(ri)
                        Mhalo_contracted.append(Mhalo_contracted_idx)

                self.R_not_contracted = np.array(R_not_contracted)
                self.mass_profile = np.array(Mhalo_contracted)
                self.contraction_coefficient = contraction_coefficient
                self.original_density_profile = self.density_function(R_array)
                self.original_mass_profile = self.mass_function(R_array)

                self.density_profile = np.abs( np.divide( np.diff(self.mass_profile) / np.diff(R_array), (4*pi* R_array[1:]**2),
                                                          where=R_array[1:]!=0, out=np.zeros_like(np.diff(R_array)) ) )

                # amit_k = 3 / 2 * (self.overdensity / (200 * self.c)) * (disk.rd / self.rvir) ** 2 * (self.mass / disk.mass)
                # tplus = np.cbrt(1 + np.sqrt(1 + (4*amit_k * (1-md)**3) / 27))
                # tminus = np.cbrt(1 - np.sqrt(1 + (4*amit_k * (1-md)**3) / 27))
                # A = np.cbrt(amit_k / 2) * (tplus + tminus)
                # self.rs_effective = self.rs * A
                # self.c_effective = self.rvir / self.rs_effective

    def get_RC(self, R_array, disk=None, bulge=None, contracted=None):
        if contracted is None:
            contracted = self.contracted

        self.get_mass_profile(np.abs(R_array), disk=disk, bulge=bulge, contracted=contracted)
        self.V2 = np.divide(G * self.mass_profile, np.abs(R_array), out=np.zeros_like(self.mass_profile), where=R_array!=0)
        self.V = np.sqrt(self.V2) * np.sign(R_array)
        self.V = np.nan_to_num(self.V)
        if self.contracted:
            self.V_original = np.sqrt(np.divide(G * self.original_mass_profile, np.abs(R_array), out=np.zeros_like(self.original_mass_profile), where=R_array!=0)) * np.sign(R_array)