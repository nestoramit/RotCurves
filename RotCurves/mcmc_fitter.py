import os
import logging
import time
import datetime
import numpy as np
import pandas as pd
import emcee
import corner
import parmap
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from multiprocessing import Pool
from scipy.interpolate import CubicSpline
from matplotlib.ticker import MultipleLocator

from RotCurves.base_utils import figure, colors
from RotCurves.rotation_curve import RotationCurveObject, calculate_fraction_at_re
from RotCurves.mass_model import create_components

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')

"""
These functions are used to create MCMC variables, unpack values from theta, and add f to theta and samples.
"""

def create_mcmc_variables(galaxy, mcmc_hparameters):
    initial_values = []

    switches = galaxy.switches
    for parameter in switches["parameters"]:
        if switches["parameters"][parameter] == 1:
            initial_values.append(galaxy.priors[parameter].initial)

    initial_values = np.array(initial_values)
    ndim = len(initial_values)
    nwalkers = mcmc_hparameters["nwalkers"]
    p0 = np.array([np.array(initial_values) + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)])

    return p0


def unpack_values_from_theta(theta, galaxy):
    '''
    order of params in theta:
    Re, M_baryon, M_vir, BT, DT, R_peak, ring_FWHM, sigma, c, alpha, i
    '''

    switches = galaxy.switches

    parameters = {}
    idx = 0
    for parameter in switches["parameters"]:
        if switches["parameters"][parameter] == 1:
            parameters[parameter] = float(theta[idx])
            idx += 1
        else:
            parameters[parameter] = float(galaxy.priors[parameter].initial)

    # if switches["use Moster"] == 1:
    #     parameters['M_vir'] = Mvir_Moster2018(galaxy.z, parameters['M_baryon'])

    return parameters


def unpack_mcmc_walker_results(samples, nwalkers, galaxy, mcmc_hparameters):
    starttime = time.time_ns()
    switches = galaxy.switches

    mcmc_fluxes = []
    mcmc_rotation_curves = []
    mcmc_dispersion_curves = []

    for theta in samples[np.random.randint(len(samples), size=nwalkers)]:
        # [re, logm_baryon, logm_vir, BT, sigma, c, alpha, i] = unpack_values_from_theta(theta, galaxy)
        model_params = unpack_values_from_theta(theta, galaxy)

        mass_components = create_components(
            include_halo=galaxy.mass_components_switches['halo'], include_disk=galaxy.mass_components_switches['disk'],
            include_ring=galaxy.mass_components_switches['ring'], include_bulge=galaxy.mass_components_switches['bulge'],
            z=galaxy.z, halo_profile=galaxy.halo_profile, logM_vir=model_params['M_vir'], c=model_params['c'],
            alpha=model_params['alpha'], AC=switches['adiabatic contraction'], logM_baryon=model_params['M_baryon'],
            DT=model_params['DT'], disk_re=model_params['Re'], disk_n=galaxy.disk_n, disk_q=galaxy.disk_q, disk_lw=galaxy.disk_lw,
            BT=model_params['BT'], bulge_n=galaxy.bulge_n, bulge_q=galaxy.bulge_q, bulge_lw=galaxy.bulge_lw,
            ring_rpeak=model_params['R_peak'], ring_FWHM=model_params['ring_FWHM'], ring_lw=galaxy.ring_lw,
            running_in_cluster=mcmc_hparameters['running in cluster'], apply2D=galaxy.apply_2D)

        RC = RotationCurveObject(galaxy=galaxy, Halo=mass_components['halo'], Disk=mass_components['disk'],
                                 Ring=mass_components['ring'], Bulge=mass_components['bulge'],
                                 sigma_dispersion=model_params['sigma'], pressure_support=galaxy.pressure_support,
                                 inclination=model_params['i'], sigma_beam=galaxy.sigma_beam, apply_2D=galaxy.apply_2D,
                                 include_beam_smearing=True)

        mcmc_fluxes.append(RC.smeared_light_profile)
        mcmc_rotation_curves.append(RC.smeared_with_inclination)
        mcmc_dispersion_curves.append(RC.velocity_dispersion)

    walker_resuts_dictionary = {
        "flux": mcmc_fluxes,
        "RC": mcmc_rotation_curves,
        "dispersion": mcmc_dispersion_curves
    }

    runtime = time.time_ns() - starttime
    print('Unpacking walkers runtime: %s minutes' % np.round(runtime * 1e-9 / 60, 1))

    return walker_resuts_dictionary


def add_f_to_theta(galaxy, theta, mcmc_hparameters):
    switches = galaxy.switches
    model_params = unpack_values_from_theta(theta, galaxy)

    mass_components = create_components(
        include_halo=galaxy.mass_components_switches['halo'], include_disk=galaxy.mass_components_switches['disk'],
        include_ring=galaxy.mass_components_switches['ring'], include_bulge=galaxy.mass_components_switches['bulge'],
        z=galaxy.z, halo_profile=galaxy.halo_profile, logM_vir=model_params['M_vir'], c=model_params['c'],
        alpha=model_params['alpha'], AC=switches['adiabatic contraction'],
        logM_baryon=model_params['M_baryon'], DT=model_params['DT'], disk_re=model_params['Re'], disk_n=galaxy.disk_n,
        disk_q=galaxy.disk_q, disk_lw=galaxy.disk_lw, BT=model_params['BT'], bulge_n=galaxy.bulge_n, bulge_q=galaxy.bulge_q,
        bulge_lw=galaxy.bulge_lw, ring_rpeak=model_params['R_peak'], ring_FWHM=model_params['ring_FWHM'], ring_lw=galaxy.ring_lw,
        running_in_cluster=mcmc_hparameters['running in cluster'], apply2D=galaxy.apply_2D)

    if mass_components['halo'] is not None:
        if mass_components['disk'] is not None:
            fraction_i = calculate_fraction_at_re(mass_components=mass_components, reval=model_params['Re'])
        elif mass_components['ring'] is not None:
            fraction_i = calculate_fraction_at_re(mass_components=mass_components, reval=model_params['R_peak'])
        else:
            fraction_i = 0
            logger.warning('No disk or ring component, cant evaluate DM fractions! Setting f=0...')
    else:
        logger.warning('No halo component, cant evaluate DM fractions! Setting f=0...')


    updated_theta = np.append(theta, fraction_i)

    return updated_theta


def add_f_to_samples(samples, galaxy, mcmc_hparameters):
    samples_with_f = []
    for theta in samples:
        updated_theta = add_f_to_theta(galaxy, theta, mcmc_hparameters)
        samples_with_f.append(updated_theta)

    return samples_with_f


def add_f_to_samples_mp(idx, galaxy, samples, mcmc_hparameters):
    updated_theta = add_f_to_theta(galaxy, samples[idx], mcmc_hparameters)

    return updated_theta


"""
This class is used to create the priors for the MCMC fitting.
"""
class Prior:
    def __init__(self, type, initial_value, min_value, max_value, gauss_sigma):
        self.type = self.type_altnames(type)
        self.initial = initial_value
        self.min = min_value
        self.max = max_value
        self.sig = gauss_sigma

    def type_altnames(self, type):
        if type in [ 'fixed', 'Fixed']:
            return 'fixed'
        if type in ['uniform', 'u', 'Uniform', 'flat', 'f']:
            return 'uniform'
        elif type in ['gaussian', 'g', 'Gaussian']:
            return 'gaussian'
        else:
            raise ValueError(f"Unknown prior type: {self.type}")

    def lnprob(self, value):
        if self.type == 'fixed':
            return 0.0
        elif self.min <= value <= self.max:
            if self.type == 'uniform':
                return 0.0
            if self.type == 'gaussian':
                return -1/2 * ((value - self.initial) / self.sig)**2
        else:
            return - np.inf

"""
These functions are used to calculate the log-prior and log-likelihood of the MCMC fitting.
The log-prior is calculated based on the priors defined in the galaxy model object.
The log-likelihood is calculated based on the data and the model, via least squares.

"""
def lnprior(theta, galaxy):
    '''
    :param theta: model parameters.
        order: Re, M_baryon, M_vir, BT, DT, R_peak, ring_FWHM, sigma, c, alpha, i
    :param galaxy: galaxy model object
    :param mcmc_hparameters: dict of mcmc variables
    '''

    switches = galaxy.switches

    lp = 0
    theta = np.array(theta)

    theta_idx = 0
    for parameter in switches["parameters"]:
        prior = galaxy.priors[parameter]

        # extract value of the specific parameter.
        # If switch is 1 (on) it's taken from theta, if switch is 0 (off) it's taken as the initial value
        if switches["parameters"][parameter] == 1:
            value = theta[theta_idx]
            theta_idx += 1
        else:
            value = prior.initial

        # update the log-prior
        lp += prior.lnprob(value)

        # For B/T, check if the minimal bulge critirea for a ring is OK
        if galaxy.fit_goals['velocity'] or galaxy.fit_goals['dispersion']:
            if parameter == 'BT':
                BT_value = value
                if galaxy.mass_components_switches['ring'] and BT_value != 0:
                    if galaxy.mass_components['ring']._is_massive():
                        if BT_value < galaxy.mass_components['ring'].min_stabilizing_mass():
                            return -np.inf

        # For D/T, check it against B/T to make sure it is <= 1.
        if parameter == 'DT':
            DT_value = value
            if galaxy.mass_components_switches['disk'] and galaxy.mass_components_switches['ring']:
                if BT_value + DT_value > 1.:
                    return -np.inf

    return lp


def get_RC_from_theta_for_lnlike(theta, galaxy, mcmc_hparameters):
    theta = np.array(theta)
    model_params = unpack_values_from_theta(theta, galaxy)

    # calculate rotation curves
    mass_components = create_components(
        include_halo=galaxy.mass_components_switches['halo'], include_disk=galaxy.mass_components_switches['disk'],
        include_ring=galaxy.mass_components_switches['ring'], include_bulge=galaxy.mass_components_switches['bulge'],
        z=galaxy.z, halo_profile=galaxy.halo_profile, logM_vir=model_params['M_vir'], c=model_params['c'],
        alpha=model_params['alpha'], AC=galaxy.switches['adiabatic contraction'],
        logM_baryon=model_params['M_baryon'], DT=model_params['DT'], disk_re=model_params['Re'], disk_n=galaxy.disk_n,
        disk_q=galaxy.disk_q, disk_lw=galaxy.disk_lw,
        BT=model_params['BT'], bulge_n=galaxy.bulge_n, bulge_q=galaxy.bulge_q, bulge_lw=galaxy.bulge_lw,
        ring_rpeak=model_params['R_peak'], ring_FWHM=model_params['ring_FWHM'], ring_lw=galaxy.ring_lw,
        running_in_cluster=mcmc_hparameters['running in cluster'], apply2D=galaxy.apply_2D)

    RC = RotationCurveObject(galaxy=galaxy, Halo=mass_components['halo'], Disk=mass_components['disk'],
                             Ring=mass_components['ring'], Bulge=mass_components['bulge'],
                             sigma_dispersion=model_params['sigma'], pressure_support=galaxy.pressure_support,
                             inclination=model_params['i'], sigma_beam=galaxy.sigma_beam, apply_2D=galaxy.apply_2D,
                             include_beam_smearing=True)

    return RC


def lnlike(theta, galaxy, mcmc_hparameters):
    '''
    :param theta: model parameters.
        order: Re, M_baryon, M_vir, BT, DT, R_peak, ring_FWHM, sigma, c, alpha, i
    :param galaxy: galaxy model object
    :param mcmc_hparameters: dict of mcmc variables
    '''

    RC = get_RC_from_theta_for_lnlike(theta, galaxy, mcmc_hparameters)

    prob = 0
    # update lnprob from flux fit
    x = galaxy.obsdata_r
    if galaxy.fit_goals['flux']:
        prob = update_prob(prob=prob,
                           xdata=x, ydata=galaxy.obsdata_flux, ydata_err=galaxy.obsdata_flux_err,
                           xinterp=galaxy.radial_space["array"], yinterp=RC.smeared_light_profile)

    # update lnprob from velocity fit
    if galaxy.fit_goals['velocity']:
        prob = update_prob(prob=prob,
                           xdata=x, ydata=galaxy.obsdata_V, ydata_err=galaxy.obsdata_V_err,
                           xinterp=galaxy.radial_space["array"], yinterp=RC.smeared_with_inclination)

    # update lnprob from dispersion fit
    if galaxy.fit_goals['dispersion']:
        prob = update_prob(prob=prob,
                           xdata=x, ydata=galaxy.obsdata_disp, ydata_err=galaxy.obsdata_disp_err,
                           xinterp=galaxy.radial_space["array"], yinterp=RC.velocity_dispersion)

    return prob


def update_prob(prob, xdata, ydata, ydata_err, xinterp, yinterp):
    interpolator = CubicSpline(x=xinterp, y=yinterp)
    y_predicted = interpolator(xdata)
    to_keep = np.argwhere(np.logical_not(np.isnan(ydata)))

    # prob += -0.5 * np.sum(np.power((ydata[to_keep] - y_predicted[to_keep]) / ydata_err[to_keep], 2))
    prob += -0.5 * np.nansum(np.power((ydata[to_keep] - y_predicted[to_keep]) / ydata_err[to_keep], 2))
    return prob


def lnprob(theta, galaxy, mcmc_hparameters):
    lp = lnprior(theta, galaxy)
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf
    else:
        lk = lnlike(theta, galaxy, mcmc_hparameters)
        if not np.isfinite(lk) or np.isnan(lk):
            return -np.inf
        else:
            return lp + lk

"""
Functions to run the mcmc fitter
"""

def find_maximum_frequency(data_array, bins, axis=0):
    '''
    find the most frequent value in a binned histogram of an array.
    works on a columns-basis in the given data_array.
    '''

    N = data_array.shape[1]
    argmax_array = [np.argmax(np.histogram(data_array[:, i], bins=bins)[0]) for i in range(N)]
    max_freq_lower_values = [np.histogram(data_array[:, i], bins=bins)[1][argmax_array[i]] for i in range(N)]
    max_freq_upper_values = [np.histogram(data_array[:, i], bins=bins)[1][argmax_array[i] + 1] for i in range(N)]
    max_freq_values = np.average([max_freq_lower_values, max_freq_upper_values], axis=0)

    return max_freq_values

def check_convergence(sampler, mcmc_hparameters, old_taus):
    aurocorrelation_steps_thersh = mcmc_hparameters['aurocorrelation_steps_thersh']
    tau_tol = mcmc_hparameters['tau_tol']

    taus = emcee.autocorr.integrated_time(sampler.chain, tol=0)
    acceptance_fraction = np.mean(sampler.acceptance_fraction)

    converged = np.all(taus >= 0)
    converged &= np.all(taus * aurocorrelation_steps_thersh < sampler.iteration)
    converged &= np.all(np.abs(old_taus - taus) / taus < tau_tol)
    converged &= (acceptance_fraction <= 0.5)
    converged &= (acceptance_fraction >= 0.2)

    return converged


def run_sampler(mcmc_hparameters, sampler, p0):
    nburnin = mcmc_hparameters["niter"][0]
    niter = mcmc_hparameters["niter"][1]

    starttime = time.time()
    logger.info("Running burn-in...")
    p0_new, _, _ = sampler.run_mcmc(initial_state=p0, nsteps=nburnin, progress=(not bool(mcmc_hparameters["running in cluster"])))
    sampler.reset()
    logger.info("Finished burn in: %s minutes" % round((time.time() - starttime) / 60, 1))

    starttime = time.time()
    logger.info("Running iterations...")

    # Split iterations to chunks and check convergence at each chunk
    # niter_per_loop --> num of chunks
    # converged if num_iter > 100*tau for every param, AND if tau has changed less than 5%
    niter_per_loop = mcmc_hparameters['niter_per_loop']
    if niter_per_loop is None:
        niter_per_loop = nburnin
    niters_converged = niter

    niter_to_run = min(niter, niter_per_loop)
    num_of_loops = int(np.ceil(niter / niter_per_loop))
    old_taus = np.zeros(sampler.ndim)
    for idx in range(1, int(num_of_loops)+1, 1):
        pos, prob, state = sampler.run_mcmc(initial_state=p0_new, nsteps=niter_to_run, progress=(not bool(mcmc_hparameters["running in cluster"])))

        logger.info('autocorrelation time after %3d iterations: %s' % (sampler.iteration, emcee.autocorr.integrated_time(sampler.chain, tol=0)))
        logger.info('acceptance ratio after %3d iterations:     %2.3f (%2.3f)' % (sampler.iteration, np.mean(sampler.acceptance_fraction), np.std(sampler.acceptance_fraction)))

        # Check convergence
        converged = check_convergence(sampler, mcmc_hparameters, old_taus)
        if converged:
            logger.info('CONVERGED after %3d iterations!' % sampler.iteration)
            niters_converged = int(sampler.iteration)
            break
        elif int(sampler.iteration) >= niter:
            logger.info(r'didnt converge, finished after %3d iterations ...' % sampler.iteration)
            break
        p0_new = pos
        niter_to_run = min(niter - idx*niter_per_loop, niter_per_loop)
        old_taus = emcee.autocorr.integrated_time(sampler.chain, tol=0)

    mcmc_hparameters['niters_converged'] = niters_converged
    logger.info("Finished iterations: %s minutes\n" % round((time.time() - starttime) / 60, 1))

    return sampler


def run_mcmc(galaxy, mcmc_hparameters):
    switches = galaxy.switches
    p0 = create_mcmc_variables(galaxy, mcmc_hparameters)
    nwalkers = mcmc_hparameters["nwalkers"]
    ndim = int(np.sum([x for x in switches["parameters"].values()]))
    args = [galaxy, mcmc_hparameters]

    logger.info("Starting mcmc for %s" % galaxy.name)
    backend_filename = '/'.join([galaxy.output_dir, 'mcmc_model.h5'])
    backend = emcee.backends.HDFBackend(backend_filename)
    backend.reset(nwalkers, ndim)

    if mcmc_hparameters["multiprocessing"]:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn=lnprob, args=args, pool=pool, backend=backend,
                                            moves=[(emcee.moves.StretchMove(a=mcmc_hparameters['strecth_move_a']), 1.)])
            sampler = run_sampler(mcmc_hparameters, sampler, p0)

    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, backend=backend,
                                        moves=[(emcee.moves.StretchMove(a=mcmc_hparameters['strecth_move_a']), 1.)])
        sampler = run_sampler(mcmc_hparameters, sampler, p0)

    starttime = time.time()
    logger.info("Adding fractions & arranging data...")
    samples = sampler.flatchain

    # if switches["use Moster"] and switches["parameters"]["M_vir"]:
    #     for theta in samples:
    #         mvir_idx = list(switches["parameters"].keys()).index("M_vir")
    #         mbar_idx = list(switches["parameters"].keys()).index("M_baryon")
    #         if isinstance(galaxy.scales["M_baryon"], (float, int)) and isinstance(galaxy.scales["M_vir"], (float, int)):
    #             theta[mvir_idx] = Mvir_Moster2018(galaxy.z, theta[mbar_idx] * galaxy.scales["M_baryon"]) / galaxy.scales["M_vir"]
    #         elif galaxy.scales["M_baryon"] == "log_mass" and galaxy.scales["M_vir"] == "log_mass":
    #             theta[mvir_idx] = np.log10(
    #                 Mvir_Moster2018(galaxy.z, np.power(10, theta[mbar_idx]) * M_solar) / M_solar)

    if switches["fractions"]:
        ndim += 1
        if mcmc_hparameters["multiprocessing"]:
            indices = range(len(samples))
            samples_with_f = parmap.map(add_f_to_samples_mp, indices, galaxy, samples, mcmc_hparameters)

        else:
            samples_with_f = add_f_to_samples(samples, galaxy, mcmc_hparameters)

        results_samples = np.array(samples_with_f)

    else:
        results_samples = samples

    results_mcmc = np.append(np.percentile(results_samples, [14, 50, 86], axis=0), find_maximum_frequency(results_samples, bins=30).reshape((1,ndim)), axis=0)
    results_mcmc = [*map(lambda v: (v[3], v[1], v[2] - v[1], v[1] - v[0]), zip(*results_mcmc))]

    all_params = [x for x in switches["parameters"]] + [["f"] if switches["fractions"] == 1 else []][0]
    cols = ["MAP", "median", "errplus", "errminus"]
    results_table = pd.DataFrame(np.zeros((len(all_params), len(cols))), index=all_params, columns=cols)
    idx = 0
    for param in all_params:
        if param == "f" and switches["fractions"]:
            results_table.loc[param] = results_mcmc[-1]
        elif switches["parameters"][param]:
            results_table.loc[param] = results_mcmc[idx]
            idx += 1
        else:
            results_table.loc[param] = (galaxy.priors[param].initial, galaxy.priors[param].initial, 0, 0)

    if mcmc_hparameters["output files"]:
        filename = "%s-RotCurves_bestfitValues.csv" % galaxy.name
        if not os.path.isdir(galaxy.output_dir) or not os.path.exists(galaxy.output_dir):
            os.mkdir(galaxy.output_dir)
        results_table.to_csv("/".join([galaxy.output_dir, filename]))

    logger.info("Finished arranging data: %s minutes\n" % round((time.time() - starttime) / 60, 1))

    return results_table, sampler, samples, results_samples, nwalkers


def full_mcmc_run(galaxy, mcmc_hparameters):
    mcmc_starttime = time.time()
    results_table, sampler, samples, results_samples, nwalkers = run_mcmc(galaxy, mcmc_hparameters)
    bestfit_params = dict(results_table["median"])
    mcmc_hparameters['runtime'] = time.time() - mcmc_starttime
    walker_results_dictionary = None

    bestfit_mass_components = create_components(
        include_halo=galaxy.mass_components_switches['halo'], include_disk=galaxy.mass_components_switches['disk'],
        include_ring=galaxy.mass_components_switches['ring'], include_bulge=galaxy.mass_components_switches['bulge'],
        z=galaxy.z, halo_profile=galaxy.halo_profile, logM_vir=bestfit_params['M_vir'], c=bestfit_params['c'],
        alpha=bestfit_params['alpha'], AC=galaxy.switches['adiabatic contraction'],
        logM_baryon=bestfit_params['M_baryon'], DT=bestfit_params['DT'], disk_re=bestfit_params['Re'],
        disk_n=galaxy.disk_n, disk_q=galaxy.disk_q, disk_lw=galaxy.disk_lw, BT=bestfit_params['BT'],
        bulge_n=galaxy.bulge_n, bulge_q=galaxy.bulge_q, bulge_lw=galaxy.bulge_lw, ring_rpeak=bestfit_params['R_peak'],
        ring_FWHM=bestfit_params['ring_FWHM'], ring_lw=galaxy.ring_lw,
        running_in_cluster=mcmc_hparameters["running in cluster"], apply2D=galaxy.apply_2D)

    bestfit_RC = RotationCurveObject(galaxy=galaxy, Halo=bestfit_mass_components['halo'],
                                     Disk=bestfit_mass_components['disk'], Ring=bestfit_mass_components['ring'],
                                     Bulge=bestfit_mass_components['bulge'], sigma_dispersion=bestfit_params['sigma'],
                                     pressure_support=galaxy.pressure_support, inclination=bestfit_params['i'],
                                     sigma_beam=galaxy.sigma_beam, apply_2D=galaxy.apply_2D, include_beam_smearing=True)

    galaxy.bestfit_chisq = red_chisq(galaxy, bestfit_RC)

    if mcmc_hparameters["show plots"] == 1 or mcmc_hparameters["output files"] == 1:
        # write mcmc log
        write_mcmc_log(galaxy, results_table, bestfit_mass_components, mcmc_hparameters, sampler)

        # Plots bestfit curves & residuals
        plot_bestfit(galaxy, bestfit_RC,
                     output_plot=mcmc_hparameters["output files"])

        # Corner plot
        plot_mcmcCornerplot(results_samples, results_table, galaxy, show_plot=mcmc_hparameters["show plots"],
                            output_plot=mcmc_hparameters["output files"])

        # save 1D RC at data points and in general
        save_fit_profiles(galaxy, bestfit_RC)

        # plot walkers independently
        plot_mcmcWalkers(sampler, galaxy, mcmc_hparameters)

        # Unpack all final walker results (RC, dispersions, fractions)
        walker_results_dictionary = unpack_mcmc_walker_results(samples, nwalkers, galaxy, mcmc_hparameters)

        # Plot mcmc rotation curves
        if galaxy.fit_goals['flux']:
            plot_mcmcFluxes(walker_results_dictionary["flux"], galaxy,
                            show_plot=mcmc_hparameters["show plots"], output_plot=mcmc_hparameters["output files"])
        if galaxy.fit_goals['velocity']:
            plot_mcmcCurves(walker_results_dictionary["RC"], galaxy,
                            show_plot=mcmc_hparameters["show plots"], output_plot=mcmc_hparameters["output files"])
            plot_intrinsicRC(galaxy, bestfit_RC, output_plot=mcmc_hparameters["output files"])
        if galaxy.fit_goals['dispersion']:
            plot_mcmcDispersion(walker_results_dictionary["dispersion"],
                                galaxy, show_plot=mcmc_hparameters["show plots"], output_plot=mcmc_hparameters["output files"])

        plt.close('all')

    return results_table, walker_results_dictionary


def save_fit_profiles(galaxy, RC):
    out_df_data = pd.DataFrame(columns=['r [kpc]', 'r [arcsec]', 'v_data', 'v_data_err', 'v_model', 'disp_data', 'disp_data_err', 'disp_model'])
    out_df_intrinsic = pd.DataFrame(columns=['r [kpc]', 'mass_cum [solMass]', 'v_circ', 'v_rot', 'v_dm', 'v_baryon'])

    Rarray = galaxy.radial_space['array']

    out_df_data['r [kpc]'] = galaxy.obsdata_r
    out_df_data['r [arcsec]'] = galaxy.obsdata_r / galaxy.kpc_to_arcsec
    interpolator = CubicSpline(x=Rarray, y=RC.smeared_with_inclination)
    out_df_data['v_data'] = galaxy.obsdata_V
    out_df_data['v_data_err'] = galaxy.obsdata_V_err
    out_df_data['v_model'] = interpolator(galaxy.obsdata_r)
    interpolator = CubicSpline(x=Rarray, y=RC.velocity_dispersion)
    out_df_data['disp_data'] = galaxy.obsdata_disp
    out_df_data['disp_data_err'] = galaxy.obsdata_disp_err
    out_df_data['disp_model'] = interpolator(galaxy.obsdata_r)

    out_df_intrinsic['r [kpc]'] = Rarray
    out_df_intrinsic['v_circ'] = RC.intrinsic_no_dispersion
    out_df_intrinsic['v_rot'] = RC.intrinsic
    out_df_intrinsic['v_baryon'] = RC.Vbaryon
    if RC.halo is not None:
        out_df_intrinsic['v_dm'] = RC.Vh
    if RC.disk is not None:
        out_df_intrinsic['v_disk'] = RC.Vd
    if RC.ring is not None:
        out_df_intrinsic['v_ring'] = RC.Vr
    if RC.bulge is not None:
        out_df_intrinsic['v_bulge'] = RC.Vb

    menc = np.zeros_like(Rarray)
    if RC.halo is not None:
        menc += RC.halo.menc(Rarray)
    if RC.bulge is not None:
        menc += RC.bulge.menc(Rarray)
    if RC.disk is not None:
        menc += RC.disk.menc(Rarray)
    if RC.ring is not None:
        menc += RC.ring.menc(Rarray)
    out_df_intrinsic['mass_cum [solMass]'] = menc

    out_df_intrinsic = out_df_intrinsic[out_df_intrinsic['r [kpc]'] >= 0]

    out_df_data.to_csv('/'.join([galaxy.output_dir, 'out_1D_profs_data.csv']), index=False)
    out_df_intrinsic.to_csv('/'.join([galaxy.output_dir, 'out_1D_profs_intrinsic.csv']), index=False)


def red_chisq(galaxy, RC):
    R_array = galaxy.radial_space["array"]
    x_data = galaxy.obsdata_r

    chisq_flux = 0
    if galaxy.fit_goals['flux']:
        interpolator_flux = CubicSpline(x=R_array, y=RC.smeared_light_profile)
        flux_matched = interpolator_flux(x_data)
        chisq_flux = np.nansum(np.power((flux_matched - galaxy.obsdata_flux) / galaxy.obsdata_flux_err, 2))

    chisq_vel = 0
    if galaxy.fit_goals['velocity']:
        interpolator_vel = CubicSpline(x=R_array, y=RC.smeared_with_inclination)
        vel_matched = interpolator_vel(x_data)
        chisq_vel = np.nansum(np.power((vel_matched - galaxy.obsdata_V) / galaxy.obsdata_V_err, 2))

    chisq_disp = 0
    if galaxy.fit_goals['dispersion']:
        interpolator_disp = CubicSpline(x=R_array, y=RC.velocity_dispersion)
        disp_matched = interpolator_disp(x_data)
        chisq_disp = np.nansum(np.power((disp_matched - galaxy.obsdata_disp) / galaxy.obsdata_disp_err, 2))

    chisq_total = np.nansum([chisq_flux, chisq_vel, chisq_disp])

    # reduce by dof
    chisq_flux /= galaxy.dof
    chisq_vel /= galaxy.dof
    chisq_disp /= galaxy.dof
    chisq_total /= galaxy.dof

    chisq_dict = {
        'total': chisq_total,
        'flux': chisq_flux,
        'velocity': chisq_vel,
        'disp': chisq_disp,
    }

    return chisq_dict


def write_mcmc_log(galaxy, results_table=None, bestfit_mass_components=None, mcmc_hparameters=None, sampler=None,):
    sep = '##########################'
    minisep = '-----------------'
    num_spaces = 10

    log_path = '/'.join([galaxy.output_dir, 'mcmc_log.txt'])
    f = open(log_path, 'w')

    len_to_fill = int((34 - 2 - len(galaxy.name))/2)
    intro = '''----------------------------------
--- RotCurves MCMC fit results ---
%s %s %s
----------------------------------\n
output folder: %s
data folder: %s
Date: %s
runtime: %s minutes
\n''' % ('-'*len_to_fill, galaxy.name, '-'*len_to_fill, galaxy.output_dir, galaxy.obsdata_dir,
         datetime.datetime.now(), np.round(mcmc_hparameters['runtime'] / 60, 0))
    f.write(intro)

    opening = '''MCMC fitting goals
%s
    fit flux: %01d
    fit velocity: %01d
    fit dispersion: %01d
num walkers: %3.0f
num burnins: %3.0f
num iterations: %3.0f
    converged after: %3.0f
''' % (sep, galaxy.fit_goals['flux'], galaxy.fit_goals['velocity'], galaxy.fit_goals['dispersion'],
       mcmc_hparameters['nwalkers'], mcmc_hparameters['niter'][0], mcmc_hparameters['niter'][1],
       mcmc_hparameters['niters_converged'])
    f.write(opening)

    # Write bestfit params
    f.write('\nBestfit params\n%s' % sep)
    for param, values in results_table.iterrows():
        if (param == 'f') or (galaxy.switches['parameters'][param]):
            s = '\n    %s:%s %2.2f (+%2.2f -%2.2f) [MAP: %2.2f]' % (param, ' '*(num_spaces - len(param)), values['median'], values['errplus'], values['errminus'], values['MAP'])
            f.write(s)

    # Write prior information
    priors = '''\n
Prior information:\n%s''' % sep
    f.write(priors)
    for param, values in results_table.iterrows():
        if not param == 'f':
            prior = galaxy.priors[param]
            if galaxy.switches['parameters'][param]:
                if prior.type == 'gaussian':
                    s = '\n    %s:%s Gaussian (mu=%2.2f, sig=%2.2f)' % (param, ' '*(num_spaces - len(param)), prior.initial, prior.sig)
                elif prior.type == 'uniform':
                    s = '\n    %s:%s Uniform (%2.2f, %2.2f)' % (param, ' '*(num_spaces - len(param)), prior.min, prior.max)
            else:
                s = '\n    %s:%s FIXED (%2.2f)' % (param, ' '*(num_spaces - len(param)), prior.initial)
            f.write(s)

    # Write model components
    model_components = '\n\nModel Components\n'
    f.write(model_components)

    component = 'disk'
    if galaxy.mass_components[component] is not None:
        M_disk = np.log10(bestfit_mass_components[component].mass)
        Reff = bestfit_mass_components[component].r_eff
        disk_n = bestfit_mass_components[component].n
        disk_q = bestfit_mass_components[component].q0
        disk_mass_to_light = bestfit_mass_components[component].mass_to_light
        s_comp = '\n%s:\n' \
                 '    logmass:%s %2.2f   [solmass]\n' \
                 '    reff:%s %2.2f    [kpc]\n' \
                 '    n_sersic:%s %2.1f     []\n' \
                 '    q0:%s %2.2f    []\n' \
                 '    M/L:%s %s\n' % (component,
                                      ' '*(num_spaces - len('logmass')), M_disk,
                                      ' '*(num_spaces - len('reff')), Reff,
                                      ' '*(num_spaces - len('n_sersic')), disk_n,
                                      ' '*(num_spaces - len('invq')), disk_q,
                                      ' '*(num_spaces - len('light')), disk_mass_to_light)
        f.write(s_comp)

    component = 'ring'
    if galaxy.mass_components[component] is not None:
        M_ring = np.log10(bestfit_mass_components[component].mass)
        Rpeak = bestfit_mass_components[component].r_s
        FWHM_ring = bestfit_mass_components[component].FWHM_ring
        ring_mass_to_light = bestfit_mass_components[component].mass_to_light
        s_comp = '%s\n%s:\n' \
                 '    logmass:%s %2.2f   [solmass]\n' \
                 '    r_peak:%s %2.2f    [kpc]\n' \
                 '    ring_FWHM:%s %2.1f     []\n' \
                 '    light:%s %s\n' % (minisep, component,
                                      ' '*(num_spaces - len('logmass')), M_ring,
                                      ' '*(num_spaces - len('r_peak')), Rpeak,
                                      ' '*(num_spaces - len('ring_FWHM')), FWHM_ring,
                                      ' '*(num_spaces - len('light')), ring_mass_to_light)
        f.write(s_comp)

    component = 'bulge'
    if galaxy.mass_components[component] is not None:
        M_bulge = np.log10(bestfit_mass_components[component].mass)
        bulge_Reff = bestfit_mass_components[component].r_eff
        bulge_n = bestfit_mass_components[component].n
        bulge_q = bestfit_mass_components[component].q0
        bulge_mass_to_light = bestfit_mass_components[component].mass_to_light
        s_comp = '%s\n%s:\n' \
                 '    logmass:%s %2.2f   [solmass]\n' \
                 '    reff:%s %2.2f    [kpc]\n' \
                 '    n_sersic:%s %2.1f     []\n' \
                 '    q0:%s %2.2f    []\n' \
                 '    M/L:%s %s\n' % (minisep, component,
                                                   ' '*(num_spaces - len('logmass')), M_bulge,
                                                   ' '*(num_spaces - len('reff')), bulge_Reff,
                                                   ' '*(num_spaces - len('n_sersic')), bulge_n,
                                                   ' '*(num_spaces - len('invq')), bulge_q,
                                                   ' '*(num_spaces - len('light')), bulge_mass_to_light)
        f.write(s_comp)

    component = 'halo'
    if galaxy.mass_components[component] is not None:
        f_dm = results_table.loc['f']['median']
        M_vir = np.log10(bestfit_mass_components[component].mass)
        c = bestfit_mass_components[component].c
        alpha = bestfit_mass_components[component].alpha
        s_comp = '%s\n%s:\n' \
                 '    f_dm:%s %2.2f    []\n' \
                 '    logmass:%s %2.2f   [solmass]\n' \
                 '    conc:%s %2.2f    []\n' \
                 '    alpha:%s %2.2f    []\n' % (minisep, component,
                                                 ' '*(num_spaces - len('f_dm')), f_dm,
                                                 ' '*(num_spaces - len('logmass')), M_vir,
                                                 ' '*(num_spaces - len('conc')), c,
                                                 ' '*(num_spaces - len('alpha')), alpha)
        f.write(s_comp)

    component = 'dispersion'
    sigma = results_table.loc['sigma']['MAP']
    pressure_support_type = galaxy.pressure_support
    s_comp = '%s\n%s:\n' \
             '    sigma0:%s %2.2f   [km/s]\n' \
             '    ps. type:%s %s\n' % (minisep, component,
                                       ' '*(num_spaces - len('sigma0')), sigma,
                                       ' '*(num_spaces - len('ps. type')), pressure_support_type)
    f.write(s_comp)

    geometry = '''
Gemoetry
%s
    inc:   %2.2f
    beam FWHM:   %2.2f" (%2.2f kpc )''' % (minisep, results_table.loc['i']['MAP'], galaxy.beam_FWHM / galaxy.kpc_to_arcsec, galaxy.beam_FWHM)
    f.write(geometry)

    additionals = '''\n
Additional knobs
%s
    Adiabatic contraction:  %1.0f''' % (minisep, galaxy.switches['adiabatic contraction'])
    f.write(additionals)

    mcmc_fitting_assesment = '''\n
mcmc information
%s 
degrees of freedom: %2.2f
red_chisq:          %2.2f
    red_chisq_flux: %2.2f
    red_chisq_vel:  %2.2f
    red_chisq_disp: %2.2f
mean acceptance ratio:    %2.2f (+/- %2.2f)
stretch move a:    %2.2f    # sampler performs a move with size sampled uniformly from [1/a, a], with size 1/sqrt(a) 
mean auto-correlation time (tau):    %2.2f
''' % (sep, galaxy.dof, galaxy.bestfit_chisq['total'], galaxy.bestfit_chisq['flux'],
       galaxy.bestfit_chisq['velocity'], galaxy.bestfit_chisq['disp'],
       np.mean(sampler.acceptance_fraction),
       np.std(sampler.acceptance_fraction),
       mcmc_hparameters['strecth_move_a'],
       np.mean(emcee.autocorr.integrated_time(sampler.chain, tol=0)))
    f.write(mcmc_fitting_assesment)

    # Write taus for every parameter
    taus = ''
    i = 0
    for param in galaxy.switches['parameters']:
        if galaxy.switches['parameters'][param]:
            tau = '    tau %s:%s%2.2f\n' % (param, ' '*(12-len(param)), emcee.autocorr.integrated_time(sampler.chain, tol=0)[i])
            taus += tau
            i += 1
    f.write(taus)

    f.close()


"""
Functions to plot and analyze mcmc results
"""

def cornerplot_labels(switches, add_f=True):
    labels = []
    for switch in switches["parameters"]:
        if switches["parameters"][switch] == 1:
            if switch == "Re":
                labels.append("$R_{e}$ $[kpc]$")
            elif switch == "M_baryon":
                labels.append(r"$log M_{baryon}$")
            elif switch == "M_vir":
                if switches["use Moster"]:
                    labels.append(r"$\log {M_{vir}^{SMHM}}$")
                else:
                    labels.append(r"$log M_{vir}$")
            elif switch == "R_peak":
                labels.append("$R_{peak}$")
            elif switch == "ring_FWHM":
                labels.append("$FWHM_{ring}$")
            elif switch == "BT":
                labels.append("$B/T$")
            elif switch == "DT":
                labels.append("$D/T$")
            elif switch == "sigma":
                labels.append(r"$\sigma$ $[km/s]$")
            elif switch == "c":
                labels.append("$c$")
            elif switch == "alpha":
                labels.append(r"$\alpha$")
            elif switch == "i":
                labels.append(r"$i [\degree]$")

    if add_f:
        labels.append("$f$")

    return labels


def plot_mcmcWalkers(sampler, galaxy, mcmc_hparameters, show_plot=False, output_plot=True):
    switches = galaxy.switches
    ndim = int(np.sum([x for x in switches["parameters"].values()]))

    fig, axes = figure(ncols=1, nrows=ndim, axwidth=10, axheight=3.5)
    samples_walkers = sampler.get_chain()
    labels = cornerplot_labels(switches, add_f=galaxy.switches['fractions'])
    niters_converged = mcmc_hparameters['niters_converged']

    x = np.arange(1, niters_converged + 1)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(x, samples_walkers[:, :, i], alpha=0.5, lw=0.5, color=colors['grey'])
        ax.set_xlim(1, niters_converged)
        ax.xaxis.set_major_locator(MultipleLocator(np.ceil(niters_converged / 4)))
        ax.xaxis.set_minor_locator(MultipleLocator(max(1, np.ceil(np.ceil(niters_converged / 4) / 4))))
        ax.set_ylabel(labels[i], fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        # ax.set_title(r"$\tau$=%s" % np.round(tau[i], 0))
    axes[-1].set_xlabel("step number")

    if output_plot:
        filename = "%s-RotCurves_mcmcWalkers.jpg" % galaxy.name
        if not os.path.isdir(galaxy.output_dir) or not os.path.exists(galaxy.output_dir):
            os.mkdir(galaxy.output_dir)
        plt.savefig("/".join([galaxy.output_dir, filename]), format='jpg', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_mcmcCornerplot(samples_with_f, results_table, galaxy, show_plot=False, output_plot=True):
    starttime = time.time_ns()
    switches = galaxy.switches
    on_switches = [x for x in switches["parameters"] if switches["parameters"][x] == 1]
    labels = cornerplot_labels(switches, galaxy.switches['fractions'])

    # # show true values
    # truth = [galaxy.true_values[x] for x in on_switches]
    # if galaxy.switches['fractions']:
    #     truth.append(galaxy.true_values['f'])

    # bestfit_params_medians = dict(results_table['median'])
    # bestfit_params_map = dict(results_table['MAP'])

    bestfit_medians = [results_table['median'].loc[x] for x in on_switches]
    bestfit_maps = [results_table['MAP'].loc[x] for x in on_switches]

    # medians = [bestfit_params_medians[x] for x in bestfit_params_medians if x in on_switches]
    if galaxy.switches['fractions']:
        bestfit_medians.append(results_table['median'].loc['f'])
        bestfit_maps.append(results_table['MAP'].loc['f'])

    true_values = [galaxy.true_values[x] for x in on_switches+["f"]]
    fig = corner.corner(np.array(samples_with_f), labels=labels, label_kwargs={'fontsize': 16},
                        bins=15, quantiles=(0.16, 0.5, 0.84),
                        smooth=3,
                        show_titles=True, title_kwargs={'fontsize': 14},
                        truths=true_values, truth_color=colors['pink'], plot_contours=True)

    for ax in fig.axes:
        for i, param in enumerate(on_switches):
            label = labels[i]
            if label in str(ax.title):
                # plot prior prob
                if galaxy.priors[param].type == 'gaussian':
                    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num=100)
                    y = np.exp(galaxy.priors[param].lnprob(x)) * (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.9
                    ax.plot(x, y, color=colors['pink'], lw=1.5, ls=':')
                elif galaxy.priors[param].type == 'uniform':
                    y = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.2
                    ax.axhline(y, color=colors['pink'], lw=1.5, ls=':')

                # plot medians
                ax.axvline(bestfit_medians[i], color=colors['red'], ls='-', lw=2.5)

                # plot MAP
                ax.axvline(bestfit_maps[i], color=colors['green'], ls='-', lw=2.5)

    if output_plot:
        filename = "%s-RotCurves_cornerPlot.jpg" % galaxy.name
        if not os.path.isdir(galaxy.output_dir) or not os.path.exists(galaxy.output_dir):
            os.mkdir(galaxy.output_dir)
        plt.savefig("/".join([galaxy.output_dir, filename]), format='jpg', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close('all')

    runtime = time.time_ns() - starttime
    logger.info('Corner plot runtime: %s minutes' % np.round(runtime * 1e-9 / 60, 1))

    del samples_with_f

def plot_single_bestfit(rawdata_x, rawdata_y, rawdata_yerr, model_x, model_y, ax_values, ax_res, kpc_to_arcsec=None):
    color_data = 'black'
    color_model = colors['red']

    # plot data and model
    ax_values.errorbar(x=rawdata_x, y=rawdata_y, yerr=rawdata_yerr,
                       ms=5, color=color_data, fmt='s', capsize=2., capthick=1., label='data')

    # interpolate model to data points
    interpolator = CubicSpline(x=model_x, y=model_y)
    y_bestfit = interpolator(rawdata_x)

    # plot model
    ax_values.scatter(rawdata_x, y_bestfit,
                      color=color_model, marker='s', s=40, label='model')
    ax_values.plot(model_x, model_y, color=color_model, lw=1.5)

    # plot residuals
    ax_res.scatter(x=rawdata_x, y=y_bestfit - rawdata_y,
                   s=40, color=color_model, marker='s')
    ax_res.errorbar(x=rawdata_x, y=np.zeros_like(rawdata_x), yerr=rawdata_yerr,
                    ms=0.1, color=color_data, fmt='s', capsize=2., capthick=2.)
    ax_res.axhline(y=0, color=color_data, ls='-', lw=1.)

    # set axes limits
    ax_values.set_xlim([np.min(rawdata_x)*1.2, np.max(rawdata_x)*1.2])
    ax_res.set_xlim([np.min(rawdata_x)*1.2, np.max(rawdata_x)*1.2])

    # set axes labels
    ax_res.set_xlabel(r'$R$ [kpc]', fontsize=16)
    if ax_res.get_xlim()[1] > 6:
        ax_res.xaxis.set_major_locator(MultipleLocator(5))
        ax_res.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        ax_res.xaxis.set_major_locator(MultipleLocator(2))
        ax_res.xaxis.set_minor_locator(MultipleLocator(0.5))

    if kpc_to_arcsec is not None:
        ax_values_twin = ax_values.twiny()
        ax_values_twin.plot(rawdata_x/kpc_to_arcsec, np.zeros_like(rawdata_x), lw=0.01, ls=':', color=colors['grey'])
        ax_values_twin.set_xlabel(r'$R$ ["]', fontsize=14)
        ax_values_twin.tick_params(axis='x', labelsize=12)
        if ax_values_twin.get_xlim()[1] > 0.6:
            ax_values_twin.xaxis.set_major_locator(MultipleLocator(0.5))
            ax_values_twin.xaxis.set_minor_locator(MultipleLocator(0.1))
        else:
            ax_values_twin.xaxis.set_major_locator(MultipleLocator(0.2))
            ax_values_twin.xaxis.set_minor_locator(MultipleLocator(0.05))


def plot_bestfit(galaxy, RC, output_plot=True):
    starttime = time.time_ns()

    ncols = np.sum(list(galaxy.fit_goals.values()))
    fig, axes = figure(nrows=2, ncols=ncols, axwidth=3.5, padx=0.2, pady=0, height_ratios=[1, 0.4])
    axes = axes.flatten()
    i = 0

    R_array = galaxy.radial_space["array"]
    for fit_goal in galaxy.fit_goals:
        if galaxy.fit_goals[fit_goal]:

            if ncols == 2 and i == 1:
                for ax in [axes[i], axes[i+ncols]]:
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                    ax.yaxis.set_ticks_position('both')

            if ncols == 3:
                if i == 2:
                    for ax in [axes[i], axes[i + ncols]]:
                        ax.yaxis.set_label_position("right")
                        ax.yaxis.tick_right()
                        ax.yaxis.set_ticks_position('both')
                elif i == 1:
                    for ax in [axes[i], axes[i + ncols]]:
                        ax.set_tick_params(labelleft=False, labelright=False)

            if fit_goal == 'flux':
                plot_single_bestfit(rawdata_x=galaxy.obsdata_r, rawdata_y=galaxy.obsdata_flux,
                                    rawdata_yerr=galaxy.obsdata_flux_err,
                                    model_x=R_array, model_y=RC.smeared_light_profile,
                                    ax_values=axes[i], ax_res=axes[i+ncols],
                                    kpc_to_arcsec=galaxy.kpc_to_arcsec)
                axes[i].set_ylabel(r'$flux$ [arb.]', fontsize=16)
                axes[i+ncols].set_ylabel(r'$flux$ res. [arb.]', fontsize=16)

            elif fit_goal == 'velocity':
                plot_single_bestfit(rawdata_x=galaxy.obsdata_r, rawdata_y=galaxy.obsdata_V,
                                    rawdata_yerr=galaxy.obsdata_V_err,
                                    model_x=R_array, model_y=RC.smeared_with_inclination,
                                    ax_values=axes[i], ax_res=axes[i + ncols],
                                    kpc_to_arcsec=galaxy.kpc_to_arcsec)
                axes[i].set_ylabel(r'$V_{rot}$ [km/s]', fontsize=16)
                axes[i + ncols].set_ylabel(r'$V_{rot}$ res. [km/s]', fontsize=16)

                yedge = np.max(np.abs(axes[i].get_ylim()))
                axes[i].set_ylim([-yedge, yedge])

            elif fit_goal == 'dispersion':
                plot_single_bestfit(rawdata_x=galaxy.obsdata_r, rawdata_y=galaxy.obsdata_disp,
                                    rawdata_yerr=galaxy.obsdata_disp_err,
                                    model_x=R_array, model_y=RC.velocity_dispersion,
                                    ax_values=axes[i], ax_res=axes[i + ncols],
                                    kpc_to_arcsec=galaxy.kpc_to_arcsec)
                axes[i].set_ylabel(r'$\sigma\ [km/s]$', fontsize=16)
                axes[i + ncols].set_ylabel(r'$\sigma$ res. [km/s]', fontsize=16)

            # add legend
            if i == 0:
                axes[i].legend(loc='upper left')

            # add PSF ellipse
            if i == 0:
                x_scale = axes[i].get_xlim()[1] - axes[i].get_xlim()[0]
                y_scale = axes[i].get_ylim()[1] - axes[i].get_ylim()[0]
                beam_FWHM_in_plot_size = galaxy.beam_FWHM / (axes[i].get_xlim()[1] - axes[i].get_xlim()[0])
                beam_FWHM_x = galaxy.beam_FWHM
                beam_FWHM_y = beam_FWHM_in_plot_size * y_scale
                ellipse = mpl_patches.Ellipse(xy=(axes[i].get_xlim()[1]*0.7, axes[i].get_ylim()[0]*0.7), width=beam_FWHM_x, height=beam_FWHM_y,
                                              edgecolor=colors['grey'], fc=colors['grey'], alpha=0.8, lw=0.5)
                axes[i].add_patch(ellipse)

            # add zero line
            axes[i].axhline(y=0, ls=':', lw=1., color=colors['grey'])

            # symmetrize residuals plots
            yedge = np.max(np.abs(axes[i+ncols].get_ylim()))
            axes[i+ncols].set_ylim([-yedge, yedge])

            i += 1

    if output_plot:
        filename = "%s-RotCurves_bestfit.jpg" % galaxy.name
        if not os.path.isdir(galaxy.output_dir) or not os.path.exists(galaxy.output_dir):
            os.mkdir(galaxy.output_dir)
        plt.savefig("/".join([galaxy.output_dir, filename]))
    # plt.close()

    runtime = time.time_ns() - starttime
    logger.info('Plot bestfits runtime: %s minutes' % np.round(runtime * 1e-9 / 60, 1))


def plot_intrinsicRC(galaxy=None, RC=None, R_array=None, output_plot=True):
    starttime = time.time_ns()

    blue = (0/255, 102/255, 204/255, 1)
    black = (108/255, 108/255, 108/255, 1)
    orange = (255/255, 156/255, 69/255, 0.75)
    green = (28/255, 148/255, 108/255, 0.85)
    green_light = (28/255, 148/255, 108/255, 0.5)
    red = (218/255, 51/255, 51/255, 0.85)

    if R_array is None:
        R_array = galaxy.radial_space["array"]

    fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(R_array, RC.smeared_with_inclination, "-", lw=2, color=red, label="$V_{obs}$")
    ax.plot(R_array, RC.intrinsic, "-", lw=2, color=red, label="$V_{rot}$")
    ax.plot(R_array, RC.intrinsic_no_dispersion, "-", lw=2, color=blue, label="$V_{circ}$")
    ax.plot(R_array, RC.Vh, "-", lw=2, color=black, label="$V_{DM}$")
    ax.plot(R_array, RC.Vbaryon, "-", lw=2, color=green, label="$V_{baryons}$")
    if np.sum(RC.V2b) > 0:
        ax.plot(R_array, RC.Vb, ":", lw=2, color=green_light, label="$V_{bulge}$")
    if np.sum(RC.V2d) > 0:
        ax.plot(R_array, RC.Vd, "--", lw=2, color=green_light, label="$V_{disk}$")
    if np.sum(RC.V2r) > 0:
        ax.plot(R_array, RC.Vr, "-.", lw=2, color=green_light, label="$V_{ring}$")

    ax.axhline(y=0, color=colors['grey'], lw=1)
    ax.legend(loc='upper right')
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("V [km/s]")
    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([-20, ax.get_ylim()[1]])
    # ax.set_title("%s - Intrinsic rotation curve" % galaxy.name)

    if output_plot:
        filename = "%s-RotCurves_intrinsicRC.jpg" % galaxy.name
        if not os.path.isdir(galaxy.output_dir) or not os.path.exists(galaxy.output_dir):
            os.mkdir(galaxy.output_dir)
        plt.savefig("/".join([galaxy.output_dir, filename]))

    plt.close()

    runtime = time.time_ns() - starttime
    logger.info('Plot Intrinsic RC runtime: %s minutes' % np.round(runtime * 1e-9 / 60, 1))


def plot_mcmcFluxes(mcmc_fluxes, galaxy, show_plot=False, output_plot=True):
    starttime = time.time_ns()

    R = galaxy.radial_space["array"]
    switches = galaxy.switches

    fig, ax = plt.subplots()
    for mcmc_flux in mcmc_fluxes:
        ax.plot(R, mcmc_flux, color="g", alpha=0.1)
    ax.errorbar(galaxy.obsdata_r, galaxy.obsdata_flux, galaxy.obsdata_flux_err, color='k', fmt=".", label="data")

    # ax.set_title("%s - Rotation Curves\n"
    #              "Free parameters: %s" % (galaxy.name, [x for x in switches["parameters"] if switches["parameters"][x] == 1]))
    buffer = 0.2
    ax.set_ylim(np.minimum(np.min(galaxy.obsdata_flux), np.min(mcmc_fluxes)) * (1 + buffer), np.maximum(np.max(galaxy.obsdata_flux), np.max(mcmc_fluxes)) * (1 + buffer))
    ax.set_xlim(np.min(galaxy.obsdata_r) * (1 + buffer), np.max(galaxy.obsdata_r) * (1 + buffer))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1.))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.set_xlabel("R [kpc]", fontsize=10)
    ax.set_ylabel("flux [arb.]", fontsize=10)
    ax.legend(loc=2)

    if output_plot:
        filename = "%s-RotCurves_mcmcFluxes.jpg" % galaxy.name
        if not os.path.isdir(galaxy.output_dir) or not os.path.exists(galaxy.output_dir):
            os.mkdir(galaxy.output_dir)
        plt.savefig("/".join([galaxy.output_dir, filename]))

    if show_plot:
        plt.show()
    else:
        plt.close()

    runtime = time.time_ns() - starttime
    logger.info('Plot mcmc fluxes runtime: %s minutes' % np.round(runtime * 1e-9 / 60, 1))


def plot_mcmcCurves(mcmc_rotation_curves, galaxy, show_plot=False, output_plot=True):
    starttime = time.time_ns()

    R = galaxy.radial_space["array"]
    switches = galaxy.switches

    fig, ax = plt.subplots()
    for mcmc_curve in mcmc_rotation_curves:
        ax.plot(R, mcmc_curve, color="g", alpha=0.1)
    ax.errorbar(galaxy.obsdata_r, galaxy.obsdata_V, galaxy.obsdata_V_err, color='k', fmt=".", label="data")

    # ax.set_title("%s - Rotation Curves\n"
    #              "Free parameters: %s" % (galaxy.name, [x for x in switches["parameters"] if switches["parameters"][x] == 1]))
    buffer = 0.2
    ax.set_ylim(np.minimum(np.min(galaxy.obsdata_V), np.min(mcmc_rotation_curves)) * (1 + buffer), np.maximum(np.max(galaxy.obsdata_V), np.max(mcmc_rotation_curves)) * (1 + buffer))
    ax.set_xlim(np.min(galaxy.obsdata_r) * (1 + buffer), np.max(galaxy.obsdata_r) * (1 + buffer))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlabel("R [kpc]", fontsize=10)
    ax.set_ylabel("V [km/s]", fontsize=10)
    ax.legend(loc=2)

    # if galaxy.name in ["COS4_01351", "D3a_6397", "D3a_15504", "GS4_43501"]:
    #     G17_fit = pd.read_csv(r"C:\Users\Amit\Dropbox\Amit research\Rotation Curve - mcmc\Genzel Data\original plots\%s_G17fit.csv" % galaxy.name)
    #     plt.plot(G17_fit["R"], G17_fit["V"], "r")

    if output_plot:
        filename = "%s-RotCurves_mcmcRCs.jpg" % galaxy.name
        if not os.path.isdir(galaxy.output_dir) or not os.path.exists(galaxy.output_dir):
            os.mkdir(galaxy.output_dir)
        plt.savefig("/".join([galaxy.output_dir, filename]))

    if show_plot:
        plt.show()
    else:
        plt.close()

    runtime = time.time_ns() - starttime
    logger.info('Plot mcmc curves runtime: %s minutes' % np.round(runtime * 1e-9 / 60, 1))


def plot_mcmcDispersion(mcmc_dispersion, galaxy, show_plot=False, output_plot=True):
    starttime = time.time_ns()

    switches = galaxy.switches

    fig, ax = plt.subplots()
    for dispersion in mcmc_dispersion:
        ax.plot(galaxy.radial_space["array"], dispersion, color="g", alpha=0.1)
    ax.errorbar(galaxy.obsdata_r, galaxy.obsdata_disp, galaxy.obsdata_disp_err, color='k', fmt=".", label="data")

    # ax.set_title("%s - Velocity Dispersion\n"
    #              "Free parameters: %s" % (galaxy.name, [x for x in switches["parameters"] if switches["parameters"][x] == 1]))
    buffer = 0.2
    ax.set_ylim(0, np.maximum(np.nanmax(mcmc_dispersion), np.nanmax(galaxy.obsdata_disp)) * (1 + buffer))
    ax.set_xlim(np.nanmin(galaxy.obsdata_r) * (1 + buffer), np.nanmax(galaxy.obsdata_r) * (1 + buffer))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlabel("R [kpc]", fontsize=10)
    ax.set_ylabel(r"$\sigma$ $[km/s]$", fontsize=10)
    ax.legend(loc=2)

    # if galaxy.name in ["COS4_01351", "D3a_6397", "D3a_15504", "GS4_43501"]:
    #     G17_disp_fit = pd.read_csv(r"C:\Users\Amit\Dropbox\Amit research\Rotation Curve - mcmc\Genzel Data\%s G17 disp fit.csv" % galaxy.name)
    #     plt.plot(G17_disp_fit["R"], G17_disp_fit["disp"], "r")

    if output_plot:
        filename = "%s-RotCurves_mcmcDisp.jpg" % galaxy.name
        if not os.path.isdir(galaxy.output_dir) or not os.path.exists(galaxy.output_dir):
            os.mkdir(galaxy.output_dir)
        plt.savefig("/".join([galaxy.output_dir, filename]))

    if show_plot:
        plt.show()
    else:
        plt.close()

    runtime = time.time_ns() - starttime
    print('Plot dispersion curves runtime: %s minutes' % np.round(runtime * 1e-9 / 60, 1))