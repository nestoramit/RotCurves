import os
import logging
import pandas as pd
import numpy as np
from astropy.cosmology import Planck18
import datetime

from mcmc_fitter import Prior
from base_utils import create_r_space
from rotation_curve import calculate_fraction_at_re
from mass_model import create_components

# Define the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RotCurves')

class GalaxyObject:
    def __init__(self, name, metadata_table_path=None, obsdata=None, galaxy_outputs_folder=None, obsdata_dir=None, running_in_cluster=False):
        if galaxy_outputs_folder is None:
            galaxy_outputs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_output_folder')

        if not os.path.exists(galaxy_outputs_folder):
            os.mkdir(galaxy_outputs_folder)

        self.name = name
        metadata = pd.read_excel(metadata_table_path, index_col="uniqID")
        self.metadata = metadata.loc[self.name]
        self.running_in_cluster = running_in_cluster

        # if obsdata_dir is None:
        #     if not running_in_cluster:
        #         self.obsdata_dir = os.path.join(main_path, 'MPE', 'RC_raw_data')
        #     else:
        #         self.obsdata_dir = "/mnt/sdceph/users/ycohen/Nestor/inputs/RC_raw_data"
        # else:
        self.obsdata_dir = obsdata_dir

        if self.name.find('-') < 0:
            self.rawdata_path = os.path.join(self.obsdata_dir, self.name + "_flux.obs_prof.txt")
        else:
            self.rawdata_path = os.path.join(self.obsdata_dir, self.name[:self.name.find('-')] + "_flux.obs_prof.txt")

        self.mass_components_switches = {'halo': self.metadata['mass_components_halo'],
                                        'disk': self.metadata['mass_components_disk'],
                                        'ring': self.metadata['mass_components_ring'],
                                        'bulge': self.metadata['mass_components_bulge']}

        ## Disk properties
        self.disk_n = float(self.metadata["disk_sersic_n"])
        self.disk_invq = float(np.round(self.metadata["disk_invq"], 2))
        self.disk_q = float(np.round(1 / self.metadata["disk_invq"], 2))
        self.disk_lw = int(self.metadata['disk_lw'])

        ## Bulge properties
        self.bulge_lw = int(self.metadata['bulge_lw'])
        self.bulge_n = float(self.metadata['bulge_sersic_n'])
        self.bulge_q = float(np.round(1 / self.metadata['bulge_invq'], 2))

        ## Ring properties
        self.ring_lw = int(self.metadata['ring_lw'])

        ## Halo properties
        self.z = float(np.round(self.metadata["redshift"], 2))
        self.halo_profile = self.metadata["halo_profile"]
        self.kpc_to_arcsec = 1/Planck18.arcsec_per_kpc_proper(self.z).value

        ## Instrument
        self.sigma_inst = self.metadata['sigma_inst']
        self.beam_FWHM = self.metadata['beam_FWHM ["]'] * self.kpc_to_arcsec
        self.sigma_beam = self.beam_FWHM / (2*np.sqrt(2*np.log(2)))
        self.apply_2D = self.metadata['apply 2D']

        ## Priors
        self.Mstellar = None
        if 'Mstellar' in self.metadata.keys():
            self.Mstellar = self.metadata["Mstellar"]
        self.pressure_support = self.metadata["pressure_support_formula"]
        if 'Mvir_Moster' in self.metadata.keys():
            self.Mvir_moster = self.metadata["Mvir_Moster"]
        elif self.Mstellar is not None:
            self.Mvir_moster = self.Mvir_Moster2018(self.z, self.Mstellar)

        self.priors = {
            "Re": Prior(self.metadata["Re_type"], np.round(self.metadata["Re [kpc]"], 2), self.metadata["Re_min"],
                        self.metadata["Re_max"], self.metadata["Re_sig"]),
            "M_baryon": Prior(self.metadata["Mbaryon_type"], self.metadata["Mbaryon"], self.metadata["Mbaryon_min"],
                              self.metadata["Mbaryon_max"], self.metadata["Mbaryon_sig"]),
            "M_vir": Prior(self.metadata["Mvir_type"], self.metadata["Mvir"], self.metadata["Mvir_min"],
                           self.metadata["Mvir_max"], self.metadata["Mvir_sig"]),
            "BT": Prior(self.metadata["BT_type"], self.metadata["BT"], self.metadata["BT_min"],
                        self.metadata["BT_max"], self.metadata["BT_sig"]),
            "DT": Prior(self.metadata["DT_type"], self.metadata["DT"], self.metadata["DT_min"],
                        self.metadata["DT_max"], self.metadata["DT_sig"]),
            "R_peak": Prior(self.metadata["R_peak_type"], self.metadata["R_peak"], self.metadata["R_peak_min"],
                        self.metadata["R_peak_max"], self.metadata["R_peak_sig"]),
            "ring_FWHM": Prior(self.metadata["FWHM_ring_type"], self.metadata["FWHM_ring"], self.metadata["FWHM_ring_min"],
                        self.metadata["FWHM_ring_max"], self.metadata["FWHM_ring_sig"]),
            "sigma": Prior(self.metadata["sigma_type"], self.metadata["sigma"], self.metadata["sigma_min"],
                           self.metadata["sigma_max"], self.metadata["sigma_sig"]),
            "c": Prior(self.metadata["c_type"], self.metadata["c"], self.metadata["c_min"],
                       self.metadata["c_max"], self.metadata["c_sig"]),
            "alpha": Prior(self.metadata["alpha_type"], self.metadata["alpha"], self.metadata["alpha_min"],
                           self.metadata["alpha_max"], self.metadata["alpha_sig"]),
            "i": Prior(self.metadata["inc_type"], self.metadata["inc"], self.metadata["inc_min"],
                       self.metadata["inc_max"], self.metadata["inc_sig"])}

        self.switches = {"parameters": {"Re": self.metadata["Re_switch"],
                                        "M_baryon": self.metadata["Mbaryon_switch"],
                                        "M_vir": self.metadata["Mvir_switch"],
                                        "BT": self.metadata["BT_switch"],
                                        "DT": self.metadata["DT_switch"],
                                        "R_peak": self.metadata["Rpeak_switch"],
                                        "ring_FWHM": self.metadata["FWHMring_switch"],
                                        "sigma": self.metadata["sigma_switch"],
                                        "c": self.metadata["c_switch"],
                                        "alpha": self.metadata["alpha_switch"],
                                        "i": self.metadata["inc_switch"]},
                         "fractions": self.metadata["f_switch"],
                         "use Moster": self.metadata["use Moster"],
                         "adiabatic contraction": self.metadata["AC_switch"]}

        self.true_values = {"Re": self.metadata['Re_true'],
                            "M_baryon": self.metadata['Mbaryon_true'],
                            "M_vir": self.metadata['Mvir_true'],
                            "BT": self.metadata['BT_true'],
                            "DT": self.metadata['DT_true'],
                            "R_peak": self.metadata['R_peak_true'],
                            "ring_FWHM": self.metadata['FWHM_ring_true'],
                            "sigma": self.metadata['sigma_true'],
                            "i": self.metadata['inc_true'],
                            "c": self.metadata['c_true'],
                            "alpha": self.metadata['alpha_true'],
                            "f": None}

        self.mass_components = create_components(include_halo=self.mass_components_switches['halo'],
                                                 include_disk=self.mass_components_switches['disk'],
                                                 include_ring=self.mass_components_switches['ring'],
                                                 include_bulge=self.mass_components_switches['bulge'], z=self.z,
                                                 halo_profile=self.halo_profile, logM_vir=self.true_values['M_vir'],
                                                 c=self.true_values['c'], alpha=self.true_values['alpha'],
                                                 AC=self.switches['adiabatic contraction'],
                                                 logM_baryon=self.true_values['M_baryon'], DT=self.true_values['DT'],
                                                 disk_re=self.true_values['Re'], disk_n=self.disk_n, disk_q=self.disk_q,
                                                 disk_lw=bool(self.disk_lw), BT=self.true_values['BT'],
                                                 bulge_n=self.bulge_n, bulge_q=self.bulge_q,
                                                 bulge_lw=bool(self.bulge_lw), ring_FWHM=self.true_values['ring_FWHM'],
                                                 ring_rpeak=self.true_values['R_peak'], ring_lw=bool(self.ring_lw),
                                                 running_in_cluster=self.running_in_cluster, apply2D=self.apply_2D)
        if self.mass_components_switches['halo']:
            if self.mass_components_switches['disk']:
                true_f = calculate_fraction_at_re(mass_components=self.mass_components, reval=self.mass_components['disk'].r_eff)
            elif self.mass_components_switches['ring']:
                true_f = calculate_fraction_at_re(mass_components=self.mass_components, reval=self.mass_components['ring'].r_eff)
            else:
                true_f = 0.
                logger.warning('No disk or ring used. cannot calculate DM fractions at Re...')
        else:
            true_f = 0.
            logger.warning('No halo used. cannot calculate DM fractions at Re...')

        self.true_values['f'] = true_f

        self.dx = 0.1
        self.oversample = 1
        self.radial_space = None
        self.edge = None

        if obsdata is None:
            if os.path.exists(self.rawdata_path):
                self.rawdata = pd.read_csv(os.path.join(self.rawdata_path), sep=r'\t', header=None, engine='python',
                                           names=['r ["]', "V", "V_err", "disp", "disp_err", "flux", "flux_err"])
                self.rawdata_r = np.array(self.rawdata['r ["]']) * self.kpc_to_arcsec
                self.rawdata_V = np.array(self.rawdata['V'])
                self.rawdata_V_err = np.array(self.rawdata['V_err'])
                self.rawdata_disp = np.array(self.rawdata['disp'])
                self.rawdata_disp_err = np.array(self.rawdata['disp_err'])
                self.rawdata_flux = np.array(self.rawdata['flux'])
                self.rawdata_flux_err = np.array(self.rawdata['flux_err'])
                self.define_radial_space(edge=np.max(np.abs(self.rawdata_r)), resolution=self.dx, oversample=self.oversample)
        else:
            self.rawdata_r = np.array(obsdata['r']) * self.kpc_to_arcsec
            self.rawdata_V = np.array(obsdata['V'])
            self.rawdata_V_err = np.array(obsdata['V_err'])
            self.rawdata_disp = np.array(obsdata['disp'])
            self.rawdata_disp_err = np.array(obsdata['disp_err'])
            self.rawdata_flux = np.array(self.rawdata['flux'])
            self.rawdata_flux_err = np.array(self.rawdata['flux_err'])
            self.define_radial_space(edge=np.max(np.abs(self.rawdata_r)), resolution=self.dx, oversample=self.oversample)

        # Normalize flux
        normalization = np.max(self.rawdata_flux)
        self.rawdata_flux /= normalization
        self.rawdata_flux_err /= normalization

        self.fit_goals = {
            'flux': self.metadata['fit_flux'],
            'velocity': self.metadata['fit_velocity'],
            'dispersion': self.metadata['fit_dispersion']
        }

        self.dof = len(self.rawdata_r) - sum(self.switches['parameters'].values())

        self.output_dir = "/".join([galaxy_outputs_folder, '%s-free[%s]-%04d-%02d-%02d' %
                                    (self.name,
                                     '%s' % ', '.join(map(str, [x for x in self.switches['parameters'] if self.switches['parameters'][x] == 1])),
                                     int(datetime.datetime.now().year), int(datetime.datetime.now().month),
                                     int(datetime.datetime.now().day))])
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)


    def Mvir_Moster2018(z, log_mstar):
        z1 = 1 + z
        n = np.power(10, 1.507 - 0.124 * z / z1)
        b = -0.621 - 0.059 * z / z1
        g = 1.055 + 0.838 * z / z1 - 3.083 * (z / z1) ** 2

        log_mvir = log_mstar + np.log10(0.5) + np.log10(n) + np.log10(
            (10 ** (log_mstar - 10.6)) ** b + (10 ** (log_mstar - 10.6)) ** g)

        return log_mvir

    def define_radial_space(self, edge, resolution, oversample=1):
        self.edge = np.ceil(edge*10) / 10 + 5.
        self.dx = resolution / oversample
        self.radial_space = {'dx': resolution,
                             'array': create_r_space(edge=self.edge, resolution=resolution)}
