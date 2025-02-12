from utils import *
from mass_models import HaloObject, SersicObject, GaussianRingObject
from scaling_relations import Mvir_Moster2018


class RotationCurveObject:
    def __init__(self, galaxy=None, edge=None, dx=None, rarray=None, sigma_inst=0., oversample=4., Halo=None, Disk=None, Ring=None, Bulge=None, sigma_dispersion=None,
                 dispersion_function='const', pressure_support="general", inclination=90, sigma_beam=None, FWHM_beam=None, apply_2D=True, include_beam_smearing=True,
                 printtime=False, ndim=1., PA=0., radial_velocity=0):
        if printtime:
            self.starttime = time.time()

        self.galaxy = galaxy
        self.halo = Halo
        self.disk = Disk
        self.ring = Ring
        self.bulge = Bulge
        self.inclination = inclination
        self.sigma_beam = sigma_beam
        self.FWHM_beam = FWHM_beam
        self.apply_2D = apply_2D
        self.include_beam_smearing = include_beam_smearing
        self.oversample = oversample

        self.sigma_inst = sigma_inst
        self.sigma0 = sigma_dispersion * 1e3
        self.dispersion_function = dispersion_function
        self.pressure_support = pressure_support
        self.ndim = ndim
        self.PA = PA
        self.Vradial = radial_velocity*1e3

        if self.sigma_beam is None and self.FWHM_beam is not None:
            self.sigma_beam = self.FWHM_beam / (2 * np.sqrt(2 * np.log(2)))

        if galaxy is not None:
            self.dx = galaxy.dx
            self.edge = galaxy.edge
            self.oversample = galaxy.oversample
            self.sigma_inst = galaxy.sigma_inst
        elif edge is not None:
            if dx is None:
                print('dx not specified. assuming dx = 0.1 [kpc].')
                self.edge = edge
                self.dx = 0.1

            else:
                self.dx = dx
                self.edge = edge
                self.oversample = oversample
                self.sigma_inst = sigma_inst
        else:
            self.dx = dx
            self.oversample = oversample

        if self.sigma_inst is None:
            self.sigma_inst = 0

        if self.oversample is None:
            self.oversample = 4

        if rarray is not None:
            self.R_majoraxis = rarray
            self.edge = np.max(rarray/kpc)
            if len(rarray) > 1:
                self.dx = np.diff(rarray/kpc)[0]
        else:
            self.R_majoraxis = create_r_space(edge=self.edge, resolution=self.dx)

        if self.sigma_beam is not None:
            self.sigma_beam_pixels = round(self.sigma_beam / self.dx)

        ### only calculate intrinsic RC
        if not self.include_beam_smearing:
            self.make_intrinsic_rotationCurve(self.R_majoraxis)

        ### beam-smeared RC
        else:
            ### do a 1D rotation curve ###
            if not self.apply_2D:
                # define the 1D radial spaces
                self.sampling_edge = self.edge + self.oversample * self.sigma_beam
                self.oversample_pixels = int(round(self.oversample * self.sigma_beam / self.dx))
                self.sampling_rarray_1D = create_r_space(edge=self.sampling_edge, resolution=self.dx)

                # create intrinsic RC for the sampling range
                self.make_intrinsic_rotationCurve(self.sampling_rarray_1D)

                # create beam-smeared RC along the major axis
                self.smeared = self.apply_1D_beam_smearing(self.intrinsic)
                self.smeared_with_inclination = self.apply_1D_beam_smearing(self.intrinsic_with_inclination)
                self.smeared_no_dispersion = self.apply_1D_beam_smearing(self.intrinsic_no_dispersion)
                self.smeared_no_dispersion_with_inclination = self.apply_1D_beam_smearing(self.intrinsic_no_dispersion_with_inclination)

                # estimate velocity dispersion
                V_average = self.smeared_with_inclination
                V_sqaured = self.intrinsic_with_inclination ** 2
                V_sqaured_average = self.apply_1D_beam_smearing(V_sqaured)
                dispersion_squared = V_sqaured_average - V_average**2
                self.velocity_dispersion = np.sqrt(dispersion_squared + self.sigma_profile ** 2)

                # choose major axis only from the intrinsic curves
                N = len(self.intrinsic)
                self.intrinsic = self.intrinsic[self.oversample_pixels:N-self.oversample_pixels]
                self.intrinsic_with_inclination = self.intrinsic_with_inclination[self.oversample_pixels:N-self.oversample_pixels]
                self.intrinsic_no_dispersion = self.intrinsic_no_dispersion[self.oversample_pixels:N-self.oversample_pixels]
                self.intrinsic_no_dispersion_with_inclination = self.intrinsic_no_dispersion_with_inclination[self.oversample_pixels:N-self.oversample_pixels]

                if printtime:
                        print('time for 1D:', np.round(time.time() - self.starttime, 1))

            ### do a 2D rotation curve ###
            if apply_2D:
                geometrical_factor_elliptical = 1 / np.cos(np.deg2rad(self.inclination))

                self.sigma_beam_x = self.sigma_beam
                self.sigma_beam_y = self.sigma_beam * geometrical_factor_elliptical
                self.sigma_beam_pixels_x = int(round(self.sigma_beam_x / self.dx))
                self.sigma_beam_pixels_y = int(round(self.sigma_beam_y / self.dx))
                self.oversample_pixels_x = int(round(self.oversample * self.sigma_beam_pixels_x))
                self.oversample_pixels_y = int(round(self.oversample * self.sigma_beam_pixels_y))
                # self.sampling_edge_x = np.round(self.edge + self.oversample_pixels_x * self.dx, 1)
                # self.sampling_edge_y = np.round(self.edge + self.oversample_pixels_y * self.dx, 1)
                self.sampling_edge_x = self.edge + self.oversample_pixels_x * self.dx
                self.sampling_edge_y = self.edge + self.oversample_pixels_y * self.dx
                self.sampling_rarray_x = create_r_space(edge=self.sampling_edge_x, resolution=self.dx)
                self.sampling_rarray_y = create_r_space(edge=self.sampling_edge_y, resolution=self.dx)
                self.sampling_edge_2D = np.sqrt(self.sampling_edge_x**2 + self.sampling_edge_y**2)
                self.sampling_rarray_2D = create_r_space(edge=self.sampling_edge_2D, resolution=self.dx)

                # create the velocity array used to build the 2D grid from
                self.make_intrinsic_rotationCurve(self.sampling_rarray_2D)

                # create the beam-smeared rotation curve along the major axis
                self.smeared_with_inclination, self.smeared_light_profile = \
                    self.apply_2D_beam_smearing(one_dimensional_vel=self.intrinsic_with_inclination,one_dimensional_rarray=self.sampling_rarray_2D, ndim=self.ndim)
                self.smeared = self.smeared_with_inclination / np.sin(np.deg2rad(self.inclination))
                V_average = self.smeared_with_inclination
                V_squared = self.intrinsic_with_inclination ** 2
                # V_squared_average, _ = self.apply_2D_beam_smearing(one_dimensional_vel=V_squared,
                #                                                 one_dimensional_rarray=self.sampling_rarray_2D,
                #                                                 ndim=self.ndim, for_dispersion=True)
                V_squared_average, _ = self.apply_2D_beam_smearing(one_dimensional_vel=self.intrinsic_with_inclination,
                                                                one_dimensional_rarray=self.sampling_rarray_2D,
                                                                ndim=self.ndim, for_dispersion=True)
                dispersion_squared = V_squared_average - V_average ** 2
                if self.ndim == 1:
                    dispersion_squared = dispersion_squared.flatten()

                # create the intrinsic curves in the original space
                self.make_intrinsic_rotationCurve(self.R_majoraxis)

                self.velocity_dispersion = np.sqrt(dispersion_squared + self.sigma_profile ** 2 + self.sigma_inst ** 2)

                if self.ndim == 2.:
                    self.smeared_with_inclination = rotate(self.smeared_with_inclination, angle=90-self.PA, reshape=False)
                    self.velocity_dispersion = rotate(self.velocity_dispersion, angle=90-self.PA, reshape=False)

                self.smeared_with_inclination = np.round(self.smeared_with_inclination) * 1e-3
                self.smeared = np.round(self.smeared) * 1e-3
                self.velocity_dispersion = np.round(self.velocity_dispersion) * 1e-3

                if self.ndim == 1:
                    self.smeared_with_inclination = np.reshape(self.smeared_with_inclination, len(self.smeared_with_inclination))
                    self.smeared = np.reshape(self.smeared, len(self.smeared))
                    self.velocity_dispersion = np.reshape(self.velocity_dispersion, len(self.velocity_dispersion))

                if printtime:
                    print('time for 2D:', np. round(time.time() - self.starttime, 1))

    ### get intrinsic dispersion profile
    def get_dispersion_profile(self, R_array, functional_form='const'):
        if functional_form in ['const', 'constant', 'flat']:
            self.dispersion_func = lambda r: 1.
        elif functional_form in ['constant_h', 'constant_height', 'const_h']:
            if self.disk is not None:
                self.dispersion_func = lambda r: np.sqrt(self.disk.density_function(np.abs(r)) / self.disk.Sig0)
            elif self.ring is not None:
                self.dispersion_func = lambda r: np.sqrt(self.ring.density_function(np.abs(r)) / self.ring.Sig0)
        elif functional_form in ['power_law']:
            if self.disk is not None:
                self.dispersion_func = lambda r: np.divide(1, 1 + (np.divide(np.abs(r), self.disk.rd, out=np.zeros_like(r), where=r!=0)))

        self.sigma_profile = self.sigma0 * self.dispersion_func(R_array)

    def make_intrinsic_rotationCurve(self, R_array):
        absR = np.abs(R_array)

        self.V2d = np.zeros_like(R_array)
        self.Vd = np.zeros_like(R_array)
        self.V2b = np.zeros_like(R_array)
        self.Vb = np.zeros_like(R_array)
        self.V2r = np.zeros_like(R_array)
        self.Vr = np.zeros_like(R_array)
        self.V2h = np.zeros_like(R_array)
        self.Vh = np.zeros_like(R_array)

        if self.disk is not None:
            if self.disk._is_massive():
                self.disk.get_RC(R_array)
                self.V2d = self.disk.V2
                self.Vd = self.disk.V
        if self.ring is not None:
            if self.ring._is_massive():
                self.ring.get_RC(R_array)
                self.V2r = self.ring.V2
                self.Vr = self.ring.V
        if self.bulge is not None:
            if self.bulge._is_massive():
                self.bulge.get_RC(R_array)
                self.V2b = self.bulge.V2
                self.Vb = self.bulge.V
        if self.halo is not None:
            if (self.disk is not None) and (self.bulge is not None):
                self.halo.get_RC(R_array, disk=self.disk, bulge=self.bulge)
            elif (self.ring is not None) and (self.bulge is not None):
                self.halo.get_RC(R_array, disk=self.ring, bulge=self.bulge)
            else:
                self.halo.get_RC(R_array, disk=self.disk, bulge=self.bulge)
            self.V2h = self.halo.V2
            self.Vh = self.halo.V

        self.get_dispersion_profile(R_array, functional_form=self.dispersion_function)
        self.V2sigma = np.zeros_like(R_array)
        ### Regular exponential profile (using re)
        if self.pressure_support in ['exponential', 'Exponential']:
            if self.disk is not None:
                re = self.disk.re
            elif self.ring is not None:
                re = self.ring.rpeak
            else:
                logger.warning('PRESSURE SUPPORT: exponential: No disk or rings component found, assuming Re=1 kpc...')
                re = 1.

            self.V2sigma += 3.36 * (absR / re) * self.sigma_profile ** 2

        ### General Burkert(2010) formula using analytical derivatives of density profiles
        elif self.pressure_support in ['general', 'General', 'Generalized', 'generalized', 'burkert', 'burkert10', 'Burkert', 'Burkert10']:
            if self.disk is not None:
                if self.disk._is_massive():
                    self.V2sigma += - 2 * self.sigma_profile ** 2 * absR * (self.disk.density_prime_to_density_function(absR))
            if self.ring is not None:
                if self.ring._is_massive():
                    # self.V2sigma += - 2 * self.sigma_profile ** 2 * absR * (self.ring.density_prime_to_density_function(absR)) # old based on eq. 9 of Burkert(2010). Using 2*sigma0^2*dlnSigma/dlnr
                    self.V2sigma += - self.sigma_profile ** 2 * absR * (self.ring.density_prime_to_density_function(absR))   # # new based on eq. 3 of Burkert(2010). Using sigma0^2*dlnrho/dlnr

        ### using sersic n profile specifically
        elif self.pressure_support in ['Sersic_disk', 'sersic_disk']:
            self.V2sigma += 2 * (self.sigma_profile ** 2) * sersic_b() / self.disk.n * (absR / self.disk.re) ** (1 / self.disk.n)

        ### Gaussian ring density profile
        elif self.pressure_support in ['Gaussian_ring', 'gaussian_ring']:
            # self.V2sigma += 4 * self.sigma_profile**2 * self.ring.x0 * (absR / self.ring.r0) * ((absR / self.ring.r0) - 1)  # old based on eq. 9 of Burkert(2010). Using 2*sigma0^2*dlnSigma/dlnr
            self.V2sigma += 2 * self.sigma_profile**2 * self.ring.x0 * (absR / self.ring.r0) * ((absR / self.ring.r0) - 1)    # new based on eq. 3 of Burkert(2010). Using sigma0^2*dlnrho/dlnr

        self.V2sigma = np.nan_to_num(self.V2sigma)
        Vsigma_interim = np.copy(self.V2sigma)
        Vsigma_interim[Vsigma_interim < 0] = 0
        self.Vsigma = np.sqrt(Vsigma_interim) * np.sign(R_array)

        ### baryons velocity
        self.V2baryon = self.V2d + self.V2b + self.V2r
        self.Vbaryon = np.sqrt(np.maximum(0, self.V2baryon)) * np.sign(R_array)

        ### circular velocity
        self.V2circ = self.V2baryon + self.V2h
        self.Vcirc = np.sqrt(np.maximum(0, self.V2circ)) * np.sign(R_array)

        ### correct for pressure support (Vrot)
        self.V2rot = self.V2h + self.V2baryon - self.V2sigma
        self.Vrot = np.sqrt(np.maximum(0, self.V2rot))

        ### final velocities
        self.intrinsic_no_dispersion = np.sqrt(np.maximum(0, self.V2circ)) * np.sign(R_array)
        self.intrinsic_no_dispersion_with_inclination = self.intrinsic_no_dispersion * np.sin(np.deg2rad(self.inclination))
        self.intrinsic = np.sqrt(np.maximum(0, self.V2rot)) * np.sign(R_array)
        self.intrinsic_with_inclination = self.intrinsic * np.sin(np.deg2rad(self.inclination))

        self.fdm = self.V2h / self.V2circ

    def apply_1D_beam_smearing(self, velocity_array, truncate=4.0, mode="nearest"):
            smeared = gaussian_filter1d(velocity_array, self.sigma_beam_pixels, truncate=truncate, mode=mode)
            N = len(smeared)
            return smeared[self.oversample_pixels: N - self.oversample_pixels]

    def apply_2D_beam_smearing(self, one_dimensional_vel, one_dimensional_rarray, for_dispersion=False, ndim=None):

        if ndim is None:
            ndim = self.ndim

        # build 2D grid from the size of the input 1D array
        xx, yy = np.meshgrid(self.sampling_rarray_x, self.sampling_rarray_y)
        rgrid = np.sqrt(xx ** 2 + yy ** 2)

        interpolator = scp_interp.CubicSpline(x=one_dimensional_rarray, y=one_dimensional_vel)
        Vgrid = interpolator(rgrid)

        costhetha_grid = np.divide(xx, rgrid, out=np.zeros_like(rgrid), where=rgrid!=0)
        sinthetha_grid = np.divide(yy, rgrid, out=np.zeros_like(rgrid), where=rgrid!=0)

        # project on LOS velocity
        Vgrid *= costhetha_grid

        # Add radial flow
        Vgrid += self.Vradial * np.sin(np.deg2rad(self.inclination)) * sinthetha_grid

        if for_dispersion:
            Vgrid = Vgrid**2

        # apply light weighting filter
        Igrid = np.zeros_like(rgrid)
        if self.disk is not None:
            if self.disk.light_weighting:
                Igrid += self.disk.light_profile(x=xx, y=yy)
        if self.ring is not None:
            if self.ring.light_weighting:
                Igrid += self.ring.light_profile(x=xx, y=yy)
        if self.bulge is not None:
            if self.bulge.light_weighting:
                Igrid += self.bulge.light_profile(x=xx, y=yy)

        # Check that Igrid is non-zero
        if np.all(Igrid == 0):
            logger.warning('No light weighting used. Assuming constant light...')
            Igrid = np.ones_like(rgrid)

        # normalize Igrid in case of really low values
        # Igrid /= np.max(Igrid)

        kernel_y = scp_sig.windows.gaussian(2*self.oversample_pixels_y + 1, self.sigma_beam_pixels_y)
        kernel_x = scp_sig.windows.gaussian(2*self.oversample_pixels_x + 1, self.sigma_beam_pixels_x)
        gaussian_kernel = np.outer(kernel_y, kernel_x)

        if ndim == 1.:
            majoraxis_idx = round((rgrid.shape[0] - 1) / 2)
            majoraxis = rgrid[majoraxis_idx]
            N = len(majoraxis)

            V_major_axis = []
            smeared_light_major_axis = []
            for idx in range(self.oversample_pixels_x, N-self.oversample_pixels_x):
                y_min = majoraxis_idx - self.oversample_pixels_y
                y_max = majoraxis_idx + self.oversample_pixels_y + 1
                x_min = idx - self.oversample_pixels_x
                x_max = idx + self.oversample_pixels_x + 1

                Vgrid_idx = Vgrid[y_min:y_max, x_min:x_max]
                Igrid_idx = Igrid[y_min:y_max, x_min:x_max]

                weights = Igrid_idx * gaussian_kernel
                numerator = np.sum(Vgrid_idx * weights)
                denominator = np.sum(weights)
                V_major_axis_idx = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
                V_major_axis.append(V_major_axis_idx)

                numerator = np.sum(Igrid_idx * gaussian_kernel)
                denominator = np.sum(gaussian_kernel)
                smeared_light_major_axis_idx = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
                smeared_light_major_axis.append(smeared_light_major_axis_idx)

            # make sure velocity array has no Nans
            V_major_axis = np.array(V_major_axis)
            V_major_axis = np.nan_to_num(V_major_axis)

            # normalize light profile
            smeared_light_major_axis = np.array(smeared_light_major_axis)
            smeared_light_major_axis = np.nan_to_num(smeared_light_major_axis)
            # smeared_light_major_axis /= np.max(smeared_light_major_axis)

            return V_major_axis, smeared_light_major_axis

        elif ndim == 2.:
            V2d_array = np.zeros(shape=(len(self.R_majoraxis), len(self.R_majoraxis)))

            for idx in range(self.oversample_pixels_y, len(self.sampling_rarray_y) - self.oversample_pixels_y):
                V2d_array_idx = np.zeros(shape=len(self.R_majoraxis))

                for j in range(self.oversample_pixels_x, len(self.sampling_rarray_x) - self.oversample_pixels_x):
                    y_min = idx - self.oversample_pixels_y
                    y_max = idx + self.oversample_pixels_y + 1
                    x_min = j - self.oversample_pixels_x
                    x_max = j + self.oversample_pixels_x + 1

                    Vgrid_idx = Vgrid[y_min:y_max, x_min:x_max]
                    Igrid_idx = Igrid[y_min:y_max, x_min:x_max]
                    V2d_array_idx_j = np.divide(np.sum(Vgrid_idx * Igrid_idx * gaussian_kernel), np.sum(Igrid_idx * gaussian_kernel), out=np.zeros(shape=(1)), where=np.sum(Igrid_idx*gaussian_kernel)!=0)
                    V2d_array_idx[j - self.oversample_pixels_x] = (V2d_array_idx_j)

                V2d_array[idx - self.oversample_pixels_y] = V2d_array_idx

                Igrid = np.asarray(Igrid)
                Igrid = np.nan_to_num(Igrid)

            return V2d_array, Igrid


class Prior:
    def __init__(self, type, initial_value, min_value, max_value, gauss_sigma):
        self.type = type
        self.initial = initial_value
        self.min = min_value
        self.max = max_value
        self.sig = gauss_sigma


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
        self.kpc_to_arcsec = cosmology(self.z).kpc2arcsec

        ## Instrument
        self.sigma_inst = self.metadata['sigma_inst'] * 1e3
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
            self.Mvir_moster = Mvir_Moster2018(self.z, self.Mstellar)

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
                true_f = calculate_fraction_at_re(mass_components=self.mass_components, reval=self.true_values['Re'])
            elif self.mass_components_switches['ring']:
                true_f = calculate_fraction_at_re(mass_components=self.mass_components, reval=self.mass_components['ring'].re())
            else:
                true_f = 0.
                logger.warning('No disk or ring used. cannot calculate DM fractions at Re...')
        else:
            true_f = 0.
            logger.warning('No halo used. cannot calculate DM fractions at Re...')

        self.true_values['f'] = true_f

        self.dx = 0.1
        self.oversample = 4
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

    def define_radial_space(self, edge, resolution, oversample=4):
        self.edge = np.ceil(edge*10) / 10 + 5.
        self.dx = resolution
        self.radial_space = {'dx': resolution,
                             'array': create_r_space(edge=self.edge, resolution=resolution, scale=kpc)}


def create_components(include_halo=False, include_disk=False, include_ring=False, include_bulge=False,
                      z=None, halo_profile='NFW', logM_vir=None, c=None, alpha=1., beta=3., gamma=1., AC=False,
                      logM_baryon=None, DT=None, disk_re=None, disk_n=1.0, disk_q=0.2, disk_lw=True,
                      BT=None, bulge_n=4.0, bulge_q=1.0, bulge_lw=False,
                      ring_rpeak=None, ring_FWHM=None, ring_lw=False,
                      running_in_cluster=False, apply2D=True):
    """
    :param mass_components_switches: dictionary of component switches    [halo, disk, ring, bulge]
    :param z: redshift
    :param halo_profile: type of halo to use                    [NFW (default), Burkert, Einasto, Dekel-Zhao]
    :param logM_vir: log virial mass                            [Msol]
    :param c: halo concentration parameter                      [dimless]
    :param alpha: halo inner slope                              [def = 1]
    :param beta: halo outer slope                               [def = 3]
    :param gamma: halo transition between slopes                [def = 1]
    :param AC: include adiabatic contraction                    [True / False]
    :param logM_baryon: log baryonic mass                       [Msol]
    :param DT: disk-to-total ratio                              [dimless]
    :param disk_re: disk effective radius                       [kpc]
    :param disk_n: disk Sersic index                            [def 1.0]
    :param disk_q: disk axis-ratio                              [def 0.2]
    :param disk_lw: whether to apply light weighting            [True / False]
    :param BT: bulge-to-total ratio                             [dimless]
    :param bulge_n: bulge Sersic index                          [def 4.0]
    :param bulge_q: bulge axis ratio                            [def 1.0 - spherical]
    :param bulge_lw: whether to apply light weighting           [True / False]
    :param ring_rpeak: ring peak radius                         [kpc]
    :param ring_FWHM: Full-width half-maximum of Gaussian ring  [kpc]
    :param ring_re: ring effective radius                       [kpc]
    :param ring_invh: ring_Rpeak / ring_FWHM                    [dimless]
    :param ring_lw: whether to apply light weighting            [True / False]
    :param apply2D: use a 2D grid for the RC                    [True / False]
    :param running_in_cluster: if running in CCA cluster        [True / False]

    :return: disk, bulge, halo Objects
    """

    halo = None
    disk = None
    ring = None
    bulge = None

    if include_disk:
        if include_ring:
            RT = 1 - BT - DT
        else:
            RT = 0.
            DT = 1 - BT
    else:
        DT = 0.
        if include_ring:
            if include_bulge:
                RT = 1 - BT
            else:
                RT = 1
                BT = 0
        else:
            RT = 0.

    if include_ring and BT == 0:
        logM_ring_tmp = logM_baryon + np.log10(RT)
        ring_tmp = GaussianRingObject(mass=np.power(10, logM_ring_tmp) * M_solar,
                                  rpeak=ring_rpeak if ring_rpeak is None else ring_rpeak * kpc,
                                  ring_FWHM=ring_FWHM if ring_FWHM is None else ring_FWHM * kpc,
                                  light_weighting=ring_lw,
                                  running_in_cluster=running_in_cluster, apply_2D=apply2D)
        ring_tmp.find_minimal_bulge()
        BT = ring_tmp.BT_min
        include_bulge = True
        # logger.warning('Mass components: BT cant be zero with a Gaussian ring. Setting BT_min = %0.2f ...' % BT)

    if RT < 0:
        logger.warning('Mass components: Negative ring mass with: BT=%0.2f, DT=%0.2f, RT=%0.2f' % (BT, DT, RT))

    if DT < 0:
        logger.warning('Mass components: Negative disk mass with: BT=%0.2f, DT=%0.2f, RT=%0.2f' % (BT, DT, RT))

    if include_halo:
        halo = HaloObject(profile=halo_profile,
                          mass=np.power(10, logM_vir) * M_solar,
                          concentration_parameter=c,
                          redshift=z,
                          alpha=alpha,
                          beta=beta,
                          gamma=gamma,
                          contracted=AC)

    if include_disk:
        logM_disk = logM_baryon + np.log10(DT)
        disk = SersicObject(mass=np.power(10, logM_disk) * M_solar,
                            re=disk_re * kpc,
                            sersic_n=disk_n,
                            q=disk_q,
                            light_weighting=disk_lw,
                            apply_2D=apply2D,
                            running_in_cluster=running_in_cluster)
    if not include_disk:
        if disk_lw:
            disk = SersicObject(mass=np.power(10, 0) * M_solar,
                                re=disk_re * kpc,
                                sersic_n=disk_n,
                                q=disk_q,
                                light_weighting=disk_lw,
                                apply_2D=apply2D,
                                running_in_cluster=running_in_cluster)

    if include_ring:
        logM_ring = logM_baryon + np.log10(RT)
        ring = GaussianRingObject(mass=np.power(10, logM_ring) * M_solar,
                                  rpeak=ring_rpeak if ring_rpeak is None else ring_rpeak * kpc,
                                  ring_FWHM=ring_FWHM if ring_FWHM is None else ring_FWHM * kpc,
                                  light_weighting=ring_lw,
                                  running_in_cluster=running_in_cluster, apply_2D=apply2D)
    if not include_ring:
        if ring_lw:
            ring = GaussianRingObject(mass=np.power(10, 0) * M_solar,
                                      rpeak=ring_rpeak if ring_rpeak is None else ring_rpeak * kpc,
                                      ring_FWHM=ring_FWHM if ring_FWHM is None else ring_FWHM * kpc,
                                      light_weighting=ring_lw,
                                      running_in_cluster=running_in_cluster, apply_2D=apply2D)

    if include_bulge:
        logM_bulge = logM_baryon + np.log10(BT)
        bulge = SersicObject(mass=np.power(10, logM_bulge) * M_solar,
                             re=1 * kpc,
                             sersic_n=bulge_n,
                             q=bulge_q,
                             light_weighting=bulge_lw,
                             apply_2D=apply2D,
                             running_in_cluster=running_in_cluster)

    mass_components = {'halo': halo, 'disk': disk, 'ring': ring, 'bulge': bulge}

    return mass_components


def calculate_fraction_at_re(mass_components=None, reval=None):
    if reval is None:
        print('r_fdm not specified. Assuming reval = 1kpc...')


    if mass_components['halo'] is None:
        print('Cant calculate DM fractions for a model with no halo!')
        return 0.
    else:
        rc = RotationCurveObject(rarray=[reval], Halo=mass_components['halo'], Disk=mass_components['disk'],
                                 Ring=mass_components['ring'], Bulge=mass_components['bulge'],
                                 sigma_dispersion=0., pressure_support='general',
                                 include_beam_smearing=False, apply_2D=False)
        fraction = rc.V2h / (rc.V2h + rc.V2baryon)
        return fraction[0]
