import numpy as np
import time
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.signal import windows

from RotCurves.baryons import *
from RotCurves.dm_halos import *
from RotCurves.base_utils import create_r_space


class RotationCurveObject:
    def __init__(self, galaxy=None, edge=None, dx=None, rarray=None, sigma_inst=0.,
                 oversample=1., oversample_edge=4.,
                 Halo=None, Disk=None, Ring=None, Bulge=None, sigma_dispersion=None,
                 dispersion_function='const', pressure_support="general",
                 inclination=90, PA=0., sigma_beam=None, FWHM_beam=None,
                 apply_2D=True, include_beam_smearing=True,
                 printtime=False, ndim=1., radial_velocity=0):
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
        self.oversample_edge = oversample_edge
        self.oversample = oversample

        self.sigma_inst = sigma_inst
        self.sigma0 = sigma_dispersion
        self.dispersion_function = dispersion_function
        self.pressure_support = pressure_support
        self.ndim = ndim
        self.PA = PA
        self.Vradial = radial_velocity

        if self.sigma_beam is None and self.FWHM_beam is not None:
            self.sigma_beam = self.FWHM_beam / (2 * np.sqrt(2 * np.log(2)))

        if galaxy is not None:
            self.dx = galaxy.dx
            self.edge = galaxy.edge
            self.oversample_edge = galaxy.oversample_edge
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
                self.oversample_edge = oversample_edge
                self.sigma_inst = sigma_inst
        else:
            self.dx = dx
            self.oversample_edge = oversample_edge

        if self.sigma_inst is None:
            self.sigma_inst = 0

        if self.oversample_edge is None:
            self.oversample_edge = 4

        # apply oversample to pixels scale
        if self.dx is not None:
            self.dx = self.dx / self.oversample

        if rarray is not None:
            self.R_majoraxis = rarray
            self.edge = np.max(rarray)
            if len(rarray) > 1:
                self.dx = np.diff(rarray)[0]
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
                self.sampling_edge = self.edge + self.oversample_edge * self.sigma_beam
                self.oversample_pixels = int(round(self.oversample_edge * self.sigma_beam / self.dx))
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
                # TODO: THIS IS AN ERROR, ROUNDING SIGMA_BEAM ALTERS THE SIZE OF THE BEAM
                # TODO: IT SHOULD BE ROUNDED DOWN AND HANDLE THE LEFTOVERS CAREFULLY
                self.sigma_beam_pixels_x = int(np.ceil(self.sigma_beam_x / self.dx))
                self.sigma_beam_pixels_y = int(np.ceil(self.sigma_beam_y / self.dx))
                self.oversample_edge_pixels_x = int(np.ceil(self.oversample_edge * self.sigma_beam_pixels_x))
                self.oversample_edge_pixels_y = int(np.ceil(self.oversample_edge * self.sigma_beam_pixels_y))
                self.sampling_edge_x = self.edge + self.oversample_edge_pixels_x * self.dx
                self.sampling_edge_y = self.edge + self.oversample_edge_pixels_y * self.dx
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

                self.smeared_with_inclination = np.round(self.smeared_with_inclination, 4)
                self.smeared = np.round(self.smeared, 4)
                self.velocity_dispersion = np.round(self.velocity_dispersion, 4)

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
            self.V2d = self.disk.vcirc2(R_array)
            self.Vd = self.disk.vcirc2(R_array)
        if self.ring is not None:
            self.V2r = self.ring.vcirc2(R_array)
            self.Vr = self.ring.vcirc(R_array)
        if self.bulge is not None:
            self.V2b = self.bulge.vcirc2(R_array)
            self.Vb = self.bulge.vcirc(R_array)
        if self.halo is not None:
            # if (self.disk is not None) and (self.bulge is not None):
            #     self.halo.get_RC(R_array, disk=self.disk, bulge=self.bulge)
            # elif (self.ring is not None) and (self.bulge is not None):
            #     self.halo.get_RC(R_array, disk=self.ring, bulge=self.bulge)
            # else:
            self.V2h = self.halo.vcirc2(R_array)
            self.Vh = self.halo.vcirc(R_array)

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
            for comp in [self.disk, self.ring]:
                if comp is not None:
                    if comp._is_massive():
                        # self.V2sigma += 2 * self.sigma_profile ** 2 * (absR/comp.r_s) * comp.dlnrho_dlnr(absR)
                        self.V2sigma += 2 * self.sigma_profile ** 2 * comp.dlnrho_dlnr(absR)

        self.V2sigma = np.nan_to_num(self.V2sigma)
        # Vsigma_interim = np.copy(self.V2sigma)
        # Vsigma_interim[Vsigma_interim < 0] = 0
        self.Vsigma = np.sqrt(np.abs(self.V2sigma)) * np.sign(self.V2sigma)

        ### baryons velocity
        self.V2baryon = self.V2d + self.V2b + self.V2r
        self.Vbaryon = np.sqrt(np.maximum(0, self.V2baryon)) * np.sign(R_array)

        ### circular velocity
        self.V2circ = self.V2baryon + self.V2h
        self.Vcirc = np.sqrt(np.maximum(0, self.V2circ)) * np.sign(R_array)

        ### correct for pressure support (Vrot)
        self.V2rot = self.V2circ + self.V2sigma
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

        interpolator = CubicSpline(x=one_dimensional_rarray, y=one_dimensional_vel)
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
            Igrid += self.disk.light_profile(xx, yy)
        if self.ring is not None:
            Igrid += self.ring.light_profile(xx, yy)
        if self.bulge is not None:
            Igrid += self.bulge.light_profile(xx, yy)

        # Check that Igrid is non-zero
        if np.all(Igrid == 0):
            logger.warning('No light weighting used. Assuming constant light...')
            Igrid = np.ones_like(rgrid)

        # normalize Igrid in case of really low values
        # Igrid /= np.max(Igrid)

        # kernel_y = windows.gaussian(2 * self.oversample_edge_pixels_y + 1, self.sigma_beam_pixels_y)
        # kernel_x = windows.gaussian(2 * self.oversample_edge_pixels_x + 1, self.sigma_beam_pixels_x)
        # gaussian_kernel = np.outer(kernel_y, kernel_x)

        # TODO: correction for rounding sigma_pixels
        sigma_x = self.sigma_beam_x / self.dx
        sigma_y = self.sigma_beam_y / self.dx
        size_x = int(np.ceil(self.oversample_edge * sigma_x))
        size_y = int(np.ceil(self.oversample_edge * sigma_y))
        x = np.arange(-size_x, size_x + 1)
        y = np.arange(-size_y, size_y + 1)
        xx, yy = np.meshgrid(x, y)
        gaussian_kernel = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
        gaussian_kernel /= np.sum(gaussian_kernel)

        if ndim == 1.:
            majoraxis_idx = round((rgrid.shape[0] - 1) / 2)
            majoraxis = rgrid[majoraxis_idx]
            N = len(majoraxis)

            V_major_axis = []
            smeared_light_major_axis = []
            for idx in range(self.oversample_edge_pixels_x, N - self.oversample_edge_pixels_x):
                # y_min = majoraxis_idx - self.oversample_edge_pixels_y
                # y_max = majoraxis_idx + self.oversample_edge_pixels_y + 1
                # x_min = idx - self.oversample_edge_pixels_x
                # x_max = idx + self.oversample_edge_pixels_x + 1

                # Kernel shape
                kernel_h, kernel_w = gaussian_kernel.shape  # e.g., (121, 97)

                # When slicing from the grid, always extract a patch of the same shape
                # (centered at (cy, cx) with half-widths)
                half_h = kernel_h // 2
                half_w = kernel_w // 2

                # For a point at (y, x) in your grid:
                y_min = majoraxis_idx - half_h
                y_max = majoraxis_idx + half_h + 1
                x_min = idx - half_w
                x_max = idx + half_w + 1

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

            for idx in range(self.oversample_edge_pixels_y, len(self.sampling_rarray_y) - self.oversample_edge_pixels_y):
                V2d_array_idx = np.zeros(shape=len(self.R_majoraxis))

                for j in range(self.oversample_edge_pixels_x, len(self.sampling_rarray_x) - self.oversample_edge_pixels_x):
                    y_min = idx - self.oversample_edge_pixels_y
                    y_max = idx + self.oversample_edge_pixels_y + 1
                    x_min = j - self.oversample_edge_pixels_x
                    x_max = j + self.oversample_edge_pixels_x + 1

                    Vgrid_idx = Vgrid[y_min:y_max, x_min:x_max]
                    Igrid_idx = Igrid[y_min:y_max, x_min:x_max]
                    V2d_array_idx_j = np.divide(np.sum(Vgrid_idx * Igrid_idx * gaussian_kernel), np.sum(Igrid_idx * gaussian_kernel), out=np.zeros(shape=(1)), where=np.sum(Igrid_idx*gaussian_kernel)!=0)
                    V2d_array_idx[j - self.oversample_edge_pixels_x] = (V2d_array_idx_j)

                V2d_array[idx - self.oversample_edge_pixels_y] = V2d_array_idx

                Igrid = np.asarray(Igrid)
                Igrid = np.nan_to_num(Igrid)

            return V2d_array, Igrid


def calculate_fraction_at_re(mass_components=None, reval=None):
    if reval is None:
        print('r_fdm not specified. Assuming reval = 1kpc...')


    if mass_components['halo'] is None:
        print('Cant calculate DM fractions for a model with no halo!')
        return 0.
    else:
        rc = RotationCurveObject(rarray=[reval], Halo=mass_components['halo'], Disk=mass_components['disk'],
                                 Ring=mass_components['ring'], Bulge=mass_components['bulge'], sigma_dispersion=0.,
                                 pressure_support='general', apply_2D=False, include_beam_smearing=False)
        fraction = rc.V2h / (rc.V2h + rc.V2baryon)
        return fraction[0]
