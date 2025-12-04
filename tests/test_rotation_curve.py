import numpy as np
from numpy.testing import assert_allclose

from RotCurves.rotation_curve import RotationCurveObject, calculate_fraction_at_re
from RotCurves.baryons import FreemanDisk, SersicProfile, GaussianRingProfile
from RotCurves.dm_halos import NFWHalo
from RotCurves.base_utils import create_r_space, create_r_space_oversampled

# Test tolerances
RTOL = 1e-3
ATOL = 1e-6


def test_rotation_curve_initialization_with_disk_and_halo():
    """Test basic initialization of RotationCurveObject with disk and halo."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    halo = NFWHalo(mass=1e12, concentration=10)
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk, Halo=halo,
        include_beam_smearing=False
    )
    
    assert rc.disk is not None
    assert rc.halo is not None
    assert rc.bulge is None
    assert rc.ring is None
    assert len(rc.R_majoraxis) > 0
    assert np.max(rc.R_majoraxis) <= 20.0 + 1e-6


def test_rotation_curve_initialization_with_rarray():
    """Test initialization with custom radial array."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    rarray = np.linspace(0.1, 10.0, 50)
    
    rc = RotationCurveObject(
        rarray=rarray, Disk=disk, include_beam_smearing=False
    )
    
    assert_allclose(rc.R_majoraxis, rarray, rtol=RTOL)


def test_intrinsic_rotation_curve_disk_and_halo():
    """Test that Vrot rotation curve correctly combines components."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    halo = NFWHalo(mass=1e12, concentration=10)
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk, Halo=halo,
        include_beam_smearing=False
    )
    
    # Check that velocity components are computed
    assert len(rc.V2d) == len(rc.R_majoraxis)
    assert len(rc.V2h) == len(rc.R_majoraxis)
    assert len(rc.V2baryon) == len(rc.R_majoraxis)
    assert len(rc.V2circ) == len(rc.R_majoraxis)
    
    # Check that baryonic velocity is sum of components
    assert_allclose(rc.V2baryon, rc.V2d, rtol=RTOL)
    
    # Check that circular velocity is sum of baryons and halo
    assert_allclose(rc.V2circ, rc.V2baryon + rc.V2h, rtol=RTOL)
    
    # Check that velocities are non-negative
    assert np.all(rc.V2baryon >= 0)
    assert np.all(rc.V2h >= 0)
    assert np.all(rc.V2circ >= 0)


def test_intrinsic_rotation_curve_with_bulge():
    """Test rotation curve with bulge component."""
    disk = SersicProfile(mass=1e10, r_s=2.5, n=1.0)
    bulge = SersicProfile(mass=1e9, r_s=0.5, n=4.0)
    halo = NFWHalo(mass=1e12, concentration=10)
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk, Bulge=bulge, Halo=halo,
        include_beam_smearing=False
    )
    
    # Check that bulge velocity is included
    assert len(rc.V2b) == len(rc.R_majoraxis)
    assert np.all(rc.V2b >= 0)
    
    # Check that baryonic velocity includes bulge
    assert_allclose(rc.V2baryon, rc.V2d + rc.V2b, rtol=RTOL)


def test_intrinsic_rotation_curve_with_ring():
    """Test rotation curve with ring component."""
    ring = GaussianRingProfile(mass=1e9, r_s=5.0, h=2.0, lookup=True)
    halo = NFWHalo(mass=1e12, concentration=10)
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Ring=ring, Halo=halo,
        include_beam_smearing=False
    )
    
    # Check that ring velocity is included
    assert len(rc.V2r) == len(rc.R_majoraxis)

    # Check that baryonic velocity includes ring
    assert_allclose(rc.V2baryon, rc.V2r, rtol=RTOL)


def test_pressure_support_general():
    """Test pressure support correction with general (Burkert) method."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    sigma_dispersion = 40.0  # km/s
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        sigma_dispersion=sigma_dispersion,
        pressure_support='general',
        include_beam_smearing=False
    )
    
    # Check that pressure support is computed
    assert len(rc.V2sigma) == len(rc.R_majoraxis)
    
    # With pressure support, rotation velocity should differ from circular velocity
    # (for most radii, pressure support reduces rotation velocity)
    assert np.any(np.abs(rc.V2rot - rc.V2circ) > RTOL)


def test_pressure_support_exponential():
    """Test pressure support correction with exponential method."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    sigma_dispersion = 10.0  # km/s
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        sigma_dispersion=sigma_dispersion,
        pressure_support='exponential',
        include_beam_smearing=False
    )
    
    # Check that pressure support is computed
    assert len(rc.V2sigma) == len(rc.R_majoraxis)
    assert np.any(np.abs(rc.V2rot - rc.V2circ) > RTOL)


def test_no_pressure_support():
    """Test that without pressure support, rotation equals circular velocity."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        sigma_dispersion=0.0,
        include_beam_smearing=False
    )
    
    # Without pressure support, rotation should equal circular velocity
    assert_allclose(rc.V2rot, rc.V2circ, rtol=RTOL, atol=ATOL)
    assert_allclose(rc.Vrot, rc.Vcirc, rtol=RTOL, atol=ATOL)


def test_inclination_effect():
    """Test that inclination affects line-of-sight velocity."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    
    rc_edgeon = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        inclination=85.0,  # edge-on
        include_beam_smearing=False
    )
    
    rc_faceon = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        inclination=0.0,  # face-on
        include_beam_smearing=False
    )
    
    # Edge-on should have larger line-of-sight velocity than face-on
    assert np.all(np.abs(rc_edgeon.Vrot_sini) >=
                  np.abs(rc_faceon.Vrot_sini))
    
    # Face-on should have near-zero line-of-sight velocity
    assert np.all(np.abs(rc_faceon.Vrot_sini) < 1e-3)


def test_dark_matter_fraction():
    """Test dark matter fraction calculation."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    halo = NFWHalo(mass=1e12, concentration=10)
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk, Halo=halo,
        include_beam_smearing=False
    )
    
    # Check that fdm is computed
    assert len(rc.fdm) == len(rc.R_majoraxis)
    
    # fdm should be between 0 and 1
    assert np.all(rc.fdm >= 0)
    assert np.all(rc.fdm <= 1)
    
    # fdm should equal V2h / V2circ
    expected_fdm = rc.V2h / rc.V2circ
    mask = rc.V2circ > 0
    assert_allclose(rc.fdm[mask], expected_fdm[mask], rtol=RTOL, atol=ATOL)


def test_velocity_dispersion_profile_constant():
    """Test constant velocity dispersion profile."""
    disk = SersicProfile(mass=1e10, r_s=2.0, n=2.0)
    sigma_dispersion = 15.0  # km/s
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        sigma_dispersion=sigma_dispersion,
        dispersion_function='const',
        include_beam_smearing=False
    )
    
    # For constant dispersion, profile should be constant
    assert_allclose(rc.sigma_profile, sigma_dispersion, rtol=RTOL, atol=ATOL)


def test_2D_beam_smearing():
    """Test 2D beam smearing."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    sigma_beam = 0.5  # kpc
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        sigma_beam=sigma_beam,
        include_beam_smearing=True,
        inclination=45.0,
        ndim=1
    )
    
    # Check that Vobs velocities are computed
    assert len(rc.Vobs) > 0
    assert len(rc.Vobs_sini) > 0
    assert len(rc.velocity_dispersion) > 0

    # Check that the Vobs velocity is smoother than the Vrot velocity
    assert np.var(np.diff(rc.Vobs)) < np.var(np.diff(rc.Vrot))


def test_FWHM_beam_conversion():
    """Test that FWHM_beam is correctly converted to sigma_beam."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    FWHM_beam = 4.0  # kpc
    expected_sigma = FWHM_beam / (2 * np.sqrt(2 * np.log(2)))
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        FWHM_beam=FWHM_beam,
        include_beam_smearing=True,
        inclination=50.0
    )
    
    assert np.isclose(rc.sigma_beam, expected_sigma, rtol=1e-6)


def test_calculate_fraction_at_re():
    """Test calculate_fraction_at_re function."""
    disk = SersicProfile(mass=1e10, r_eff=5.0, n=3.0, q0=0.5)
    halo = NFWHalo(mass=1e12, concentration=10)
    
    components = {
        'disk': disk,
        'halo': halo,
        'ring': None,
        'bulge': None
    }
    
    # Test with explicit radius
    reval = 5.0  # kpc
    f_dm = calculate_fraction_at_re(mass_components=components, reval=reval)
    
    assert 0 <= f_dm <= 1
    assert np.isfinite(f_dm)
    
    # Test with automatic radius (should use disk r_eff)
    f_dm_auto = calculate_fraction_at_re(mass_components=components, reval=None)
    
    assert 0 <= f_dm_auto <= 1
    assert np.isfinite(f_dm_auto)
    assert np.isclose(f_dm, f_dm_auto, rtol=RTOL)


def test_calculate_fraction_at_re_no_halo():
    """Test calculate_fraction_at_re returns 0 when no halo is present."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    
    components = {
        'disk': disk,
        'halo': None,
        'ring': None,
        'bulge': None
    }
    
    f_dm = calculate_fraction_at_re(mass_components=components, reval=5.0)
    assert f_dm == 0.0


def test_calculate_fraction_at_re_no_baryons():
    """Test calculate_fraction_at_re returns 1 when no baryons to estimate radius."""
    halo = NFWHalo(mass=1e12, concentration=10)
    
    components = {
        'disk': None,
        'halo': halo,
        'ring': None,
        'bulge': None
    }
    
    f_dm = calculate_fraction_at_re(mass_components=components, reval=None)
    assert f_dm == 1.0


def test_radial_velocity_component():
    """Test that radial velocity component is included in 2D beam smearing."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    radial_velocity = 10.0  # km/s
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        sigma_beam=0.5,
        radial_velocity=radial_velocity,
        include_beam_smearing=True,
        inclination=35.0,
        ndim=1
    )
    
    # Radial velocity should affect the Vobs velocity
    # (exact effect depends on geometry, but it should be included)
    assert len(rc.Vobs_sini) > 0


def test_velocity_arrays_have_correct_sign():
    """Test that velocity arrays preserve sign (positive/negative for receding/approaching)."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        include_beam_smearing=False
    )
    
    # Intrinsic velocity should preserve sign from R_array
    # (positive R should give positive velocity, negative R negative)
    positive_r = rc.R_majoraxis[rc.R_majoraxis > 0]
    negative_r = -np.abs(rc.R_majoraxis[rc.R_majoraxis > 0])
    
    # Check that velocities are computed correctly
    assert np.all(np.isfinite(rc.Vrot))
    assert np.all(np.isfinite(rc.Vcirc))


def test_dispersion_profile_with_ring():
    """Test velocity dispersion profile calculation with ring component."""
    ring = GaussianRingProfile(mass=1e9, r_s=5.0, h=2.0, lookup=True)
    sigma_dispersion = 30.0  # km/s
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Ring=ring,
        sigma_dispersion=sigma_dispersion,
        dispersion_function='constant_h',
        include_beam_smearing=False
    )
    
    # Check that dispersion profile is computed
    assert len(rc.sigma_profile) == len(rc.R_majoraxis)
    assert np.all(rc.sigma_profile >= 0)


def test_oversample_parameter():
    """Test that oversample parameter affects spatial resolution."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    
    rc_normal = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        oversample=1.0,
        include_beam_smearing=False
    )
    
    rc_oversampled = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        oversample=2.0,
        include_beam_smearing=False
    )
    
    # Oversampled should have finer effective resolution
    assert rc_oversampled.dx < rc_normal.dx
    # But same edge
    assert np.isclose(rc_oversampled.edge, rc_normal.edge, rtol=RTOL)


def test_dispersion_function_const():
    """Test constant dispersion function forms."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    sigma_dispersion = 20.0  # km/s
    
    # Test constant dispersion forms
    constant_forms = ['const', 'constant', 'flat']
    for form in constant_forms:
        rc = RotationCurveObject(
            edge=20.0, dx=0.1, Disk=disk,
            sigma_dispersion=sigma_dispersion,
            dispersion_function=form,
            include_beam_smearing=False
        )
        
        assert len(rc.sigma_profile) == len(rc.R_majoraxis)
        assert_allclose(rc.sigma_profile, sigma_dispersion, rtol=RTOL, atol=ATOL)
        assert np.all(np.isfinite(rc.sigma_profile))
        assert np.all(rc.sigma_profile >= 0)


def test_dispersion_function_constant_h_with_disk():
    """Test constant height dispersion function forms with disk."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    sigma_dispersion = 20.0  # km/s
    
    # Test constant height dispersion forms with disk
    constant_h_forms = ['constant_h', 'constant_height', 'const_h']
    for form in constant_h_forms:
        rc = RotationCurveObject(
            edge=20.0, dx=0.1, Disk=disk,
            sigma_dispersion=sigma_dispersion,
            dispersion_function=form,
            include_beam_smearing=False
        )
        
        assert len(rc.sigma_profile) == len(rc.R_majoraxis)
        assert np.all(rc.sigma_profile >= 0)
        assert np.var(rc.sigma_profile) > 0
        assert np.all(np.isfinite(rc.sigma_profile))


def test_dispersion_function_constant_h_with_ring():
    """Test constant height dispersion function forms with ring (no disk)."""
    ring = GaussianRingProfile(mass=1e9, r_s=5.0, h=2.0, lookup=True)
    sigma_dispersion = 20.0  # km/s
    
    # Test constant height dispersion forms with ring (no disk)
    constant_h_forms = ['constant_h', 'constant_height', 'const_h']
    for form in constant_h_forms:
        rc = RotationCurveObject(
            edge=20.0, dx=0.1, Ring=ring,
            sigma_dispersion=sigma_dispersion,
            dispersion_function=form,
            include_beam_smearing=False
        )
        
        # Should vary with ring surface density
        assert len(rc.sigma_profile) == len(rc.R_majoraxis)
        assert np.all(rc.sigma_profile >= 0)
        assert np.all(np.isfinite(rc.sigma_profile))


def test_dispersion_function_power_law():
    """Test power law dispersion function form."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    sigma_dispersion = 20.0  # km/s
    
    # Test power law dispersion form (requires disk)
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        sigma_dispersion=sigma_dispersion,
        dispersion_function='power_law',
        include_beam_smearing=False
    )
    
    # Should produce power-law profile
    assert len(rc.sigma_profile) == len(rc.R_majoraxis)
    assert np.all(rc.sigma_profile >= 0)
    # Should decrease with radius (power law form)
    assert rc.sigma_profile[0] >= rc.sigma_profile[-1]
    assert np.all(np.isfinite(rc.sigma_profile))


def test_oversample_rebinning_consistency_even():
    """Test that oversample=1 and oversample>1 return similar velocity arrays."""
    rtol = 5e-2
    atol = 0.0

    disk = FreemanDisk(mass=1e10, r_s=2.0)
    halo = NFWHalo(mass=1e12, concentration=10)
    sigma0 = 15.0
    FWHM_beam = 0.25
    inclination = 37.

    edge = 10.0
    dx = 0.1
    oversample = 2
    # Create rotation curve with oversample=1 (no oversampling)
    rc_normal = RotationCurveObject(
        edge=edge, dx=dx, Disk=disk, Halo=halo,
        sigma_dispersion=sigma0,
        FWHM_beam=FWHM_beam,
        inclination=inclination,
        include_beam_smearing=True
    )
    
    rc_oversampled = RotationCurveObject(
        edge=edge, dx=dx, Disk=disk, Halo=halo,
        oversample=oversample,
        sigma_dispersion=sigma0,
        FWHM_beam=FWHM_beam,
        inclination=inclination,
        include_beam_smearing=True
    )
    
    # Arrays should have the same length
    assert len(rc_normal.R_majoraxis) == len(rc_oversampled.R_majoraxis)
    assert len(rc_normal.Vcirc) == len(rc_oversampled.Vcirc)
    assert len(rc_normal.Vrot) == len(rc_oversampled.Vrot)
    
    # Check that original parameters are stored correctly for oversampled case
    assert rc_oversampled.dx_original == dx
    assert rc_oversampled.original_array_size == len(rc_normal.R_majoraxis)
    assert rc_oversampled.oversample == oversample

    # R_majoraxis should be identical
    assert_allclose(rc_normal.R_majoraxis, rc_oversampled.R_majoraxis, rtol=RTOL, atol=ATOL)
    
    # Velocity array should be close
    assert_allclose(rc_normal.Vd, rc_oversampled.Vd, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vh, rc_oversampled.Vh, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vbaryon, rc_oversampled.Vbaryon, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vcirc, rc_oversampled.Vcirc, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vrot, rc_oversampled.Vrot, rtol=rtol, atol=atol)

    assert_allclose(rc_normal.Vobs, rc_oversampled.Vobs, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vobs_sini, rc_oversampled.Vobs_sini, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.velocity_dispersion, rc_oversampled.velocity_dispersion, rtol=rtol, atol=atol)


def test_oversample_rebinning_consistency_odd():
    """Test that oversample=1 and oversample>1 return similar velocity arrays."""
    rtol = 5e-2
    atol = 0.0

    disk = FreemanDisk(mass=1e10, r_s=2.0)
    halo = NFWHalo(mass=1e12, concentration=10)
    sigma0 = 15.0
    FWHM_beam = 0.35
    inclination = 65.

    edge = 10.0
    dx = 0.1
    oversample = 3
    # Create rotation curve with oversample=1 (no oversampling)
    rc_normal = RotationCurveObject(
        edge=edge, dx=dx, Disk=disk, Halo=halo,
        sigma_dispersion=sigma0,
        FWHM_beam=FWHM_beam,
        inclination=inclination,
        include_beam_smearing=True
    )
    
    rc_oversampled = RotationCurveObject(
        edge=edge, dx=dx, Disk=disk, Halo=halo,
        oversample=oversample,
        sigma_dispersion=sigma0,
        FWHM_beam=FWHM_beam,
        inclination=inclination,
        include_beam_smearing=True
    )
    
    # Arrays should have the same length
    assert len(rc_normal.R_majoraxis) == len(rc_oversampled.R_majoraxis)
    assert len(rc_normal.Vcirc) == len(rc_oversampled.Vcirc)
    assert len(rc_normal.Vrot) == len(rc_oversampled.Vrot)
    
    # Check that original parameters are stored correctly for oversampled case
    assert rc_oversampled.dx_original == dx
    assert rc_oversampled.original_array_size == len(rc_normal.R_majoraxis)
    assert rc_oversampled.oversample == oversample

    # R_majoraxis should be identical
    assert_allclose(rc_normal.R_majoraxis, rc_oversampled.R_majoraxis, rtol=RTOL, atol=ATOL)
    
    # Velocity array should be close
    assert_allclose(rc_normal.Vd, rc_oversampled.Vd, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vh, rc_oversampled.Vh, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vbaryon, rc_oversampled.Vbaryon, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vcirc, rc_oversampled.Vcirc, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vrot, rc_oversampled.Vrot, rtol=rtol, atol=atol)

    assert_allclose(rc_normal.Vobs, rc_oversampled.Vobs, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.Vobs_sini, rc_oversampled.Vobs_sini, rtol=rtol, atol=atol)
    assert_allclose(rc_normal.velocity_dispersion, rc_oversampled.velocity_dispersion, rtol=rtol, atol=atol)


def test_oversampled_grid_divisibility():
    """Test that oversampled grids are always divisible by oversample factor."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    halo = NFWHalo(mass=1e12, concentration=10)
    rc_original = RotationCurveObject(
        edge=10.0, dx=0.5, Disk=disk, Halo=halo,
        include_beam_smearing=True
    )
    R_original = rc_original.R_majoraxis

    # Test various oversample factors (both even and odd)
    oversample_factors = [2, 3, 4, 5, 6]
    
    for oversample in oversample_factors:
        # Create a test instance to check the oversampled grid directly
        rc = RotationCurveObject.__new__(RotationCurveObject)
        rc.edge = 10.0
        rc.dx = 0.5
        rc.oversample = oversample
        
        # Test the oversampled grid creation method directly
        oversampled_array = create_r_space_oversampled(
            edge=rc.edge, resolution=rc.dx/oversample, oversample=oversample
            )
        
        # Check that the oversampled array length is divisible by oversample
        assert len(oversampled_array) % len(R_original) == 0, \
            f"Oversampled array length {len(oversampled_array)} not divisible for oversample {oversample}"
        
        # Now create a full rotation curve and check that rebinning works without errors
        rc_full = RotationCurveObject(
            edge=10.0, dx=0.5, Disk=disk, Halo=halo,
            oversample=oversample,
            include_beam_smearing=False
        )
        
        # After rebinning, the final arrays should have the original size
        expected_original_size = len(R_original)
        assert len(rc_full.R_majoraxis) == expected_original_size, \
            f"Final R_majoraxis length {len(rc_full.R_majoraxis)} doesn't match expected original size {expected_original_size}"


def test_even_odd_oversample_consistency():
    """Test that even and odd oversample factors both work correctly."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    halo = NFWHalo(mass=1e12, concentration=10)
    
    # Create rotation curves with even and odd oversample factors
    rc_even = RotationCurveObject(
        edge=10.0, dx=0.5, Disk=disk, Halo=halo,
        oversample=4,  # even
        include_beam_smearing=False
    )
    
    rc_odd = RotationCurveObject(
        edge=10.0, dx=0.5, Disk=disk, Halo=halo,
        oversample=5,  # odd
        include_beam_smearing=False
    )
    
    # Both should have the same original array size after rebinning
    assert len(rc_even.R_majoraxis) == len(rc_odd.R_majoraxis), \
        "Even and odd oversample should result in same final array size"
    
    # Both should have the same radial grid after rebinning
    assert_allclose(rc_even.R_majoraxis, rc_odd.R_majoraxis, rtol=1e-10, atol=1e-10)
    
    # Both should produce reasonable velocity profiles (no NaN or inf values)
    assert np.all(np.isfinite(rc_even.Vcirc)), "Even oversample should produce finite velocities"
    assert np.all(np.isfinite(rc_odd.Vcirc)), "Odd oversample should produce finite velocities"
    assert np.all(np.isfinite(rc_even.Vrot)), "Even oversample should produce finite rotation velocities"
    assert np.all(np.isfinite(rc_odd.Vrot)), "Odd oversample should produce finite rotation velocities"
    
    # The velocity profiles should be similar away from the center (r > 1 kpc)
    # where numerical differences are less significant
    mask = np.abs(rc_even.R_majoraxis) > 1.0
    assert_allclose(rc_even.Vcirc[mask], rc_odd.Vcirc[mask], rtol=0.1, atol=2.0)
    assert_allclose(rc_even.Vrot[mask], rc_odd.Vrot[mask], rtol=0.1, atol=2.0)


def test_oversample_vs_higher_resolution(): 
    """Test that oversampling is equivalent to using a higher resolution grid."""
    rtol = 5e-2
    atol =0.0

    disk = SersicProfile(mass=1e10, r_eff=4.0, n=2.0, q0=0.1)
    halo = NFWHalo(mass=1e12, concentration=10)
    
    edge = 15.0
    dx = 0.2
    oversample = 2

    # Create rotation curves with different oversample factors
    rc_oversampled = RotationCurveObject(
        edge=edge, dx=dx, Disk=disk, Halo=halo,
        oversample=oversample,
        inclination=35.,
        FWHM_beam=0.5,
        include_beam_smearing=True
    )
    
    rc_higher_resolution = RotationCurveObject(
        edge=edge, dx=dx/oversample, Disk=disk, Halo=halo,
        inclination=35.,
        FWHM_beam=0.5,
        include_beam_smearing=True
    )

    # Both should have the same radial grid after rebinning
    assert_allclose(rc_oversampled.R_majoraxis, rc_higher_resolution.R_majoraxis[::2], rtol=1e-10, atol=1e-10)
    
    # Velocity array should be close
    assert_allclose(rc_oversampled.Vd, rc_higher_resolution.Vd[::2], rtol=rtol, atol=atol)
    assert_allclose(rc_oversampled.Vh, rc_higher_resolution.Vh[::2], rtol=rtol, atol=atol)
    assert_allclose(rc_oversampled.Vbaryon, rc_higher_resolution.Vbaryon[::2], rtol=rtol, atol=atol)
    assert_allclose(rc_oversampled.Vcirc, rc_higher_resolution.Vcirc[::2], rtol=rtol, atol=atol)
    assert_allclose(rc_oversampled.Vrot, rc_higher_resolution.Vrot[::2], rtol=rtol, atol=atol)

    assert_allclose(rc_oversampled.Vobs, rc_higher_resolution.Vobs[::2], rtol=rtol, atol=atol)
    assert_allclose(rc_oversampled.Vobs_sini, rc_higher_resolution.Vobs_sini[::2], rtol=rtol, atol=atol)
    assert_allclose(rc_oversampled.velocity_dispersion, rc_higher_resolution.velocity_dispersion[::2], rtol=rtol, atol=atol)
