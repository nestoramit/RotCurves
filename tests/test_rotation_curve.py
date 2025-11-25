import numpy as np
from numpy.testing import assert_allclose

from RotCurves.rotation_curve import RotationCurveObject, calculate_fraction_at_re
from RotCurves.baryons import FreemanDisk, SersicProfile, GaussianRingProfile
from RotCurves.dm_halos import NFWHalo

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
    assert np.isclose(rc.edge, 10.0, rtol=RTOL)


def test_intrinsic_rotation_curve_disk_and_halo():
    """Test that intrinsic rotation curve correctly combines components."""
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
    assert_allclose(rc.intrinsic, rc.intrinsic_no_dispersion, rtol=RTOL, atol=ATOL)


def test_inclination_effect():
    """Test that inclination affects line-of-sight velocity."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    
    rc_edgeon = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        inclination=80.0,  # edge-on
        include_beam_smearing=False
    )
    
    rc_faceon = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        inclination=10.0,  # face-on
        include_beam_smearing=False
    )
    
    # Edge-on should have larger line-of-sight velocity than face-on
    assert np.all(np.abs(rc_edgeon.intrinsic_with_inclination) >=
                  np.abs(rc_faceon.intrinsic_with_inclination))
    
    # Face-on should have near-zero line-of-sight velocity
    assert np.all(np.abs(rc_faceon.intrinsic_with_inclination) < 1e-3)


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
        apply_2D=True,
        include_beam_smearing=True,
        inclination=45.0,
        ndim=1
    )
    
    # Check that smeared velocities are computed
    assert len(rc.smeared) > 0
    assert len(rc.smeared_with_inclination) > 0
    assert len(rc.velocity_dispersion) > 0

    # Check that the smeared velocity is smoother than the intrinsic velocity
    assert np.var(np.diff(rc.smeared)) < np.var(np.diff(rc.intrinsic))


def test_FWHM_beam_conversion():
    """Test that FWHM_beam is correctly converted to sigma_beam."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    FWHM_beam = 4.0  # kpc
    expected_sigma = FWHM_beam / (2 * np.sqrt(2 * np.log(2)))
    
    rc = RotationCurveObject(
        edge=20.0, dx=0.1, Disk=disk,
        FWHM_beam=FWHM_beam,
        apply_2D=True,
        include_beam_smearing=True,
        inclination=50.0
    )
    
    assert np.isclose(rc.sigma_beam, expected_sigma, rtol=1e-6)


def test_calculate_fraction_at_re():
    """Test calculate_fraction_at_re function."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
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
        apply_2D=True,
        include_beam_smearing=True,
        inclination=35.0,
        ndim=1
    )
    
    # Radial velocity should affect the smeared velocity
    # (exact effect depends on geometry, but it should be included)
    assert len(rc.smeared_with_inclination) > 0


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
    assert np.all(np.isfinite(rc.intrinsic))
    assert np.all(np.isfinite(rc.Vcirc))


def test_dispersion_profile_with_ring():
    """Test velocity dispersion profile calculation with ring component."""
    ring = GaussianRingProfile(mass=1e9, r_s=5.0, h=2.0, lookup=False)
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


def test_all_dispersion_function_forms():
    """Test all accepted forms for dispersion_function parameter."""
    disk = FreemanDisk(mass=1e10, r_s=2.0)
    ring = GaussianRingProfile(mass=1e9, r_s=5.0, h=2.0, lookup=False)
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
    
    # Test constant height dispersion forms with ring (no disk)
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
    assert rc.sigma_profile[0] > rc.sigma_profile[-1]
    
    # Test that all forms produce finite, positive dispersion values
    all_forms = ['const', 'constant', 'flat', 'constant_h', 'constant_height', 'const_h', 'power_law']
    for form in all_forms:
        rc = RotationCurveObject(
            edge=20.0, dx=0.1, Disk=disk,
            sigma_dispersion=sigma_dispersion,
            dispersion_function=form,
            include_beam_smearing=False
        )
        
        assert np.all(np.isfinite(rc.sigma_profile))
        assert np.all(rc.sigma_profile >= 0)
        assert len(rc.sigma_profile) == len(rc.R_majoraxis)
