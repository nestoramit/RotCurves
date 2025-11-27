import numpy as np
from numpy.testing import assert_allclose
from scipy.special import gammaincinv
import pytest
from itertools import product
import random
from RotCurves.baryons import (
    SersicProfile,
    FreemanDisk,
    GaussianRingProfile,
    LightSersicProfile,
    LightFreemanDiskProfile,
    LightGaussianRingProfile,
)


# Test tolerances
RTOL = 1e-3
ATOL = 0.0

# Noordermeer params to test
noordermeer_params_allowed = list(product(
    [0.5, 1.0, 4.0],
    [0.0, 0.2, 1.0]
))

def get_random_sersic_params():
    """Randomly select (n, q0) from noordermeer_params_allowed."""
    return random.choice(noordermeer_params_allowed)

def test_freeman_sersic_equivalence():
    """Test that FreemanDisk and SersicProfile with n=1 produce identical results.
    
    An exponential disk is mathematically equivalent to a Sérsic profile with n=1.
    This test verifies that both implementations produce the same surface density,
    enclosed mass, and circular velocity profiles.
    """
    mass = 1e11  # M_sun
    r_eff = 5.0  # kpc
    n, q0 = 1.0, 0.0  # Required for equivalence test
    radii = np.linspace(0, 4*r_eff, num=100)  # kpc
    
    sersic = SersicProfile(mass=mass, r_eff=r_eff, n=n, q0=q0)
    freeman = FreemanDisk(mass=mass, r_eff=r_eff)
    
    sersic_surface_density = sersic.surface_density(radii)
    freeman_surface_density = freeman.surface_density(radii)
    assert_allclose(
        sersic_surface_density, 
        freeman_surface_density, 
        rtol=1e-3, 
        equal_nan=True,
        err_msg="Surface density profiles differ"
    )
    
    sersic_menc = sersic.menc(radii)
    freeman_menc = freeman.menc(radii)
    assert_allclose(
        sersic_menc, 
        freeman_menc, 
        rtol=1e-3, 
        equal_nan=True,
        err_msg="Enclosed mass profiles differ"
    )
    
    sersic_vcirc = sersic.vcirc(radii)
    freeman_vcirc = freeman.vcirc(radii)
    assert_allclose(
        sersic_vcirc, 
        freeman_vcirc, 
        rtol=1e-3, 
        equal_nan=True,
        err_msg="Circular velocity profiles differ"
    )


def test_sersic_r_eff_vs_r_s():
    """Test that SersicProfile can be initialized with either r_eff or r_s."""
    mass = 1e10
    n = 1.2
    q0 = 0.1
    r_eff = 3.0
    r_s = 1.248558842
    
    prof1 = SersicProfile(mass=mass, r_eff=r_eff, n=n, q0=q0)
    prof2 = SersicProfile(mass=mass, r_s=r_s, n=n, q0=q0)
    
    assert_allclose(
        prof1.r_s, 
        r_s, 
        rtol=RTOL, 
        err_msg="r_s should match analytic"
    )
    assert_allclose(
        prof1.r_s, 
        prof2.r_s, 
        rtol=RTOL, 
        err_msg="r_s should match calculation"
    )
    assert_allclose(
        r_eff, 
        prof2.r_eff, 
        rtol=RTOL, 
        err_msg="r_eff should match analytic"
    )
    assert_allclose(
        prof1.r_eff, 
        prof2.r_eff, 
        rtol=RTOL, 
        err_msg="r_eff should match calculation"
    )
    
    radii = np.linspace(0.1, 10, num=50)
    assert_allclose(
        prof1.surface_density(radii),
        prof2.surface_density(radii),
        rtol=RTOL,
        err_msg="Surface density should be identical"
    )
    assert_allclose(
        prof1.menc(radii),
        prof2.menc(radii),
        rtol=RTOL,
        err_msg="Enclosed mass should be identical"
    )


def test_freeman_r_eff_vs_r_s():
    """Test that FreemanDisk can be initialized with either r_eff or r_s."""
    mass = 1e10
    r_eff = 2.0
    r_s = 1.191648694755
    
    prof1 = FreemanDisk(mass=mass, r_eff=r_eff)
    prof2 = FreemanDisk(mass=mass, r_s=r_s)
    
    assert_allclose(
        r_eff, 
        prof2.r_eff, 
        rtol=RTOL, 
        err_msg="r_eff should match analytic"
    )
    assert_allclose(
        prof1.r_eff, 
        prof2.r_eff, 
        rtol=RTOL, 
        err_msg="r_eff should match calculation"
    )
    assert_allclose(
        prof1.r_s,
        r_s,
        rtol=RTOL,
        err_msg="r_s should match analytic"
    )
    assert_allclose(
        prof1.r_s,
        prof2.r_s,
        rtol=RTOL,
        err_msg="r_s should match calculation"
    )

    radii = np.linspace(0.1, 10, num=50)
    assert_allclose(
        prof1.surface_density(radii),
        prof2.surface_density(radii),
        rtol=RTOL,
        err_msg="Surface density should be identical"
    )
    assert_allclose(
        prof1.menc(radii),
        prof2.menc(radii),
        rtol=RTOL,
        err_msg="Enclosed mass should be identical"
    )


def test_sersic_sersic_b():
    """Test Sersic b parameter calculation for different n values."""
    test_cases = [
        (1.0, 1.6783469900),  # Exponential disk
        # (2.0, 3.6720607488),
        # (3.0, 5.6701611887),
        # (4.0, 7.6692494425),  # de Vaucouleurs
    ]
    
    for n, expected_b_approx in test_cases:
        _, q0 = get_random_sersic_params()
        prof = SersicProfile(mass=1e10, r_s=1.0, n=n, q0=q0)
        b = prof.sersic_b()
        # Check that b is reasonable (should be positive and finite)
        assert b > 0, f"b should be positive for n={n}"
        assert np.isfinite(b), f"b should be finite for n={n}"
        # For n=1, b should be close to 1.678
        assert_allclose(
            b, 
            expected_b_approx, 
            rtol=RTOL, 
            err_msg=f"Sersic b different from analytical"
        )


def test_sersic_surface_density():
    """Test that Sersic surface density decreases with radius."""
    mass = 1e10
    r_s = 2.0
    n, q0 = get_random_sersic_params()
    
    prof = SersicProfile(mass=mass, r_s=r_s, n=n, q0=q0)
    radii = np.linspace(0.1, 20, num=100)
    sigma = prof.surface_density(radii)
    
    # Surface density should be monotonically decreasing
    assert np.all(np.diff(sigma) <= 0), "Surface density should decrease with radius"


def test_sersic_menc():
    """Test that enclosed mass increases with radius."""
    mass = 1e10
    r_s = 2.0
    n, q0 = get_random_sersic_params()
    
    prof = SersicProfile(mass=mass, r_s=r_s, n=n, q0=q0)
    radii = np.linspace(0.1, 20, num=100)
    menc = prof.menc(radii)
    
    # Enclosed mass should be monotonically increasing
    assert np.all(np.diff(menc) >= 0), "Enclosed mass should increase with radius"
    
    # At large radius, enclosed mass should approach total mass
    large_r = np.inf
    assert_allclose(
        prof.menc(large_r),
        mass,
        rtol=1e-2,
        err_msg="Enclosed mass at large radius should approach total mass"
    )

def test_sersic_surface_density_function():
    """Test Sérsic dimensionless surface density function."""
    mass=1e10
    r_s=2.0
    n, q0 = get_random_sersic_params()
    
    prof = SersicProfile(mass=mass, r_s=r_s, n=n, q0=q0)
    
    x = np.linspace(0, 5, num=20)
    expected = np.exp(-(x)**(1/n))
    
    assert_allclose(
        prof.surface_density_function(x), 
        expected, 
        rtol=RTOL, 
        err_msg="Surface density function should be exp(-x)"
    )


def test_freeman_surface_density_function():
    """Test Freeman disk dimensionless surface density function."""
    mass=1e10
    r_eff=2.0
    
    prof = FreemanDisk(mass=mass, r_eff=r_eff)
    
    x = np.linspace(0, 5, num=20)
    expected = np.exp(-x)
    
    assert_allclose(
        prof.surface_density_function(x), 
        expected, 
        rtol=RTOL, 
        err_msg="Surface density function should be exp(-x)"
    )


def test_freeman_menc_dimless():
    """Test Freeman disk dimensionless enclosed mass."""
    prof = FreemanDisk(mass=1e10, r_s=2.0)
    
    x = np.array([0.0, 1.0, 2.0, 5.0])
    result = prof.menc_dimless(x)
    
    # At x=0, enclosed mass should be 0
    assert_allclose(result[0], 0.0, atol=1e-10, err_msg="Enclosed mass at x=0 should be 0")
    
    # At large x, enclosed mass should approach 1 (dimensionless)
    large_x = 100.0
    assert_allclose(
        prof.menc_dimless(large_x),
        1.0,
        rtol=1e-2,
        err_msg="Enclosed mass at large x should approach 1"
    )


def test_freeman_vcirc2_dimless_at_origin():
    """Test that Freeman circular velocity squared is zero at origin."""
    prof = FreemanDisk(mass=1e10, r_s=2.0)
    
    # At x=0, vcirc^2 should be 0
    v2_at_origin = prof.vcirc2_dimless(0.0)
    assert_allclose(v2_at_origin, 0.0, atol=1e-6, err_msg="Circular velocity squared at origin should be 0")


def test_gaussian_ring_parameter_solver():
    """Test GaussianRing parameter solver with different input combinations."""
    mass = 1e9
    r_s = 5.0
    h = 2.0
    
    # Test initialization with r_s and h
    ring1 = GaussianRingProfile(mass=mass, r_s=r_s, h=h)
    
    # Test initialization with r_s and FWHM_ring
    FWHM_ring = ring1.FWHM_ring
    ring2 = GaussianRingProfile(mass=mass, r_s=r_s, FWHM_ring=FWHM_ring)
    
    # Test initialization with h and sigma_ring
    sigma_ring = ring1.sigma_ring
    ring3 = GaussianRingProfile(mass=mass, h=h, sigma_ring=sigma_ring)
    
    # All should have the same parameters
    assert_allclose(ring1.r_s, ring2.r_s, rtol=1e-6, err_msg="r_s should match")
    assert_allclose(ring1.r_s, ring3.r_s, rtol=1e-6, err_msg="r_s should match")
    assert_allclose(ring1.h, ring2.h, rtol=1e-6, err_msg="h should match")
    assert_allclose(ring1.h, ring3.h, rtol=1e-6, err_msg="h should match")


def test_gaussian_ring_surface_density_peak():
    """Test that Gaussian ring surface density peaks at r_s."""
    mass = 1e9
    r_s = 5.0
    h = 2.0
    
    ring = GaussianRingProfile(mass=mass, r_s=r_s, h=h)
    
    # Test around the peak
    radii = np.linspace(0.5*r_s, 1.5*r_s, num=100)
    sigma = ring.surface_density(radii)
    
    # Find the peak
    peak_idx = np.argmax(sigma)
    peak_radius = radii[peak_idx]
    
    # Peak should be close to r_s
    assert_allclose(
        peak_radius,
        r_s,
        rtol=0.1,
        err_msg="Surface density should peak at r_s"
    )


def test_gaussian_ring_menc_increases():
    """Test that Gaussian ring enclosed mass increases with radius."""
    mass = 1e9
    r_s = 5.0
    h = 2.0
    
    ring = GaussianRingProfile(mass=mass, r_s=r_s, h=h)
    radii = np.linspace(0.1, 2*r_s, num=100)
    menc = ring.menc(radii)
    
    # Enclosed mass should be monotonically increasing
    assert np.all(np.diff(menc) >= 0), "Enclosed mass should increase with radius"


def test_gaussian_ring_mass_normalization():
    """Test that Gaussian ring total mass is correct."""
    mass = 1e9
    r_s = 5.0
    h = 2.0
    
    ring = GaussianRingProfile(mass=mass, r_s=r_s, h=h)
    
    # At very large radius, enclosed mass should approach total mass
    large_r = 100 * r_s
    assert_allclose(
        ring.menc(large_r),
        mass,
        rtol=1e-2,
        err_msg="Enclosed mass at large radius should approach total mass"
    )


def test_light_profiles_zero_mass():
    """Test that light profiles have zero mass."""
    light_sersic = LightSersicProfile(r_s=2.0, n=1.0)
    light_freeman = LightFreemanDiskProfile(r_s=2.0)
    light_ring = LightGaussianRingProfile(r_s=5.0, h=2.0)
    
    assert light_sersic.mass == 0.0, "LightSersicProfile should have zero mass"
    assert light_freeman.mass == 0.0, "LightFreemanDiskProfile should have zero mass"
    assert light_ring.mass == 0.0, "LightGaussianRingProfile should have zero mass"


def test_light_profiles_surface_density_zero():
    """Test that light profiles with zero mass have zero surface density."""
    light_sersic = LightSersicProfile(r_s=2.0, n=1.0)
    light_freeman = LightFreemanDiskProfile(r_s=2.0)
    
    radii = np.linspace(0.1, 10, num=50)
    
    # Surface density should be zero everywhere
    assert_allclose(
        light_sersic.surface_density(radii),
        0.0,
        atol=1e-10,
        err_msg="Light profile surface density should be zero"
    )
    assert_allclose(
        light_freeman.surface_density(radii),
        0.0,
        atol=1e-10,
        err_msg="Light profile surface density should be zero"
    )


def test_light_profiles_circular_velocity_zero():
    """Test that light profiles with zero mass have zero circular velocity."""
    light_sersic = LightSersicProfile(r_s=2.0, n=1.0)
    light_freeman = LightFreemanDiskProfile(r_s=2.0)
    
    radii = np.linspace(0.1, 10, num=50)
    
    # Circular velocity should be zero everywhere
    assert_allclose(
        light_sersic.vcirc(radii),
        0.0,
        atol=1e-10,
        err_msg="Light profile circular velocity should be zero"
    )
    assert_allclose(
        light_freeman.vcirc(radii),
        0.0,
        atol=1e-10,
        err_msg="Light profile circular velocity should be zero"
    )


def test_sersic_dlnrho_dlnr():
    """Test Sersic logarithmic density slope."""
    mass = 1e10
    r_s = 2.0
    n, q0 = get_random_sersic_params()
    
    prof = SersicProfile(mass=mass, r_s=r_s, n=n, q0=q0)
    radii = np.array([1.0, 2.0, 5.0])
    dlnrho = prof.dlnrho_dlnr(radii)
    
    # Logarithmic slope should be negative (density decreases with radius)
    assert np.all(dlnrho < 0), "Logarithmic slope should be negative"
    
    # At larger radii, slope should be more negative
    assert dlnrho[2] < dlnrho[0], "Slope should become more negative at larger radii"


def test_freeman_dlnrho_dlnr():
    """Test Freeman disk logarithmic density slope."""
    mass = 1e10
    r_s = 2.0
    
    prof = FreemanDisk(mass=mass, r_s=r_s)
    radii = np.array([1.0, 2.0, 5.0])
    dlnrho = prof.dlnrho_dlnr(radii)
    
    # For exponential disk, dlnrho/dlnr = -x
    x = radii / r_s
    expected = -x
    
    assert_allclose(
        dlnrho,
        expected,
        rtol=1e-6,
        err_msg="Logarithmic slope should be -x for exponential disk"
    )


def test_gaussian_ring_dlnrho_dlnr():
    """Test Gaussian ring logarithmic density slope."""
    mass = 1e9
    r_s = 5.0
    h = 2.0
    
    ring = GaussianRingProfile(mass=mass, r_s=r_s, h=h)
    radii = np.array([0.5*r_s, r_s, 1.5*r_s])
    dlnrho = ring.dlnrho_dlnr(radii)
    
    # At r < r_s, slope should be positive (density increases)
    # At r > r_s, slope should be negative (density decreases)
    # At r = r_s, slope should be zero (peak)
    assert dlnrho[0] > 0, "Slope should be positive inside peak"
    assert_allclose(dlnrho[1], 0.0, atol=1e-6, err_msg="Slope should be zero at peak")
    assert dlnrho[2] < 0, "Slope should be negative outside peak"


def test_gaussian_ring_initialization_errors():
    """Test that GaussianRing raises errors for invalid initialization."""
    mass = 1e9
    
    # Should raise error if neither r_s nor h is provided
    with pytest.raises(ValueError, match="Either r_s or h must be provided"):
        GaussianRingProfile(mass=mass, FWHM_ring=2.0, sigma_ring=1.0)


def test_sersic_vcirc_lookup_vs_no_lookup():
    """Test that SersicProfile vcirc is the same with or without lookup tables."""
    mass = 1e9
    r_eff = 3.0
    n, q0 = get_random_sersic_params()

    prof_lookup = SersicProfile(mass=mass, r_eff=r_eff, n=n, q0=q0, lookup=True)
    prof_no_lookup = SersicProfile(mass=mass, r_eff=r_eff, n=n, q0=q0, lookup=False)
    
    radii = np.linspace(0.1, 5*r_eff, num=50)
    
    vcirc_lookup = prof_lookup.vcirc(radii)
    vcirc_no_lookup = prof_no_lookup.vcirc(radii)
    
    assert_allclose(
        vcirc_lookup,
        vcirc_no_lookup,
        rtol=RTOL,
        equal_nan=True,
        err_msg="Circular velocity should be the same with or without lookup tables"
    )


def test_gaussian_ring_vcirc_lookup_vs_no_lookup():
    """Test that GaussianRingProfile vcirc is the same with or without lookup tables."""
    mass = 1e9
    r_s = 5.0
    h = 1.0

    # Create profiles with and without lookup tables
    ring_lookup = GaussianRingProfile(mass=mass, r_s=r_s, h=h, lookup=True)
    ring_no_lookup = GaussianRingProfile(mass=mass, r_s=r_s, h=h, lookup=False)

    # Test at various radii
    radii = np.linspace(0.1, 5*r_s, num=50)

    vcirc_lookup = ring_lookup.vcirc(radii)
    vcirc_no_lookup = ring_no_lookup.vcirc(radii)

    # Compare results - they should be similar
    assert_allclose(
        vcirc_lookup,
        vcirc_no_lookup,
        rtol=RTOL,
        equal_nan=True,
        err_msg="Circular velocity should be the same with or without lookup tables"
    )