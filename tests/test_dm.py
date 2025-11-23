import numpy as np
from numpy.testing import assert_allclose
from astropy.cosmology import Planck18

from RotCurves.dm_halos import (
    NFWHalo,
    alhpaNFWHalo,
    BurkertHalo,
    EinastoHalo,
    DekelZhaoHalo,
)

from RotCurves.const import (
    G_CONST
)


RTOL = 1e-3
ATOL = 0.0

def test_all_halos_match_input_mass():
    """Each halo should recover its input mass at the virial radius."""
    halo_classes = [
        (NFWHalo, {}),
        (alhpaNFWHalo, {"alpha": 0.7}),
        (BurkertHalo, {}),
        (EinastoHalo, {"n": 4.0}),
        (DekelZhaoHalo, {"alpha": 0.6, "g": 3.2, "b": 1.8}),
    ]
    mass = 5e10
    concentration = 8
    for cls, extra_kwargs in halo_classes:
        halo = cls(
            mass=mass,
            concentration=concentration,
            z=0.2,
            virial_overdensity=200,
            **extra_kwargs,
        )
        total_mass = halo.menc(halo.r_vir)
        assert_allclose(
            total_mass,
            mass,
            rtol=1e-6,
            err_msg=f"{cls.__name__} failed to recover the input mass.",
        )


def test_generalized_nfw_recovers_nfw_when_alpha_is_one():
    """alpha-NFW with alpha=1 must reproduce the standard NFW profile."""
    mass = 1.2e11
    concentration = 9
    halo_nfw = NFWHalo(mass=mass, concentration=concentration, z=0.0)
    halo_alpha = alhpaNFWHalo(
        mass=mass,
        concentration=concentration,
        z=0.0,
        alpha=1.0,
    )
    radii = halo_nfw.r_s * np.logspace(-2, 2, 6)
    assert_allclose(
        halo_alpha.density(radii),
        halo_nfw.density(radii),
        rtol=1e-10,
        err_msg="density mismatch between alpha-NFW (alpha=1) and canonical NFW.",
    )
    assert_allclose(
        halo_alpha.menc(radii),
        halo_nfw.menc(radii),
        rtol=1e-10,
        err_msg="enclosed mass mismatch between alpha-NFW (alpha=1) and canonical NFW.",
    )
    assert_allclose(
        halo_alpha.vcirc(radii),
        halo_nfw.vcirc(radii),
        rtol=1e-10,
        err_msg="circular velocity mismatch between alpha-NFW (alpha=1) and canonical NFW.",
    )


def test_burkert_core_remains_finite():
    """Burkert halos should remain cored while NFW halos diverge toward r=0."""
    mass = 8e10
    concentration = 6
    burkert = BurkertHalo(mass=mass, concentration=concentration, z=0.0)
    nfw = NFWHalo(mass=mass, concentration=concentration, z=0.0)
    r_small = 1e-6 * burkert.r_s

    # Burkert approaches the scale density at the origin.
    assert_allclose(
        burkert.density(r_small),
        burkert.scale_density,
        rtol=1e-4,
        err_msg="Burkert core density deviates from expected finite core limit.",
    )

    # NFW diverges and should greatly exceed its own normalization.
    assert (
        nfw.density(r_small) > 1e4 * nfw.scale_density
    ), "NFW profile no longer diverges steeply toward the centre."


def test_standard_nfw_matches_analytic():
    """Verify the NFWHalo reproduces closed-form parameter relations."""
    H0 = 70    # km/s / Mpc
    mass = 1e12
    concentration = 10
    delta = 200
    z = 0

    rho_crit = 135.99294  # Msun / kpc^3
    r_vir_expected = 206.27899  # kpc
    r_s_expected = 20.627899  # kpc
    x = [
        0., 0.52631579, 1.05263158, 1.57894737,
        2.10526316, 2.63157895, 3.15789474, 3.68421053, 
        4.21052632, 4.73684211, 5.26315789, 5.78947368,
        6.31578947, 6.84210526, 7.36842105, 7.89473684, 
        8.42105263, 8.94736842, 9.47368421, 10.,
    ]
    menc_expected = [
        0.00000000e+00, 5.24106904e+10, 1.38569015e+11, 2.25104405e+11,
        3.05703270e+11, 3.79519202e+11, 4.47013200e+11, 5.08918804e+11,
        5.65957545e+11, 6.18765536e+11, 6.67883843e+11, 7.13767553e+11,
        7.56798965e+11, 7.97300259e+11, 8.35544353e+11, 8.71763815e+11,
        9.06158063e+11, 9.38899140e+11, 9.70136351e+11, 1.00000000e+12,
    ]
    vcirc_expected = [
        0., 144.09197161, 165.67154045, 172.40950828, 174.00025718,
        173.40504095, 171.79653174, 169.7090774,  167.40821934, 165.03332389,
        162.65982924, 160.32893269, 158.06273235, 155.87236323, 153.76255247,
        151.73425372, 149.78620984, 147.9158976,  146.1201064,  144.3952952,
    ]
    
    halo = NFWHalo(
        mass=mass,
        concentration=concentration,
        z=z,
        virial_overdensity=delta,
    )

    assert_allclose(
        halo.rho_crit,
        rho_crit,
        rtol=RTOL,
        err_msg="Critical density deviates from analytic expectation.",
    )
    assert_allclose(
        halo.r_vir,
        r_vir_expected,
        rtol=RTOL,
        err_msg="Virial radius deviates from analytic expectation.",
    )
    assert_allclose(
        halo.r_s,
        r_s_expected,
        rtol=RTOL,
        err_msg="Scale radius deviates from analytic expectation.",
    )
    assert_allclose(
        halo.menc(np.asarray(x) * r_s_expected),
        menc_expected,
        rtol=RTOL,
        err_msg="Enclosed mass deviated from analytic expectation."
    )
    assert_allclose(
        halo.vcirc(np.asarray(x) * r_s_expected),
        vcirc_expected,
        rtol=RTOL,
        err_msg="Circular velocity deviated from analytic expectation."
    )

def test_standard_burkert_matches_analytic():
    """Verify the BrukertHalo reproduces closed-form parameter relations."""
    H0 = 70    # km/s / Mpc
    mass = 1e12
    concentration = 10
    delta = 200
    z = 0

    rho_crit = 135.99294  # Msun / kpc^3
    r_vir_expected = 206.27899  # kpc
    r_s_expected = 20.627899  # kpc
    x = [
        0., 0.52631579, 1.05263158, 1.57894737,
        2.10526316, 2.63157895, 3.15789474, 3.68421053, 
        4.21052632, 4.73684211, 5.26315789, 5.78947368,
        6.31578947, 6.84210526, 7.36842105, 7.89473684, 
        8.42105263, 8.94736842, 9.47368421, 10.,
    ]
    menc_expected = [
        0.00000000e+00, 1.87485520e+10, 8.68702269e+10, 1.75166402e+11,
        2.63398284e+11, 3.45370824e+11, 4.20048877e+11, 4.87904509e+11,
        5.49755154e+11, 6.06413804e+11, 6.58595315e+11, 7.06903372e+11,
        7.51841327e+11, 7.93828241e+11, 8.33214202e+11, 8.70293327e+11,
        9.05314329e+11, 9.38488950e+11, 9.69998662e+11, 1.00000000e+12,
    ]
    vcirc_expected = [
        0., 86.1814327,  131.17477367, 152.08788526, 161.51256661,
        165.41986589, 166.53446739, 166.16832392, 164.99451415, 163.37783171,
        161.52478134, 159.55614227, 157.54416308, 155.53260278, 153.54799767,
        151.60622738, 149.71645984, 147.88358304, 146.10973682, 144.3952952,
    ]
    
    halo = BurkertHalo(
        mass=mass,
        concentration=concentration,
        z=z,
        virial_overdensity=delta,
    )

    assert_allclose(
        halo.rho_crit,
        rho_crit,
        rtol=RTOL,
        err_msg="Critical density deviates from analytic expectation.",
    )
    assert_allclose(
        halo.r_vir,
        r_vir_expected,
        rtol=RTOL,
        err_msg="Virial radius deviates from analytic expectation.",
    )
    assert_allclose(
        halo.r_s,
        r_s_expected,
        rtol=RTOL,
        err_msg="Scale radius deviates from analytic expectation.",
    )
    assert_allclose(
        halo.menc(np.asarray(x) * r_s_expected),
        menc_expected,
        rtol=RTOL,
        err_msg="Enclosed mass deviated from analytic expectation."
    )
    assert_allclose(
        halo.vcirc(np.asarray(x) * r_s_expected),
        vcirc_expected,
        rtol=RTOL,
        err_msg="Circular velocity deviated from analytic expectation."
    )
