import numpy as np
from numpy.testing import assert_allclose
from RotCurves.baryons import SersicProfile, FreemanDisk


def test_freeman_sersic_equivalence():
    """Test that FreemanDisk and SersicProfile with n=1 produce identical results.
    
    An exponential disk is mathematically equivalent to a Sérsic profile with n=1.
    This test verifies that both implementations produce the same surface density,
    enclosed mass, and circular velocity profiles.
    """
    
    mass = 1e11  # M_sun
    r_eff = 5.0  # kpc
    n = 1.0
    q0 = 0.0
    radii = np.linspace(0, 20, num=100)  # kpc
    rtol = 1e-3
    atol = 0.0
    
    sersic = SersicProfile(mass=mass, r_eff=r_eff, n=n, q0=q0)
    freeman = FreemanDisk(mass=mass, r_eff=r_eff)
    
    sersic_surface_density = sersic.surface_density(radii)
    freeman_surface_density = freeman.surface_density(radii)
    assert_allclose(
        sersic_surface_density, 
        freeman_surface_density, 
        rtol=rtol, 
        atol=atol, 
        equal_nan=True,
        err_msg="Surface density profiles differ"
    )
    
    sersic_menc = sersic.menc(radii)
    freeman_menc = freeman.menc(radii)
    assert_allclose(
        sersic_menc, 
        freeman_menc, 
        rtol=rtol, 
        atol=atol, 
        equal_nan=True,
        err_msg="Enclosed mass profiles differ"
    )
    
    sersic_vcirc = sersic.vcirc(radii)
    freeman_vcirc = freeman.vcirc(radii)
    assert_allclose(
        sersic_vcirc, 
        freeman_vcirc, 
        rtol=rtol, 
        atol=atol, 
        equal_nan=True,
        err_msg="Circular velocity profiles differ"
    )
