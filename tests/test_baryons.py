import numpy as np
from RotCurves.baryons import SersicProfile, FreemanDisk


def test_freeman_sersic():
    r"""
    Test that a FreemanDisk and SersicProfile with n=1 are equivalent.

    This function verifies that an exponential disk (:class:`FreemanDisk`) and
    a Sérsic profile with :math:`n=1` (:class:`SersicProfile`) produce identical
    results for surface density, enclosed mass, and circular velocity.

    The test uses a mass of :math:`10^{11} M_\odot` and an effective radius of
    5 kpc, comparing the profiles at 100 radii from 0 to 20 kpc.

    Raises
    ------
    AssertionError
        If the profiles differ by more than the specified tolerance.

    Notes
    -----
    The exponential disk is mathematically equivalent to a Sérsic profile with
    :math:`n=1`, so this test serves as a consistency check between the two
    implementations.
    """
    M = 1e11
    Reff = 5.
    n = 1.0
    q0 = 0

    r = np.linspace(0, 20, num=100)
    sersic = SersicProfile(mass=M, r_eff=Reff, n=n, q0=q0)
    freeman = FreemanDisk(mass=M, r_eff=Reff)

    rtol = 1e-3
    atol = 0.

    assert np.allclose(sersic.surface_density(r), freeman.surface_density(r), rtol=rtol, atol=atol, equal_nan=True)
    assert np.allclose(sersic.menc(r), freeman.menc(r), rtol=rtol, atol=atol, equal_nan=True)
    assert np.allclose(sersic.vcirc(r), freeman.vcirc(r), rtol=rtol, atol=atol, equal_nan=True)

    print("Freeman and Sersic n=1 profiles are equal within the tolerance limits.")
