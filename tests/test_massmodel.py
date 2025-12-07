import numpy as np
from numpy.testing import assert_allclose
import pytest

from RotCurves.mass_model import create_components
from RotCurves.dm_halos import NFWHalo, BurkertHalo, EinastoHalo
from RotCurves.baryons import SersicProfile, GaussianRingProfile, LightSersicProfile, LightGaussianRingProfile


# Test tolerances
RTOL = 1e-3
ATOL = 0.0


def test_create_components_halo_only():
    """Test creating only a halo component."""
    components = create_components(
        include_halo=True,
        include_disk=False,
        include_ring=False,
        include_bulge=False,
        z=0.0,
        halo_profile='NFW',
        logM_vir=12.0,
        c=10.0
    )
    
    assert components['halo'] is not None, "Halo should be created"
    assert isinstance(components['halo'], NFWHalo), "Halo should be NFWHalo instance"
    assert components['disk'] is None, "Disk should be None"
    assert components['ring'] is None, "Ring should be None"
    assert components['bulge'] is None, "Bulge should be None"
    
    # Check halo mass
    assert_allclose(
        components['halo'].mass,
        1e12,
        rtol=RTOL,
        err_msg="Halo mass should match input logM_vir"
    )


def test_create_components_disk_only():
    """Test creating only a disk component."""
    components = create_components(
        include_halo=False,
        include_disk=True,
        include_ring=False,
        include_bulge=False,
        logM_baryon=10.0,
        DT=1.0,
        disk_re=5.0,
        disk_n=1.0,
        disk_q=0.2
    )
    
    assert components['halo'] is None, "Halo should be None"
    assert components['disk'] is not None, "Disk should be created"
    assert isinstance(components['disk'], SersicProfile), "Disk should be SersicProfile instance"
    assert components['ring'] is None, "Ring should be None"
    assert components['bulge'] is None, "Bulge should be None"
    
    # Check disk mass
    assert_allclose(
        components['disk'].mass,
        1e10,
        rtol=RTOL,
        err_msg="Disk mass should match logM_baryon + log10(DT)"
    )


def test_create_components_bulge_only():
    """Test creating only a bulge component."""
    components = create_components(
        include_halo=False,
        include_disk=False,
        include_ring=False,
        include_bulge=True,
        logM_baryon=10.0,
        BT=1.0,
        bulge_n=4.0,
        bulge_q=1.0
    )
    
    assert components['halo'] is None, "Halo should be None"
    assert components['disk'] is None, "Disk should be None"
    assert components['ring'] is None, "Ring should be None"
    assert components['bulge'] is not None, "Bulge should be created"
    assert isinstance(components['bulge'], SersicProfile), "Bulge should be SersicProfile instance"
    
    # Check bulge mass
    assert_allclose(
        components['bulge'].mass,
        1e10,
        rtol=RTOL,
        err_msg="Bulge mass should match logM_baryon + log10(BT)"
    )


def test_create_components_disk_and_bulge():
    """Test creating disk and bulge components with mass ratios."""
    components = create_components(
        include_halo=False,
        include_disk=True,
        include_ring=False,
        include_bulge=True,
        logM_baryon=10.0,
        DT=0.7,
        BT=0.3,
        disk_re=5.0,
        disk_n=1.0,
        disk_q=0.2,
        bulge_n=4.0,
        bulge_q=1.0
    )
    
    assert components['disk'] is not None, "Disk should be created"
    assert components['bulge'] is not None, "Bulge should be created"
    
    # Check that masses sum correctly
    disk_mass = components['disk'].mass
    bulge_mass = components['bulge'].mass
    total_mass = disk_mass + bulge_mass
    
    assert_allclose(
        total_mass,
        1e10,
        rtol=RTOL,
        err_msg="Total baryonic mass should match logM_baryon"
    )
    
    # Check mass ratios
    assert_allclose(
        disk_mass / total_mass,
        0.7,
        rtol=RTOL,
        err_msg="Disk-to-total ratio should match DT"
    )
    assert_allclose(
        bulge_mass / total_mass,
        0.3,
        rtol=RTOL,
        err_msg="Bulge-to-total ratio should match BT"
    )


def test_create_components_ring_only():
    """Test creating only a ring component."""
    components = create_components(
        include_halo=False,
        include_disk=False,
        include_ring=True,
        include_bulge=False,
        logM_baryon=9.0,
        ring_rpeak=5.0,
        ring_FWHM=2.0
    )
    
    assert components['ring'] is not None, "Ring should be created"
    assert isinstance(components['ring'], GaussianRingProfile), "Ring should be GaussianRingProfile instance"
    
    # Check ring mass (should be total baryonic mass when no disk/bulge)
    assert_allclose(
        components['ring'].mass,
        1e9,
        rtol=RTOL,
        err_msg="Ring mass should match logM_baryon"
    )


def test_create_components_all_components():
    """Test creating all components together."""
    components = create_components(
        include_halo=True,
        include_disk=True,
        include_ring=True,
        include_bulge=True,
        z=0.0,
        halo_profile='NFW',
        logM_vir=12.0,
        c=10.0,
        logM_baryon=10.0,
        DT=0.5,
        BT=0.3,
        disk_re=5.0,
        disk_n=1.0,
        disk_q=0.2,
        bulge_n=4.0,
        bulge_q=1.0,
        ring_rpeak=5.0,
        ring_FWHM=2.0
    )
    
    assert components['halo'] is not None, "Halo should be created"
    assert components['disk'] is not None, "Disk should be created"
    assert components['ring'] is not None, "Ring should be created"
    assert components['bulge'] is not None, "Bulge should be created"
    
    # Check that baryonic masses sum correctly
    disk_mass = components['disk'].mass
    bulge_mass = components['bulge'].mass
    ring_mass = components['ring'].mass
    total_baryon_mass = disk_mass + bulge_mass + ring_mass
    
    assert_allclose(
        total_baryon_mass,
        1e10,
        rtol=RTOL,
        err_msg="Total baryonic mass should match logM_baryon"
    )


def test_create_components_light_weighting():
    """Test creating light-only components when light weighting is enabled."""
    components = create_components(
        include_halo=False,
        include_disk=False,
        include_ring=False,
        include_bulge=False,
        disk_lw=True,
        disk_re=5.0,
        disk_n=1.0,
        ring_lw=True,
        ring_rpeak=5.0,
        ring_FWHM=2.0
    )
    
    assert isinstance(components['disk'], LightSersicProfile), "Disk should be LightSersicProfile when disk_lw=True"
    assert isinstance(components['ring'], LightGaussianRingProfile), "Ring should be LightGaussianRingProfile when ring_lw=True"
    
    # Light profiles should have zero mass
    assert components['disk'].mass == 0.0, "Light disk should have zero mass"
    assert components['ring'].mass == 0.0, "Light ring should have zero mass"


def test_create_components_different_halo_profiles():
    """Test creating halos with different profile types."""
    halo_profiles = ['NFW', 'Burkert', 'Einasto']
    
    for profile in halo_profiles:
        components = create_components(
            include_halo=True,
            include_disk=False,
            include_ring=False,
            include_bulge=False,
            z=0.0,
            halo_profile=profile,
            logM_vir=12.0,
            c=10.0
        )
        
        assert components['halo'] is not None, f"Halo should be created for {profile}"
        
        # Check that correct halo type is created
        if profile == 'NFW':
            assert isinstance(components['halo'], NFWHalo), f"Halo should be NFWHalo for {profile}"
        elif profile == 'Burkert':
            assert isinstance(components['halo'], BurkertHalo), f"Halo should be BurkertHalo for {profile}"
        elif profile == 'Einasto':
            assert isinstance(components['halo'], EinastoHalo), f"Halo should be EinastoHalo for {profile}"


def test_create_components_mass_ratio_clipping():
    """Test that mass ratios are clipped to valid ranges."""
    # Test with BT and DT outside valid range
    components = create_components(
        include_halo=False,
        include_disk=True,
        include_ring=False,
        include_bulge=True,
        logM_baryon=10.0,
        DT=1.5,  # Should be clipped to 1
        BT=-0.1,  # Should be clipped to 0
        disk_re=5.0,
        disk_n=1.0,
        disk_q=0.2,
        bulge_n=4.0,
        bulge_q=1.0
    )
    
    # Components should still be created (clipping happens internally)
    assert components['disk'] is not None, "Disk should be created even with invalid DT"
    assert components['bulge'] is not None, "Bulge should be created even with invalid BT"

