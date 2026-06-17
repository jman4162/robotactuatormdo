"""Scaling/sanity tests for first-order radial PMSM sizing (no absolute datasheet assertions)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from robotactuatormdo.geometry.radial_flux import RadialPMGeometry, size_radial_pm
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42, MagnetMaterial

BASE = RadialPMGeometry(
    air_gap_radius_m=0.045,
    stack_length_m=0.040,
    outer_radius_m=0.070,
    pole_pairs=7,
    slots=24,
    magnet_thickness_m=0.004,
    air_gap_m=0.0008,
    turns_per_phase=40,
)


def size(geom, magnet=NDFEB_N42):
    return size_radial_pm(geom, magnet, M250_35A, COPPER)


def test_all_outputs_positive():
    p = size(BASE)
    for name in ("flux_linkage_wb", "r_s_ohm_20", "l_d_h", "max_phase_current_a_rms",
                 "total_mass_kg", "rotor_inertia_kg_m2", "iron_mass_kg"):
        assert getattr(p, name) > 0.0


def test_flux_linkage_scales_with_remanence():
    weak = MagnetMaterial("weak", br_t=0.65, temp_coeff_br_per_c=-0.0012, recoil_mu_r=1.05,
                          max_op_temp_c=80.0, density_kg_m3=7500.0, resistivity_ohm_m=1.4e-6)
    psi_strong = size(BASE, NDFEB_N42).flux_linkage_wb
    psi_weak = size(BASE, weak).flux_linkage_wb
    assert psi_strong > psi_weak  # higher Br -> more flux


def test_torque_capability_grows_with_radius():
    # Scale the whole machine radially (preserve slot depth): torque capability ~ psi_m * I_max
    # should grow with a larger bore.
    small = size(replace(BASE, air_gap_radius_m=0.030, outer_radius_m=0.055))
    large = size(replace(BASE, air_gap_radius_m=0.060, outer_radius_m=0.085))
    cap_small = small.flux_linkage_wb * small.max_phase_current_a_rms
    cap_large = large.flux_linkage_wb * large.max_phase_current_a_rms
    assert cap_large > cap_small


def test_mass_grows_with_stack_length():
    short = size(replace(BASE, stack_length_m=0.020))
    long = size(replace(BASE, stack_length_m=0.060))
    assert long.total_mass_kg > short.total_mass_kg


def test_more_poles_reduce_flux_per_pole():
    p_low = size(replace(BASE, pole_pairs=4))
    p_high = size(replace(BASE, pole_pairs=14))
    # psi_m ~ 1/p (flux per pole falls as pole count rises), so psi_m * p is invariant.
    assert p_high.flux_linkage_wb < p_low.flux_linkage_wb
    assert p_low.flux_linkage_wb * p_low.pole_pairs == pytest.approx(
        p_high.flux_linkage_wb * p_high.pole_pairs, rel=1e-6
    )
    # both designs stay below saturation (derived yoke/teeth).
    assert max(p_low.b_yoke_t, p_low.b_tooth_t) <= p_low.b_sat_t


def test_flux_linkage_scales_with_turns():
    p1 = size(replace(BASE, turns_per_phase=20))
    p2 = size(replace(BASE, turns_per_phase=40))
    assert p2.flux_linkage_wb == pytest.approx(2.0 * p1.flux_linkage_wb, rel=1e-6)
