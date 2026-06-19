"""Tests for the RadialPMSM operating-point model (exact given lumped parameters)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robotactuatormdo import DutyCycle, evaluate_over_duty_cycle
from robotactuatormdo.geometry.radial_flux import RadialPMGeometry, size_radial_pm
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.motors.radial_pmsm import RadialPMParameters, RadialPMSM

SQRT2 = math.sqrt(2.0)


def make_params(**overrides) -> RadialPMParameters:
    base = dict(
        pole_pairs=4,
        flux_linkage_wb=0.05,
        r_s_ohm_20=0.1,
        l_d_h=5e-4,
        l_q_h=5e-4,
        max_phase_current_a_rms=20.0,
        rated_bus_voltage_v=48.0,
        iron_mass_kg=0.5,
        core_b_peak_t=1.0,
        k_hyst=0.02,
        steinmetz_alpha=1.8,
        k_eddy=8e-5,
        copper_temp_coeff_per_c=0.00393,
        r_th_winding_ambient_c_w=0.5,
        r_th_magnet_ambient_c_w=1.0,
        winding_temp_limit_c=155.0,
        magnet_temp_limit_c=80.0,
        b_tooth_t=1.2,
        b_yoke_t=1.0,
        b_sat_t=1.8,
        demag_id_limit_a_pk=40.0,
        total_mass_kg=1.0,
        rotor_inertia_kg_m2=1e-4,
        copper_mass_kg=0.2,
        magnet_mass_kg=0.1,
    )
    base.update(overrides)
    return RadialPMParameters(**base)


def test_satisfies_protocol():
    assert isinstance(RadialPMSM(make_params()), MotorModel)


def test_iq_from_torque_and_current():
    m = RadialPMSM(make_params())
    res = m.evaluate_operating_point(torque_nm=6.0, speed_rad_s=10.0, bus_voltage_v=48.0,
                                     ambient_temp_c=25.0)
    # i_q = T / (1.5 p psi_m) = 6 / (1.5*4*0.05) = 20 (peak); i_d = 0 at low speed.
    assert res.phase_current_a_rms == pytest.approx(20.0 / SQRT2)
    assert res.feasibility.current_ok


def test_loss_and_thermal_self_consistency():
    p = make_params()
    m = RadialPMSM(p)
    res = m.evaluate_operating_point(6.0, 10.0, 48.0, 25.0)
    r_s_eff = p.r_s_ohm_20 * (1.0 + p.copper_temp_coeff_per_c * (res.winding_temp_c - 20.0))
    assert res.copper_loss_w == pytest.approx(3 * res.phase_current_a_rms**2 * r_s_eff, rel=1e-4)
    assert res.winding_temp_c == pytest.approx(
        25.0 + p.r_th_winding_ambient_c_w * (res.copper_loss_w + res.core_loss_w), rel=1e-4
    )


def test_winding_temp_rises_with_torque():
    m = RadialPMSM(make_params())
    lo = m.evaluate_operating_point(2.0, 10.0, 48.0, 25.0)
    hi = m.evaluate_operating_point(6.0, 10.0, 48.0, 25.0)
    assert hi.winding_temp_c > lo.winding_temp_c


def test_current_limit_infeasible():
    m = RadialPMSM(make_params())
    # T=15 -> i_q=50 pk > I_max_pk=28.3 -> current_ok False.
    res = m.evaluate_operating_point(15.0, 10.0, 48.0, 25.0)
    assert not res.feasibility.current_ok


def test_high_speed_voltage_limited():
    m = RadialPMSM(make_params())
    res = m.evaluate_operating_point(6.0, 300.0, 48.0, 25.0)
    # Back-EMF far exceeds the ceiling; field weakening can't stay within current/demag.
    assert not res.feasibility.feasible


def test_thermal_trip_with_high_resistance_path():
    m = RadialPMSM(make_params(r_th_winding_ambient_c_w=10.0))
    res = m.evaluate_operating_point(6.0, 10.0, 48.0, 25.0)
    assert res.winding_temp_c > 155.0
    assert not res.feasibility.winding_temp_ok


def test_envelope_shape_and_current_limit_plateau():
    p = make_params()
    m = RadialPMSM(p)
    env = m.torque_speed_envelope(n_speeds=20)
    assert env.speed_rad_s.shape == env.peak_torque_nm.shape == (20,)
    t_cl = 1.5 * p.pole_pairs * p.flux_linkage_wb * p.max_phase_current_a_rms * SQRT2
    assert env.peak_torque_nm[0] == pytest.approx(t_cl, rel=1e-2)
    # continuous never exceeds peak.
    assert np.all(env.continuous_torque_nm <= env.peak_torque_nm + 1e-6)
    # peak torque does not increase with speed (current- then voltage-limited).
    assert env.peak_torque_nm[-1] <= env.peak_torque_nm[0] + 1e-6


def test_efficiency_map_bounds():
    m = RadialPMSM(make_params())
    emap = m.efficiency_map(n_speeds=10, n_torques=10)
    assert emap.efficiency.shape == (10, 10)
    finite = emap.efficiency[np.isfinite(emap.efficiency)]
    assert finite.size > 0
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_single_node_network_reproduces_default_winding_temp():
    from dataclasses import replace

    from robotactuatormdo.thermal.lumped_network import single_node_network

    p = make_params()
    base = RadialPMSM(p).evaluate_operating_point(6.0, 10.0, 48.0, 25.0)
    net = single_node_network(p.r_th_winding_ambient_c_w, p.r_th_magnet_ambient_c_w)
    netted = RadialPMSM(replace(p, thermal_network=net))
    res = netted.evaluate_operating_point(6.0, 10.0, 48.0, 25.0)
    assert res.winding_temp_c == pytest.approx(base.winding_temp_c, rel=1e-4)


def test_multi_node_network_orders_temps_and_fixes_magnet():
    from dataclasses import replace

    from robotactuatormdo.thermal.lumped_network import radial_pm_network

    p = make_params()
    net = radial_pm_network(
        r_winding_stator_c_w=0.1, r_stator_ambient_c_w=0.3, r_magnet_ambient_c_w=2.0, ambient_c=25.0
    )
    motor = RadialPMSM(replace(p, thermal_network=net))
    res = motor.evaluate_operating_point(6.0, 10.0, 48.0, 25.0)
    # Magnet carries only (zero) eddy loss now -> sits at ambient, not driven by copper loss.
    assert res.magnet_temp_c == pytest.approx(25.0, abs=1e-6)
    assert res.winding_temp_c > res.magnet_temp_c


def test_network_requires_winding_and_magnet_nodes():
    from dataclasses import replace

    from robotactuatormdo.thermal.lumped_network import ThermalEdge, ThermalNetwork, ThermalNode

    bad = ThermalNetwork(
        nodes=(ThermalNode("winding", 100.0), ThermalNode("amb", fixed_temp_c=25.0)),
        edges=(ThermalEdge.from_resistance("winding", "amb", 0.5),),
    )  # no 'magnet' node
    with pytest.raises(ValueError):
        replace(make_params(), thermal_network=bad)


def test_vertical_slice_through_reducer():
    geom = RadialPMGeometry(
        air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
        pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
        turns_per_phase=40,
    )
    motor = RadialPMSM(size_radial_pm(geom, NDFEB_N42, M250_35A, COPPER))
    peak = float(np.max(motor.torque_speed_envelope().peak_torque_nm))
    duty = DutyCycle.constant(torque_nm=0.3 * peak, speed_rad_s=8.0, duration_s=1.0)
    mission = evaluate_over_duty_cycle(motor, duty, bus_voltage_v=48.0, ambient_temp_c=30.0)
    assert mission.mechanical_energy_j > 0.0
    assert mission.electrical_energy_j >= mission.mechanical_energy_j
    assert 0.0 <= mission.average_efficiency <= 1.0
    assert mission.peak_phase_current_a > 0.0
