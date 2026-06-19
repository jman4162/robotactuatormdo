"""Tests for the PowerStage (battery + cable + inverter wrapping a motor/actuator)."""

from __future__ import annotations

import pytest

from robotactuatormdo import DutyCycle, evaluate_over_duty_cycle
from robotactuatormdo.actuators.actuator import Actuator
from robotactuatormdo.actuators.gearbox import planetary
from robotactuatormdo.electronics.battery import Battery
from robotactuatormdo.electronics.cable import Cable
from robotactuatormdo.electronics.inverter import Inverter, gan
from robotactuatormdo.electronics.power_stage import PowerStage
from robotactuatormdo.geometry.radial_flux import RadialPMGeometry, size_radial_pm
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.motors.radial_pmsm import RadialPMSM

GEOM = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
    turns_per_phase=40,
)


def make_motor() -> RadialPMSM:
    return RadialPMSM(size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER))


def _ideal_inverter() -> Inverter:
    # Lossless: no conduction (R=0), no switching (t=0). f_sw must be > 0.
    return Inverter(r_ds_on_ohm=0.0, switching_time_s=0.0, f_sw_hz=1e3)


def test_power_stage_is_motor_model():
    stage = PowerStage(make_motor(), gan(), Battery(internal_resistance_ohm=0.05))
    assert isinstance(stage, MotorModel)


def test_ideal_source_matches_inner():
    motor = make_motor()
    stage = PowerStage(motor, _ideal_inverter(), Battery(internal_resistance_ohm=0.0))
    m = motor.evaluate_operating_point(2.0, 20.0, 48.0, 25.0)
    s = stage.evaluate_operating_point(2.0, 20.0, 48.0, 25.0)
    assert s.phase_current_a_rms == pytest.approx(m.phase_current_a_rms, rel=1e-6)
    assert s.inverter_loss_w == pytest.approx(0.0)
    link = stage.link_state(2.0, 20.0, 48.0, 25.0)
    assert link.dc_link_voltage_v == pytest.approx(48.0, rel=1e-6)


def test_bus_sags_under_load_and_with_resistance():
    motor = make_motor()
    op = (8.0, 10.0)  # high-current point
    soft = PowerStage(motor, gan(), Battery(internal_resistance_ohm=0.5))
    stiff = PowerStage(motor, gan(), Battery(internal_resistance_ohm=0.05))
    v_soft = soft.link_state(*op, bus_voltage_v=48.0).dc_link_voltage_v
    v_stiff = stiff.link_state(*op, bus_voltage_v=48.0).dc_link_voltage_v
    assert v_soft < v_stiff < 48.0  # sag, more with higher internal resistance


def test_inverter_loss_enters_budget():
    stage = PowerStage(make_motor(), gan(), Battery(internal_resistance_ohm=0.05))
    res = stage.evaluate_operating_point(8.0, 10.0, 48.0, 25.0)
    assert res.inverter_loss_w > 0.0
    assert res.total_loss_w >= res.copper_loss_w + res.inverter_loss_w


def test_battery_current_limit_trips_source_ok():
    stage = PowerStage(
        make_motor(), gan(), Battery(internal_resistance_ohm=0.02, max_current_a=1.0)
    )
    res = stage.evaluate_operating_point(8.0, 10.0, 48.0, 25.0)
    assert not res.feasibility.source_ok
    assert not res.feasibility.feasible


def test_cable_resistance_adds_to_sag():
    motor = make_motor()
    op = (8.0, 10.0)
    no_cable = PowerStage(motor, gan(), Battery(internal_resistance_ohm=0.05))
    with_cable = PowerStage(
        motor, gan(), Battery(internal_resistance_ohm=0.05), Cable(resistance_ohm=0.1)
    )
    v0 = no_cable.link_state(*op, bus_voltage_v=48.0).dc_link_voltage_v
    v1 = with_cable.link_state(*op, bus_voltage_v=48.0).dc_link_voltage_v
    assert v1 < v0


def test_low_voltage_raises_dc_current_and_can_break_feasibility():
    motor = make_motor()
    stage = PowerStage(motor, gan(), Battery(internal_resistance_ohm=0.05))
    # Same loaded point: lower pack voltage -> higher DC current (power conservation).
    i_dc_24 = stage.link_state(8.0, 10.0, 24.0).dc_current_a
    i_dc_48 = stage.link_state(8.0, 10.0, 48.0).dc_current_a
    assert i_dc_24 > 1.7 * i_dc_48  # ~2x for half the voltage
    # A high-speed point feasible at 48 V but not at 24 V (lower voltage ceiling).
    hi48 = stage.evaluate_operating_point(0.5, 150.0, 48.0, 25.0)
    hi24 = stage.evaluate_operating_point(0.5, 150.0, 24.0, 25.0)
    assert hi48.feasibility.feasible
    assert not hi24.feasibility.feasible


def test_nesting_with_actuator_and_reducer():
    motor = make_motor()
    inner = Actuator(motor, planetary(6.0))
    stack = PowerStage(inner, gan(), Battery(internal_resistance_ohm=0.05))
    assert isinstance(stack, MotorModel)
    duty = DutyCycle.constant(torque_nm=30.0, speed_rad_s=3.0, duration_s=1.0)
    mission = evaluate_over_duty_cycle(stack, duty, bus_voltage_v=48.0, ambient_temp_c=30.0)
    assert mission.mechanical_energy_j > 0.0
    assert 0.0 <= mission.average_efficiency <= 1.0
