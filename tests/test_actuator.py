"""Tests for the Actuator (motor + gearbox at the output shaft)."""

from __future__ import annotations

import numpy as np
import pytest

from robotactuatormdo import DutyCycle, evaluate_over_duty_cycle
from robotactuatormdo.actuators.actuator import Actuator
from robotactuatormdo.actuators.gearbox import Gearbox, direct_drive, planetary
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


def test_actuator_is_motor_model():
    assert isinstance(Actuator(make_motor(), planetary(6.0)), MotorModel)


def test_direct_drive_matches_motor():
    motor = make_motor()
    act = Actuator(motor, direct_drive(efficiency=1.0))
    m = motor.evaluate_operating_point(2.0, 20.0, 48.0, 30.0)
    a = act.evaluate_operating_point(2.0, 20.0, 48.0, 30.0)
    assert a.torque_nm == pytest.approx(m.torque_nm)
    assert a.speed_rad_s == pytest.approx(m.speed_rad_s)
    assert a.phase_current_a_rms == pytest.approx(m.phase_current_a_rms)


def test_gearing_multiplies_torque_and_inertia():
    motor = make_motor()
    g = 6.0
    eta = 0.97
    gb = Gearbox(ratio=g, forward_efficiency=eta)  # no drag for a clean check
    act = Actuator(motor, gb)
    # For a given motor torque, output torque ~ eta * G * T_motor.
    t_motor = 1.0
    t_out = gb.output_torque(t_motor)
    assert t_out == pytest.approx(eta * g * t_motor)
    # Reflected inertia ~ G^2 * J_motor (no gearbox inertia here).
    j_motor = motor.mass_properties().rotor_inertia_kg_m2
    assert act.reflected_inertia_at_output() == pytest.approx(g**2 * j_motor)


def test_over_rating_trips_mechanical_ok():
    motor = make_motor()
    act = Actuator(motor, planetary(6.0, max_output_torque_nm=10.0))
    res = act.evaluate_operating_point(20.0, 1.0, 48.0, 30.0)
    assert not res.feasibility.mechanical_ok
    assert not res.feasibility.feasible


def test_envelope_maps_to_output():
    motor = make_motor()
    g = 6.0
    eta = 0.97
    act = Actuator(motor, Gearbox(ratio=g, forward_efficiency=eta))
    menv = motor.torque_speed_envelope()
    aenv = act.torque_speed_envelope()
    assert aenv.speed_rad_s[-1] == pytest.approx(menv.speed_rad_s[-1] / g)
    assert float(np.max(aenv.peak_torque_nm)) == pytest.approx(
        eta * g * float(np.max(menv.peak_torque_nm)), rel=1e-6
    )


def test_efficiency_map_bounds():
    act = Actuator(make_motor(), planetary(6.0))
    emap = act.efficiency_map(n_speeds=8, n_torques=8)
    assert emap.efficiency.shape == (8, 8)
    finite = emap.efficiency[np.isfinite(emap.efficiency)]
    assert finite.size > 0
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_vertical_slice_geared_beats_direct_on_output_torque():
    motor = make_motor()
    direct = Actuator(motor, direct_drive())
    qdd = Actuator(motor, planetary(6.0))
    # A high-torque, low-speed demand the bare motor cannot meet but the geared one can.
    duty = DutyCycle.constant(torque_nm=30.0, speed_rad_s=2.0, duration_s=1.0)
    m_direct = evaluate_over_duty_cycle(direct, duty, bus_voltage_v=48.0, ambient_temp_c=30.0)
    m_qdd = evaluate_over_duty_cycle(qdd, duty, bus_voltage_v=48.0, ambient_temp_c=30.0)
    assert not m_direct.all_feasible  # direct drive saturates current
    assert m_qdd.all_feasible
    assert m_qdd.mechanical_energy_j > 0.0
    assert 0.0 <= m_qdd.average_efficiency <= 1.0
