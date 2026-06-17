"""Tests for the topology-agnostic duty-cycle reducer."""

from __future__ import annotations

import numpy as np
import pytest

from robotactuatormdo.evaluation import evaluate_over_duty_cycle
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.requirements.duty_cycle import DutyCycle
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)


class LinearDummyMotor:
    """Phase current proportional to torque; fixed loss; optional current limit for feasibility."""

    def __init__(self, kt: float = 0.5, fixed_loss_w: float = 1.0, current_limit_a: float = 1e9):
        self.kt = kt
        self.fixed_loss_w = fixed_loss_w
        self.current_limit_a = current_limit_a

    def evaluate_operating_point(self, torque_nm, speed_rad_s, bus_voltage_v, ambient_temp_c):
        current = torque_nm / self.kt
        ok = current <= self.current_limit_a
        return MotorOperatingResult(
            torque_nm=torque_nm,
            speed_rad_s=speed_rad_s,
            phase_current_a_rms=current,
            phase_voltage_v_rms=bus_voltage_v,
            copper_loss_w=self.fixed_loss_w,
            core_loss_w=0.0,
            magnet_eddy_loss_w=0.0,
            mechanical_loss_w=0.0,
            winding_temp_c=ambient_temp_c + 10.0,
            magnet_temp_c=ambient_temp_c + 5.0,
            feasibility=FeasibilityFlags(current_ok=ok),
        )

    def mass_properties(self):
        return MassProperties(total_mass_kg=1.0, rotor_inertia_kg_m2=1e-4)

    def torque_speed_envelope(self):
        z = np.zeros(2)
        return TorqueSpeedEnvelope(speed_rad_s=z, peak_torque_nm=z, continuous_torque_nm=z)

    def efficiency_map(self):
        return EfficiencyMap(
            speed_rad_s=np.zeros(2), torque_nm=np.zeros(2), efficiency=np.zeros((2, 2))
        )


def test_dummy_satisfies_protocol():
    assert isinstance(LinearDummyMotor(), MotorModel)


def test_constant_duty_aggregates():
    motor = LinearDummyMotor(kt=0.5, fixed_loss_w=2.0)
    duty = DutyCycle.constant(torque_nm=4.0, speed_rad_s=10.0, duration_s=3.0)
    m = evaluate_over_duty_cycle(motor, duty, bus_voltage_v=48.0)

    assert m.peak_phase_current_a == pytest.approx(8.0)  # 4 / 0.5
    assert m.rms_phase_current_a == pytest.approx(8.0)  # constant
    assert m.mechanical_energy_j == pytest.approx(4.0 * 10.0 * 3.0)
    assert m.loss_energy_j == pytest.approx(2.0 * 3.0)
    assert m.electrical_energy_j == pytest.approx(m.mechanical_energy_j + m.loss_energy_j)
    assert m.all_feasible is True
    assert m.fraction_feasible == 1.0
    assert m.average_efficiency == pytest.approx(120.0 / 126.0)
    assert len(m.points) == duty.time_s.size


def test_infeasible_fraction():
    # current limit 8 A: torque 4 -> 8 A (ok), torque 6 -> 12 A (infeasible).
    motor = LinearDummyMotor(kt=0.5, current_limit_a=8.0)
    duty = DutyCycle.from_segments([(1.0, 4.0, 5.0), (1.0, 6.0, 5.0)])
    m = evaluate_over_duty_cycle(motor, duty, bus_voltage_v=48.0)
    assert m.all_feasible is False
    assert 0.0 < m.fraction_feasible < 1.0


def test_gear_ratio_maps_to_motor_shaft():
    motor = LinearDummyMotor(kt=0.5)
    duty = DutyCycle.constant(torque_nm=10.0, speed_rad_s=2.0, duration_s=1.0)
    m = evaluate_over_duty_cycle(motor, duty, bus_voltage_v=48.0, gear_ratio=5.0)
    # motor torque = 10 / 5 = 2 -> current = 2 / 0.5 = 4 A; motor speed = 2 * 5 = 10 rad/s
    assert m.peak_phase_current_a == pytest.approx(4.0)
    assert m.points[0].speed_rad_s == pytest.approx(10.0)


def test_invalid_gear_ratio():
    motor = LinearDummyMotor()
    duty = DutyCycle.constant(torque_nm=1.0, speed_rad_s=1.0, duration_s=1.0)
    with pytest.raises(ValueError):
        evaluate_over_duty_cycle(motor, duty, bus_voltage_v=48.0, gear_ratio=0.0)
