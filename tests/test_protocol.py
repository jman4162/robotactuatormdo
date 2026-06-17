"""A conforming class must satisfy the runtime-checkable MotorModel protocol."""

from __future__ import annotations

from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)


class DummyMotor:
    def evaluate_operating_point(self, torque_nm, speed_rad_s, bus_voltage_v, ambient_temp_c):
        return MotorOperatingResult(
            torque_nm=torque_nm,
            speed_rad_s=speed_rad_s,
            phase_current_a_rms=1.0,
            phase_voltage_v_rms=1.0,
            copper_loss_w=0.0,
            core_loss_w=0.0,
            magnet_eddy_loss_w=0.0,
            mechanical_loss_w=0.0,
            winding_temp_c=ambient_temp_c,
            magnet_temp_c=ambient_temp_c,
            feasibility=FeasibilityFlags(),
        )

    def mass_properties(self):
        return MassProperties(total_mass_kg=1.0, rotor_inertia_kg_m2=1e-4)

    def torque_speed_envelope(self):
        import numpy as np

        z = np.zeros(2)
        return TorqueSpeedEnvelope(speed_rad_s=z, peak_torque_nm=z, continuous_torque_nm=z)

    def efficiency_map(self):
        import numpy as np

        return EfficiencyMap(
            speed_rad_s=np.zeros(2), torque_nm=np.zeros(2), efficiency=np.zeros((2, 2))
        )


def test_dummy_is_motor_model():
    assert isinstance(DummyMotor(), MotorModel)


def test_incomplete_class_is_not_motor_model():
    class Incomplete:
        def evaluate_operating_point(self, *a):
            return None

    assert not isinstance(Incomplete(), MotorModel)


def test_operating_result_efficiency():
    res = DummyMotor().evaluate_operating_point(2.0, 10.0, 48.0, 25.0)
    # No losses => efficiency is 1.0; mechanical power = 20 W.
    assert res.mechanical_power_w == 20.0
    assert res.efficiency == 1.0
