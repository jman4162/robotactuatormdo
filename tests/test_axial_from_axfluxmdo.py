"""Test AxFluxMDOAdapter.from_axfluxmdo with a fake axfluxmdo-like model (no real dependency)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from robotactuatormdo.motors.axial_adapter_axfluxmdo import AxFluxMDOAdapter
from robotactuatormdo.motors.protocols import MotorModel


@dataclass
class FakeResult:
    torque_nm: float
    back_emf_v_rms: float
    phase_resistance_ohm: float
    mass_kg: float
    efficiency: float
    mass_breakdown: dict = field(default_factory=dict)


class FakeOp:
    def __init__(self, speed_rpm, current_rms, dc_bus_voltage=48.0):
        self.speed_rpm = speed_rpm
        self.current_rms = current_rms
        self.dc_bus_voltage = dc_bus_voltage


@dataclass
class FakeDesign:
    outer_radius: float = 0.05
    inner_radius: float = 0.03


class FakeModel:
    # torque = kt * current with kt = 0.09; back-emf ~ ke * omega_e
    def evaluate(self, design, op):
        torque = 0.09 * op.current_rms
        return FakeResult(
            torque_nm=torque,
            back_emf_v_rms=0.02 * op.speed_rpm,
            phase_resistance_ohm=0.03,
            mass_kg=1.1,
            efficiency=0.92,
            mass_breakdown={"magnets": 0.13, "back_iron": 0.23, "total": 1.1},
        )


def _adapter():
    return AxFluxMDOAdapter.from_axfluxmdo(
        FakeDesign(), FakeModel(), FakeOp(speed_rpm=500, current_rms=20.0),
        max_phase_current_a_rms=40.0, max_speed_rad_s=200.0, rated_bus_voltage_v=48.0,
    )


def test_from_axfluxmdo_is_motor_model():
    assert isinstance(_adapter(), MotorModel)


def test_kt_derived_from_torque_over_current():
    a = _adapter()
    assert a.kt == pytest.approx(0.09 * 20.0 / 20.0)  # torque/current = 0.09
    res = a.evaluate_operating_point(1.8, 20.0, 48.0, 25.0)
    assert res.phase_current_a_rms == pytest.approx(1.8 / a.kt)


def test_mass_and_inertia_passthrough():
    a = _adapter()
    mp = a.mass_properties()
    assert mp.total_mass_kg == pytest.approx(1.1)
    # first-order disk inertia from rotor mass (magnets+back_iron=0.36) and radii
    expected = 0.5 * 0.36 * (0.05**2 + 0.03**2)
    assert mp.rotor_inertia_kg_m2 == pytest.approx(expected)


def test_efficiency_at_is_wired():
    a = _adapter()
    # evaluate uses the efficiency_at hook -> some non-copper loss is attributed
    res = a.evaluate_operating_point(1.8, 50.0, 48.0, 25.0)
    assert res.total_loss_w >= res.copper_loss_w
    assert math.isfinite(res.core_loss_w)
