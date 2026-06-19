"""Tests for the commercial-catalog, radial-BLDC, and axial-adapter backends."""

from __future__ import annotations

import numpy as np
import pytest

from robotactuatormdo.geometry.radial_flux import RadialPMGeometry, size_radial_pm
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.motors.axial_adapter_axfluxmdo import AxFluxMDOAdapter
from robotactuatormdo.motors.commercial_catalog import CommercialMotor, CommercialMotorSpec
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.motors.radial_bldc import RadialBLDC, RadialBLDCParameters
from robotactuatormdo.motors.radial_pmsm import RadialPMSM

GEOM = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
    turns_per_phase=40,
)


def _radial_params():
    return size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER)


# --- commercial catalog ---

def _spec(**kw):
    base = dict(
        kt_nm_per_a_rms=0.1, r_phase_ohm=0.05, pole_pairs=7, max_phase_current_a_rms=40.0,
        rated_torque_nm=2.0, peak_torque_nm=4.0, max_speed_rad_s=300.0, rated_bus_voltage_v=48.0,
        total_mass_kg=1.5, rotor_inertia_kg_m2=5e-5,
    )
    base.update(kw)
    return CommercialMotorSpec(**base)


def test_commercial_is_motor_model():
    assert isinstance(CommercialMotor(_spec()), MotorModel)


def test_commercial_current_and_losses():
    m = CommercialMotor(_spec())
    res = m.evaluate_operating_point(1.0, 50.0, 48.0, 25.0)
    assert res.phase_current_a_rms == pytest.approx(1.0 / 0.1)
    r = 0.05 * (1 + 0.00393 * (25.0 - 20.0))
    assert res.copper_loss_w == pytest.approx(3 * (10.0**2) * r)


def test_commercial_rating_flags():
    m = CommercialMotor(_spec())
    assert not m.evaluate_operating_point(10.0, 10.0, 48.0, 25.0).feasibility.current_ok  # > peak
    over_speed = m.evaluate_operating_point(1.0, 400.0, 48.0, 25.0)
    assert not over_speed.feasibility.mechanical_ok  # > max speed


def test_commercial_envelope():
    env = CommercialMotor(_spec()).torque_speed_envelope(n_speeds=10)
    assert env.peak_torque_nm[0] == pytest.approx(4.0)
    assert np.all(env.continuous_torque_nm <= env.peak_torque_nm + 1e-9)


# --- radial BLDC ---

def test_bldc_is_motor_model():
    assert isinstance(RadialBLDC(RadialBLDCParameters(_radial_params())), MotorModel)


def test_bldc_derates_current_and_efficiency():
    p = _radial_params()
    pmsm = RadialPMSM(p)
    bldc = RadialBLDC(RadialBLDCParameters(p, kt_ratio_bldc_to_foc=0.9))
    a = pmsm.evaluate_operating_point(2.0, 10.0, 48.0, 25.0)
    b = bldc.evaluate_operating_point(2.0, 10.0, 48.0, 25.0)
    assert b.phase_current_a_rms == pytest.approx(a.phase_current_a_rms / 0.9, rel=1e-6)
    assert b.efficiency < a.efficiency


def test_bldc_identity_reproduces_pmsm():
    p = _radial_params()
    pmsm = RadialPMSM(p)
    bldc = RadialBLDC(RadialBLDCParameters(p, kt_ratio_bldc_to_foc=1.0, allow_field_weakening=True))
    a = pmsm.evaluate_operating_point(3.0, 40.0, 48.0, 30.0)
    b = bldc.evaluate_operating_point(3.0, 40.0, 48.0, 30.0)
    assert b.phase_current_a_rms == pytest.approx(a.phase_current_a_rms, rel=1e-9)
    assert b.winding_temp_c == pytest.approx(a.winding_temp_c, rel=1e-9)
    assert b.phase_voltage_v_rms == pytest.approx(a.phase_voltage_v_rms, rel=1e-9)


def test_bldc_envelope_plateau_scales_with_kt():
    p = _radial_params()
    kt = 0.9
    pmsm_peak = float(np.max(RadialPMSM(p).torque_speed_envelope().peak_torque_nm))
    bldc = RadialBLDC(RadialBLDCParameters(p, kt_ratio_bldc_to_foc=kt))
    bldc_peak = float(np.max(bldc.torque_speed_envelope().peak_torque_nm))
    assert bldc_peak == pytest.approx(kt * pmsm_peak, rel=0.05)


# --- axial adapter (duck-typed, no axfluxmdo) ---

class FakeAxialDesign:
    kt_nm_per_a_rms = 0.2
    ke_v_s_per_rad = 0.05
    r_phase_ohm = 0.03
    mass_kg = 2.0
    rotor_inertia_kg_m2 = 1e-3
    max_phase_current_a_rms = 50.0
    max_speed_rad_s = 200.0
    rated_bus_voltage_v = 48.0


def test_axial_adapter_is_motor_model_without_axfluxmdo():
    adapter = AxFluxMDOAdapter(FakeAxialDesign())
    assert isinstance(adapter, MotorModel)
    res = adapter.evaluate_operating_point(2.0, 20.0, 48.0, 25.0)
    assert res.phase_current_a_rms == pytest.approx(2.0 / 0.2)
    assert adapter.mass_properties().total_mass_kg == pytest.approx(2.0)


def test_axial_adapter_missing_quantity_errors():
    class Bad:
        mass_kg = 1.0  # no kt etc.

    with pytest.raises(AttributeError):
        AxFluxMDOAdapter(Bad())


def test_axial_adapter_strict_import_raises_without_package():
    pytest.importorskip  # noqa: B018 - ensure pytest available
    try:
        import axfluxmdo  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            AxFluxMDOAdapter(FakeAxialDesign(), require_axfluxmdo=True)
