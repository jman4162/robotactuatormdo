"""Tests for the Gearbox model and presets."""

from __future__ import annotations

import pytest

from robotactuatormdo.actuators.gearbox import (
    Gearbox,
    belt,
    cycloidal,
    direct_drive,
    harmonic,
    planetary,
)


def test_torque_round_trip_includes_drag():
    gb = Gearbox(ratio=6.0, forward_efficiency=0.95, no_load_drag_torque_nm=0.05)
    t_out = 12.0
    assert gb.output_torque(gb.input_torque(t_out)) == pytest.approx(t_out)


def test_input_speed_scales_with_ratio():
    gb = planetary(8.0)
    assert gb.input_speed(10.0) == pytest.approx(80.0)


def test_gear_loss_positive_when_lossy_and_zero_for_ideal():
    lossy = Gearbox(ratio=6.0, forward_efficiency=0.9)
    assert lossy.gear_loss_w(10.0, 5.0) > 0.0
    ideal = Gearbox(ratio=1.0, forward_efficiency=1.0)
    assert ideal.gear_loss_w(10.0, 5.0) == pytest.approx(0.0)


def test_reflected_inertia_scales_with_ratio_squared():
    gb = Gearbox(ratio=5.0, forward_efficiency=0.97)
    # no gearbox inertia -> exactly G^2 * J_motor
    assert gb.reflected_inertia_at_output(1e-4) == pytest.approx(25.0 * 1e-4)


def test_backdrive_torque_rises_with_ratio():
    low = planetary(3.0)
    high = planetary(10.0)
    assert high.backdrive_torque_nm() > low.backdrive_torque_nm()


def test_max_output_torque_default_infinite():
    assert direct_drive().max_output_torque_nm == float("inf")


def test_presets_are_valid():
    for gb in (direct_drive(), planetary(6.0), harmonic(100.0), cycloidal(40.0), belt(3.0)):
        assert gb.ratio > 0.0
        assert 0.0 < gb.forward_efficiency <= 1.0
        assert gb.backdrive_efficiency is not None


def test_invalid_gearbox():
    with pytest.raises(ValueError):
        Gearbox(ratio=0.0, forward_efficiency=0.9)
    with pytest.raises(ValueError):
        Gearbox(ratio=5.0, forward_efficiency=1.5)
