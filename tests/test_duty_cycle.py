"""Tests for the DutyCycle schema and its reductions."""

from __future__ import annotations

import numpy as np
import pytest

from robotactuatormdo.requirements.duty_cycle import DutyCycle


def test_constant_rms_equals_peak():
    d = DutyCycle.constant(torque_nm=5.0, speed_rad_s=10.0, duration_s=2.0)
    assert d.duration_s == 2.0
    assert d.peak_torque_nm == 5.0
    assert d.rms_torque_nm == pytest.approx(5.0)
    assert d.max_speed_rad_s == 10.0
    # E = T * omega * t = 5 * 10 * 2
    assert d.mechanical_energy_j == pytest.approx(100.0)


def test_square_wave_rms_closed_form():
    # Equal time at 0 N·m and at 10 N·m -> RMS = 10/sqrt(2).
    d = DutyCycle.from_segments([(1.0, 0.0, 4.0), (1.0, 10.0, 4.0)])
    assert d.rms_torque_nm == pytest.approx(10.0 / np.sqrt(2.0), rel=1e-6)
    assert d.peak_torque_nm == 10.0


def test_from_segments_strictly_increasing_time():
    d = DutyCycle.from_segments([(0.5, 1.0, 2.0), (0.5, 3.0, 2.0)])
    assert np.all(np.diff(d.time_s) > 0.0)
    assert d.duration_s == pytest.approx(1.0)


def test_validation_errors():
    with pytest.raises(ValueError):
        DutyCycle(time_s=np.array([0.0]), torque_nm=np.array([1.0]), speed_rad_s=np.array([1.0]))
    with pytest.raises(ValueError):  # non-monotonic time
        DutyCycle(
            time_s=np.array([0.0, 0.0]),
            torque_nm=np.array([1.0, 1.0]),
            speed_rad_s=np.array([1.0, 1.0]),
        )
    with pytest.raises(ValueError):  # mismatched lengths
        DutyCycle(
            time_s=np.array([0.0, 1.0]),
            torque_nm=np.array([1.0]),
            speed_rad_s=np.array([1.0, 1.0]),
        )
    with pytest.raises(ValueError):
        DutyCycle.constant(torque_nm=1.0, speed_rad_s=1.0, duration_s=0.0)
