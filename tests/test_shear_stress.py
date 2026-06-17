"""Tests for the air-gap shear-stress fair-comparison primitives."""

from __future__ import annotations

import math

import pytest

from robotactuatormdo.geometry import shear_stress as ss


def test_radial_round_trip():
    r_g, l_stk, sigma = 0.05, 0.04, 30_000.0
    torque = ss.torque_radial(sigma, r_g, l_stk)
    assert ss.shear_stress_radial(torque, r_g, l_stk) == pytest.approx(sigma)


def test_axial_round_trip():
    r_o, r_i, sigma = 0.08, 0.04, 25_000.0
    torque = ss.torque_axial(sigma, r_o, r_i)
    assert ss.shear_stress_axial(torque, r_o, r_i) == pytest.approx(sigma)


def test_radial_torque_scales_with_radius_squared_times_stack():
    sigma = 20_000.0
    base = ss.torque_radial(sigma, 0.05, 0.04)
    double_r = ss.torque_radial(sigma, 0.10, 0.04)
    double_l = ss.torque_radial(sigma, 0.05, 0.08)
    assert double_r == pytest.approx(4.0 * base)  # r^2
    assert double_l == pytest.approx(2.0 * base)  # L


def test_axial_torque_scales_with_radius_cubed_difference():
    sigma = 20_000.0
    r_i = 0.04
    t1 = ss.torque_axial(sigma, 0.08, r_i)
    t2 = ss.torque_axial(sigma, 0.10, r_i)
    ratio = t2 / t1
    expected = (0.10**3 - r_i**3) / (0.08**3 - r_i**3)
    assert ratio == pytest.approx(expected)


def test_area_helpers():
    assert ss.area_radial(0.05, 0.04) == pytest.approx(2 * math.pi * 0.05 * 0.04)
    assert ss.area_axial(0.08, 0.04) == pytest.approx(math.pi * (0.08**2 - 0.04**2))


@pytest.mark.parametrize(
    "call",
    [
        lambda: ss.shear_stress_radial(1.0, 0.0, 0.04),
        lambda: ss.torque_radial(1.0, 0.05, -0.01),
        lambda: ss.shear_stress_axial(1.0, 0.04, 0.04),  # r_o == r_i
        lambda: ss.torque_axial(1.0, 0.04, 0.08),  # r_o < r_i
    ],
)
def test_invalid_geometry_raises(call):
    with pytest.raises(ValueError):
        call()
