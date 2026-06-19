"""Tests for inverter loss math and the Inverter device + presets."""

from __future__ import annotations

import math

import pytest

from robotactuatormdo.electronics.inverter import Inverter, gan, si_mosfet, sic
from robotactuatormdo.losses.inverter import conduction_loss_w, switching_loss_w

SQRT2 = math.sqrt(2.0)


def test_conduction_loss_formula():
    assert conduction_loss_w(10.0, 5e-3, n_conducting=3) == pytest.approx(3 * 100 * 5e-3)


def test_switching_loss_scales():
    base = switching_loss_w(48.0, 20.0, 50e-9, 20e3, n_switches=6)
    assert switching_loss_w(48.0, 20.0, 50e-9, 40e3, n_switches=6) == pytest.approx(2 * base)
    assert switching_loss_w(96.0, 20.0, 50e-9, 20e3, n_switches=6) == pytest.approx(2 * base)


def test_inverter_loss_is_sum():
    inv = Inverter(r_ds_on_ohm=5e-3, switching_time_s=50e-9, f_sw_hz=20e3)
    i_rms = 15.0
    v = 48.0
    expected = inv.conduction_loss_w(i_rms) + inv.switching_loss_w(v, i_rms * SQRT2)
    assert inv.loss_w(i_rms, v) == pytest.approx(expected)


def test_presets_valid_and_wide_bandgap_faster():
    for inv in (si_mosfet(), gan(), sic()):
        assert inv.r_ds_on_ohm >= 0.0
        assert inv.f_sw_hz > 0.0
    # GaN/SiC switch faster than silicon.
    assert gan().switching_time_s < si_mosfet().switching_time_s
    assert sic().switching_time_s < si_mosfet().switching_time_s


def test_invalid_inverter():
    with pytest.raises(ValueError):
        Inverter(r_ds_on_ohm=5e-3, switching_time_s=50e-9, f_sw_hz=0.0)
