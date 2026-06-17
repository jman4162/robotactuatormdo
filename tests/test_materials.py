"""Tests for the built-in material records."""

from __future__ import annotations

import pytest

from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.copper import load as load_cu
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.electrical_steel import load as load_steel
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.materials.magnets import load as load_magnet


def test_records_load_by_name():
    assert load_magnet("NdFeB-N42") is NDFEB_N42
    assert load_steel("M250-35A") is M250_35A
    assert load_cu("Cu") is COPPER
    for loader in (load_magnet, load_steel, load_cu):
        with pytest.raises(KeyError):
            loader("nope")


def test_positive_fields():
    assert NDFEB_N42.br_t > 0
    assert M250_35A.b_sat_t > 0
    assert COPPER.resistivity_20_ohm_m > 0


def test_magnet_br_derates_with_temperature():
    # NdFeB has a negative temperature coefficient.
    assert NDFEB_N42.br_at(100.0) < NDFEB_N42.br_at(20.0)
    assert NDFEB_N42.br_at(20.0) == pytest.approx(NDFEB_N42.br_t)


def test_copper_resistivity_rises_with_temperature():
    rho20 = COPPER.resistivity_at(20.0)
    rho70 = COPPER.resistivity_at(70.0)
    # ~0.39%/C -> ~21% over 50 C.
    assert rho70 / rho20 == pytest.approx(1.0 + 0.00393 * 50.0, rel=1e-6)
