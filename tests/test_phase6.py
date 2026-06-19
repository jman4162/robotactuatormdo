"""Tests for Phase 6: material cost fields, BOM cost, SMC/insulation, winding process."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from robotactuatormdo.geometry.process import HAIRPIN, ROUND_WIRE, geometry_for_process
from robotactuatormdo.geometry.radial_flux import RadialPMGeometry, size_radial_pm
from robotactuatormdo.materials.copper import COPPER, Conductor
from robotactuatormdo.materials.cost import CostModel, bom_cost
from robotactuatormdo.materials.electrical_steel import M250_35A, ElectricalSteel
from robotactuatormdo.materials.insulation import CLASS_B, CLASS_F, CLASS_H
from robotactuatormdo.materials.insulation import load as load_ins
from robotactuatormdo.materials.magnets import NDFEB_N42, MagnetMaterial
from robotactuatormdo.materials.smc import SMC_NOMINAL
from robotactuatormdo.materials.smc import load as load_smc
from robotactuatormdo.motors.radial_pmsm import RadialPMSM

GEOM = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
    turns_per_phase=40,
)


def test_cost_per_kg_defaults_zero_and_registry_positive():
    assert MagnetMaterial("x", 1.0, -1e-3, 1.05, 80.0, 7500.0, 1.4e-6).cost_per_kg == 0.0
    assert NDFEB_N42.cost_per_kg > 0 and M250_35A.cost_per_kg > 0 and COPPER.cost_per_kg > 0


def test_cost_per_kg_is_trailing_field():
    for rec in (MagnetMaterial, ElectricalSteel, Conductor):
        assert [f.name for f in fields(rec)][-1] == "cost_per_kg"


def test_bom_cost_monotonic_and_total():
    cb = bom_cost(0.1, 0.2, 0.5, NDFEB_N42, M250_35A, COPPER)
    assert cb.total_usd == pytest.approx(cb.material_usd + cb.processing_usd)
    more = bom_cost(0.2, 0.2, 0.5, NDFEB_N42, M250_35A, COPPER)
    assert more.total_usd > cb.total_usd  # more magnet mass -> higher cost


def test_bom_cost_process_multiplier_scales_processing():
    a = bom_cost(0.1, 0.2, 0.5, NDFEB_N42, M250_35A, COPPER, CostModel(process_cost_multiplier=1.0))
    b = bom_cost(0.1, 0.2, 0.5, NDFEB_N42, M250_35A, COPPER, CostModel(process_cost_multiplier=2.0))
    assert b.processing_usd > a.processing_usd
    assert b.material_usd == pytest.approx(a.material_usd)


def test_smc_and_insulation_records():
    assert load_smc("SMC-nominal") is SMC_NOMINAL
    assert SMC_NOMINAL.k_eddy < M250_35A.k_eddy  # SMC: lower eddy loss
    assert load_ins("F") is CLASS_F
    assert CLASS_B.max_temp_c < CLASS_F.max_temp_c < CLASS_H.max_temp_c
    with pytest.raises(KeyError):
        load_smc("nope")


def test_round_wire_reproduces_legacy_sizing():
    a = size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER)
    b = size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER, process=ROUND_WIRE)
    assert a == b  # ROUND_WIRE is a no-op


def test_hairpin_raises_current_capability():
    base = size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER, process=ROUND_WIRE)
    hp_geom = geometry_for_process(GEOM, HAIRPIN)
    hp = size_radial_pm(hp_geom, NDFEB_N42, M250_35A, COPPER, process=HAIRPIN)
    assert hp.max_phase_current_a_rms > base.max_phase_current_a_rms  # higher fill + J


def test_ac_loss_multiplier_raises_copper_loss():
    p = size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER)
    base = RadialPMSM(p).evaluate_operating_point(3.0, 10.0, 48.0, 25.0)
    hot = RadialPMSM(replace(p, ac_loss_multiplier=1.5)).evaluate_operating_point(
        3.0, 10.0, 48.0, 25.0
    )
    assert hot.copper_loss_w > base.copper_loss_w
