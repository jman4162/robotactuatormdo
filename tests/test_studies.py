"""Tests for scoring, compare_topologies, and robust evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from robotactuatormdo.actuators.actuator import Actuator
from robotactuatormdo.actuators.gearbox import planetary
from robotactuatormdo.geometry.radial_flux import RadialPMGeometry, size_radial_pm
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.motors.radial_pmsm import RadialPMSM
from robotactuatormdo.requirements.duty_cycle import DutyCycle
from robotactuatormdo.requirements.joint import JointRequirement
from robotactuatormdo.results import EfficiencyMap, MassProperties, TorqueSpeedEnvelope
from robotactuatormdo.studies.candidate import FactoryCandidate
from robotactuatormdo.studies.compare_topologies import compare_topologies
from robotactuatormdo.studies.robust import Uncertainty, robust_score
from robotactuatormdo.studies.scoring import Objective, score_candidate

GEOM = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
    turns_per_phase=40,
)


def _radial(_params):
    return RadialPMSM(size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER))


def _qdd(params):
    return Actuator(_radial(params), planetary(params.get("ratio", 6.0)))


def _req(peak=5.0, cont=0.5, max_speed=15.0, mass=10.0, duty=True):
    # max_speed 15 rad/s < the QDD's ~30 rad/s output top speed, so both classes are feasible.
    return JointRequirement(
        peak_torque_nm=peak, continuous_rms_torque_nm=cont, max_speed_rad_s=max_speed,
        bus_voltage_v=48.0, max_phase_current_a_rms=60.0, envelope_outer_diameter_m=0.1,
        envelope_axial_length_m=0.1, max_mass_kg=mass, ambient_temp_c=30.0,
        duty_cycle=DutyCycle.constant(2.0, 8.0, 1.0) if duty else None,
    )


def test_score_feasible_and_mass_objective():
    cand = FactoryCandidate("rd", "radial_direct_drive", _radial)
    s = score_candidate(cand, _req(), (Objective.MASS_KG, Objective.PEAK_TORQUE_NM))
    assert s.feasible
    assert s.objectives[Objective.MASS_KG] == pytest.approx(
        cand.build().mass_properties().total_mass_kg
    )


def test_infeasible_when_peak_torque_too_high():
    cand = FactoryCandidate("rd", "radial_direct_drive", _radial)
    s = score_candidate(cand, _req(peak=1e6), (Objective.MASS_KG,))
    assert not s.feasible
    assert s.constraint_margins["peak_torque"] < 0.0


def test_objectives_agree_with_actuator_properties():
    cand = FactoryCandidate("qdd", "radial_qdd", _qdd, {"ratio": 6.0})
    s = score_candidate(cand, _req(), (Objective.PEAK_TORQUE_NM, Objective.REFLECTED_INERTIA))
    props = cand.build().properties()
    assert s.objectives[Objective.PEAK_TORQUE_NM] == pytest.approx(props.peak_output_torque_nm)
    assert s.objectives[Objective.REFLECTED_INERTIA] == pytest.approx(
        props.reflected_inertia_kg_m2
    )


def test_compare_topologies_per_axis_winners():
    direct = FactoryCandidate("dd", "radial_direct_drive", _radial)
    qdd = FactoryCandidate("qdd", "radial_qdd", _qdd, {"ratio": 6.0})
    comp = compare_topologies(
        _req(), [direct, qdd],
        objectives=(Objective.PEAK_TORQUE_NM, Objective.REFLECTED_INERTIA),
    )
    assert comp.winner_on(Objective.PEAK_TORQUE_NM) == "radial_qdd"  # gearing multiplies torque
    assert comp.winner_on(Objective.REFLECTED_INERTIA) == "radial_direct_drive"  # G^2 penalty


def test_compare_excludes_mass_budget_failure():
    direct = FactoryCandidate("dd", "radial_direct_drive", _radial)
    comp = compare_topologies(
        _req(mass=0.1), [direct], objectives=(Objective.MASS_KG,)
    )  # 0.1 kg budget is impossible
    assert comp.feasible_classes == ()
    assert comp.winner_on(Objective.MASS_KG) is None


# --- robust ---

class _FakeMotor:
    def __init__(self, mass):
        self._mass = mass

    def evaluate_operating_point(self, *a):  # pragma: no cover - not used (no duty)
        raise NotImplementedError

    def mass_properties(self):
        return MassProperties(total_mass_kg=self._mass, rotor_inertia_kg_m2=1e-4)

    def torque_speed_envelope(self):
        s = np.array([0.0, 100.0])
        big = np.array([1000.0, 1000.0])
        return TorqueSpeedEnvelope(speed_rad_s=s, peak_torque_nm=big, continuous_torque_nm=big)

    def efficiency_map(self):  # pragma: no cover - not used
        z = np.zeros((2, 2))
        return EfficiencyMap(speed_rad_s=np.zeros(2), torque_nm=np.zeros(2), efficiency=z)


def test_robust_p95_matches_analytic_normal():
    nominal = 2.0
    spread = 0.05
    cand = FactoryCandidate("fake", "fake", lambda p: _FakeMotor(p["x"]), {"x": nominal})
    req = _req(peak=1.0, cont=0.0, max_speed=1.0, mass=1e9, duty=False)
    rs = robust_score(
        cand, req, (Objective.MASS_KG,),
        [Uncertainty("x", "normal", spread)], n_samples=20000, seed=1,
    )
    assert rs.feasible_fraction == 1.0
    # minimize objective -> 95th percentile = nominal*(1 + spread*z95), z95 ~ 1.6449
    expected = nominal * (1.0 + spread * 1.6449)
    assert rs.p95_objectives[Objective.MASS_KG] == pytest.approx(expected, rel=0.02)


def test_robust_is_deterministic_by_seed():
    cand = FactoryCandidate("fake", "fake", lambda p: _FakeMotor(p["x"]), {"x": 2.0})
    req = _req(peak=1.0, cont=0.0, max_speed=1.0, mass=1e9, duty=False)
    kw = dict(n_samples=500, seed=7)
    a = robust_score(cand, req, (Objective.MASS_KG,), [Uncertainty("x", "normal", 0.05)], **kw)
    b = robust_score(cand, req, (Objective.MASS_KG,), [Uncertainty("x", "normal", 0.05)], **kw)
    assert a.p95_objectives[Objective.MASS_KG] == b.p95_objectives[Objective.MASS_KG]
