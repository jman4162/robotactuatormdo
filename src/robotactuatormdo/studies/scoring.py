"""Scoring: turn a built MotorModel + a JointRequirement into objectives + a feasibility verdict.

All objectives are computed from the **protocol surface** (``torque_speed_envelope`` +
``mass_properties`` + the duty-cycle reducer), so they work uniformly for a bare motor, an
``Actuator``, or a ``PowerStage``. Studies minimize internally; ``SENSE`` records each objective's
natural direction. RMS winding temperature is *derived* from ``MissionResult.points`` (it is not a
stored field).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from robotactuatormdo._numeric import trapezoid
from robotactuatormdo.evaluation import evaluate_over_duty_cycle
from robotactuatormdo.requirements.joint import JointRequirement
from robotactuatormdo.studies.candidate import DesignCandidate

__all__ = ["Objective", "SENSE", "CandidateScore", "score_candidate"]


class Objective(str, Enum):
    MASS_KG = "mass_kg"
    PEAK_TORQUE_NM = "peak_torque_nm"
    CONTINUOUS_TORQUE_NM = "continuous_torque_nm"
    TORQUE_DENSITY = "torque_density_nm_per_kg"
    REFLECTED_INERTIA = "reflected_inertia_kg_m2"
    RMS_WINDING_TEMP_C = "rms_winding_temp_c"
    MISSION_ENERGY_J = "mission_energy_j"
    MISSION_EFFICIENCY = "mission_efficiency"
    COST_USD = "cost_usd"


# +1 => minimize, -1 => maximize. Studies minimize SENSE * value.
SENSE: dict[Objective, int] = {
    Objective.MASS_KG: +1,
    Objective.PEAK_TORQUE_NM: -1,
    Objective.CONTINUOUS_TORQUE_NM: -1,
    Objective.TORQUE_DENSITY: -1,
    Objective.REFLECTED_INERTIA: +1,
    Objective.RMS_WINDING_TEMP_C: +1,
    Objective.MISSION_ENERGY_J: +1,
    Objective.MISSION_EFFICIENCY: -1,
    Objective.COST_USD: +1,
}


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """One candidate scored against one requirement."""

    name: str
    architecture_class: str
    feasible: bool
    objectives: dict[Objective, float]  # natural units; only the computable keys are present
    constraint_margins: dict[str, float]  # >= 0 satisfied (nan = not checkable)

    def vector(self, keys: tuple[Objective, ...]) -> np.ndarray:
        """Minimize-signed objective vector for ``keys`` (all must be present)."""
        return np.array([SENSE[k] * self.objectives[k] for k in keys], dtype=float)


def _rms_winding_temp_c(mission, duty) -> float:
    temps = np.array([p.winding_temp_c for p in mission.points], dtype=float)
    if temps.size < 2:
        return float(temps[0]) if temps.size else float("nan")
    return float(np.sqrt(trapezoid(temps**2, duty.time_s) / duty.duration_s))


def score_candidate(
    candidate: DesignCandidate,
    requirement: JointRequirement,
    objectives: tuple[Objective, ...],
    *,
    bus_voltage_v: float | None = None,
) -> CandidateScore:
    """Build and score ``candidate`` against ``requirement`` for the requested ``objectives``."""
    motor = candidate.build()
    env = motor.torque_speed_envelope()
    mass = motor.mass_properties()
    peak = float(np.max(env.peak_torque_nm))
    cont = float(np.max(env.continuous_torque_nm))
    feasible_speeds = np.asarray(env.speed_rad_s)[np.asarray(env.peak_torque_nm) > 1e-9]
    top_speed = float(np.max(feasible_speeds)) if feasible_speeds.size else 0.0

    mission = None
    if requirement.duty_cycle is not None:
        bus = bus_voltage_v if bus_voltage_v is not None else requirement.bus_voltage_v
        mission = evaluate_over_duty_cycle(
            motor,
            requirement.duty_cycle,
            bus_voltage_v=bus,
            ambient_temp_c=requirement.ambient_temp_c,
        )

    margins = {
        "peak_torque": peak - requirement.peak_torque_nm,
        "continuous_torque": cont - requirement.continuous_rms_torque_nm,
        "top_speed": top_speed - requirement.max_speed_rad_s,
        "mass": requirement.max_mass_kg - mass.total_mass_kg,
    }
    if mission is not None:
        margins["mission_feasible"] = 0.0 if mission.all_feasible else -1.0
    feasible = all(m >= 0.0 for m in margins.values())

    cost = getattr(candidate, "cost_usd", None)
    available: dict[Objective, float] = {
        Objective.MASS_KG: mass.total_mass_kg,
        Objective.PEAK_TORQUE_NM: peak,
        Objective.CONTINUOUS_TORQUE_NM: cont,
        Objective.TORQUE_DENSITY: peak / mass.total_mass_kg if mass.total_mass_kg else 0.0,
        Objective.REFLECTED_INERTIA: mass.rotor_inertia_kg_m2,
    }
    if mission is not None:
        available[Objective.MISSION_ENERGY_J] = mission.electrical_energy_j
        available[Objective.MISSION_EFFICIENCY] = mission.average_efficiency
        available[Objective.RMS_WINDING_TEMP_C] = _rms_winding_temp_c(
            mission, requirement.duty_cycle
        )
    if cost is not None:
        available[Objective.COST_USD] = float(cost)

    chosen = {k: available[k] for k in objectives if k in available}
    return CandidateScore(
        name=candidate.name,
        architecture_class=candidate.architecture_class,
        feasible=feasible,
        objectives=chosen,
        constraint_margins=margins,
    )
