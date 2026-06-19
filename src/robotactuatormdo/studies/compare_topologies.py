"""Topology comparison: the headline trade-study output (brief §8).

Scores a set of named architecture-class candidates against one requirement and reports a per-axis
winner map among the feasible classes — deliberately NOT a single scalar winner. A class can win on
no single axis yet still be Pareto-optimal (best balance), surfaced via ``is_dominated``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robotactuatormdo.requirements.joint import JointRequirement
from robotactuatormdo.studies.candidate import DesignCandidate
from robotactuatormdo.studies.pareto import non_dominated_indices
from robotactuatormdo.studies.scoring import SENSE, CandidateScore, Objective, score_candidate

__all__ = ["TopologyComparison", "compare_topologies", "DEFAULT_COMPARISON_OBJECTIVES"]

DEFAULT_COMPARISON_OBJECTIVES: tuple[Objective, ...] = (
    Objective.MASS_KG,
    Objective.CONTINUOUS_TORQUE_NM,
    Objective.PEAK_TORQUE_NM,
    Objective.REFLECTED_INERTIA,
    Objective.MISSION_ENERGY_J,
    Objective.MISSION_EFFICIENCY,
    Objective.COST_USD,
)


@dataclass(frozen=True, slots=True)
class TopologyComparison:
    requirement_name: str
    objective_keys: tuple[Objective, ...]
    rows: tuple[CandidateScore, ...]
    winners: dict[Objective, str | None]
    feasible_classes: tuple[str, ...]

    def winner_on(self, obj: Objective) -> str | None:
        return self.winners.get(obj)


def compare_topologies(
    requirement: JointRequirement,
    candidates: Sequence[DesignCandidate],
    objectives: tuple[Objective, ...] = DEFAULT_COMPARISON_OBJECTIVES,
    *,
    requirement_name: str = "joint",
    bus_voltage_v: float | None = None,
) -> TopologyComparison:
    """Score each candidate and compute per-axis winners among the feasible classes."""
    rows = tuple(
        score_candidate(c, requirement, objectives, bus_voltage_v=bus_voltage_v)
        for c in candidates
    )
    feasible = [s for s in rows if s.feasible]
    feasible_classes = tuple(s.architecture_class for s in feasible)

    winners: dict[Objective, str | None] = {}
    for obj in objectives:
        haves = [s for s in feasible if obj in s.objectives]
        if not haves:
            winners[obj] = None
            continue
        best = min(haves, key=lambda s: SENSE[obj] * s.objectives[obj])
        winners[obj] = best.architecture_class
    return TopologyComparison(
        requirement_name=requirement_name,
        objective_keys=objectives,
        rows=rows,
        winners=winners,
        feasible_classes=feasible_classes,
    )


def dominated_classes(comparison: TopologyComparison) -> tuple[str, ...]:
    """Architecture classes that are Pareto-dominated among the feasible set."""
    feasible = [s for s in comparison.rows if s.feasible]
    keys = tuple(
        k for k in comparison.objective_keys if all(k in s.objectives for s in feasible)
    )
    if not feasible or not keys:
        return ()
    matrix = np.array([s.vector(keys) for s in feasible], dtype=float)
    front = set(non_dominated_indices(matrix))
    return tuple(feasible[i].architecture_class for i in range(len(feasible)) if i not in front)
