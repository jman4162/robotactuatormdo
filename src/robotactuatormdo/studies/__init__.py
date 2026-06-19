"""Trade studies: Pareto fronts, robust design, topology comparison."""

from __future__ import annotations

from robotactuatormdo.studies.candidate import DesignCandidate, FactoryCandidate
from robotactuatormdo.studies.compare_topologies import (
    DEFAULT_COMPARISON_OBJECTIVES,
    TopologyComparison,
    compare_topologies,
    dominated_classes,
)
from robotactuatormdo.studies.pareto import (
    ParetoFront,
    grid,
    non_dominated_indices,
    pareto_front,
)
from robotactuatormdo.studies.robust import RobustScore, Uncertainty, robust_score
from robotactuatormdo.studies.scoring import SENSE, CandidateScore, Objective, score_candidate

__all__ = [
    "DesignCandidate",
    "FactoryCandidate",
    "Objective",
    "SENSE",
    "CandidateScore",
    "score_candidate",
    "ParetoFront",
    "non_dominated_indices",
    "pareto_front",
    "grid",
    "Uncertainty",
    "RobustScore",
    "robust_score",
    "TopologyComparison",
    "compare_topologies",
    "dominated_classes",
    "DEFAULT_COMPARISON_OBJECTIVES",
]
