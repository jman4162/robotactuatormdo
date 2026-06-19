"""Robust evaluation under first-order parameter uncertainty (brief §6).

Perturb a candidate's named uncertain parameters, re-score a seeded Monte-Carlo ensemble, and
report the worst-tail (95th-percentile for minimize, 5th for maximize) objective values over the
**feasible** draws, plus the feasible fraction. Deterministic given the seed; uses seeded numpy
only (no wall-clock, no global RNG).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robotactuatormdo.requirements.joint import JointRequirement
from robotactuatormdo.studies.candidate import DesignCandidate
from robotactuatormdo.studies.scoring import SENSE, CandidateScore, Objective, score_candidate

__all__ = ["Uncertainty", "RobustScore", "robust_score"]


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """First-order uncertainty on one candidate parameter.

    ``kind`` in {"normal", "uniform"}. ``spread`` is a relative fraction by default (multiplicative
    on the nominal); with ``additive=True`` it is an absolute additive spread (e.g. ambient °C).
    """

    param: str
    kind: str = "normal"
    spread: float = 0.0
    additive: bool = False

    def sample(self, nominal: float, rng: np.random.Generator) -> float:
        if self.kind == "normal":
            z = rng.standard_normal()
            return nominal + self.spread * z if self.additive else nominal * (1.0 + self.spread * z)
        if self.kind == "uniform":
            u = rng.uniform(-1.0, 1.0)
            return nominal + self.spread * u if self.additive else nominal * (1.0 + self.spread * u)
        raise ValueError(f"unknown uncertainty kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class RobustScore:
    name: str
    architecture_class: str
    feasible_fraction: float
    robust_feasible: bool
    p95_objectives: dict[Objective, float]
    nominal: CandidateScore


def robust_score(
    candidate: DesignCandidate,
    requirement: JointRequirement,
    objectives: tuple[Objective, ...],
    uncertainties: Sequence[Uncertainty],
    *,
    n_samples: int = 256,
    confidence: float = 0.95,
    seed: int = 0,
) -> RobustScore:
    """Monte-Carlo robust score. Worst-tail percentile per objective over feasible draws."""
    rng = np.random.default_rng(seed)
    nominal = score_candidate(candidate, requirement, objectives)
    base = dict(candidate.params) if hasattr(candidate, "params") else {}

    feasible_count = 0
    collected: dict[Objective, list[float]] = {k: [] for k in objectives}
    for _ in range(n_samples):
        overrides = {
            u.param: u.sample(float(base.get(u.param, 0.0)), rng)
            for u in uncertainties
            if u.param in base
        }
        s = score_candidate(candidate.perturb(overrides), requirement, objectives)
        if s.feasible:
            feasible_count += 1
            for k, v in s.objectives.items():
                collected[k].append(v)

    frac = feasible_count / n_samples if n_samples else 0.0
    p95: dict[Objective, float] = {}
    for k, vals in collected.items():
        if not vals:
            continue
        q = 95.0 if SENSE[k] > 0 else 5.0  # worst tail in the natural direction
        p95[k] = float(np.percentile(vals, q))
    return RobustScore(
        name=candidate.name,
        architecture_class=candidate.architecture_class,
        feasible_fraction=frac,
        robust_feasible=frac >= confidence,
        p95_objectives=p95,
        nominal=nominal,
    )
