"""Pareto-front construction over scored candidates.

The core is a dependency-free non-dominated sort. ``pymoo`` (optional ``opt`` extra) is only used
as an *optional* sampler/driver for large continuous sweeps — never required, so the front
definition stays deterministic and testable.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from robotactuatormdo.studies.candidate import FactoryCandidate
from robotactuatormdo.studies.scoring import CandidateScore, Objective

__all__ = ["non_dominated_indices", "ParetoFront", "pareto_front", "grid"]


def non_dominated_indices(points: np.ndarray) -> list[int]:
    """Indices of the non-dominated (Pareto-minimal) rows of ``points`` (shape ``(n, m)``).

    Minimization in every column. ``a`` dominates ``b`` iff ``a <= b`` elementwise and ``a < b`` in
    at least one column. Identical rows are mutually non-dominating, so duplicates are all kept.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


@dataclass(frozen=True, slots=True)
class ParetoFront:
    objective_keys: tuple[Objective, ...]
    all_scores: tuple[CandidateScore, ...]
    front_indices: tuple[int, ...]

    @property
    def front(self) -> tuple[CandidateScore, ...]:
        return tuple(self.all_scores[i] for i in self.front_indices)

    @property
    def dominated_indices(self) -> tuple[int, ...]:
        front = set(self.front_indices)
        return tuple(i for i in range(len(self.all_scores)) if i not in front)


def pareto_front(
    scores: Sequence[CandidateScore],
    objective_keys: tuple[Objective, ...],
    *,
    feasible_only: bool = True,
) -> ParetoFront:
    """Non-dominated front of ``scores`` over ``objective_keys``.

    Keys absent from any considered score are dropped (so all rows are comparable).
    """
    considered = [s for s in scores if s.feasible] if feasible_only else list(scores)
    if not considered:
        return ParetoFront(objective_keys, tuple(considered), ())
    keys = tuple(k for k in objective_keys if all(k in s.objectives for s in considered))
    if not keys:
        raise ValueError("no objective key is present in every considered score")
    matrix = np.array([s.vector(keys) for s in considered], dtype=float)
    front_local = non_dominated_indices(matrix)
    return ParetoFront(keys, tuple(considered), tuple(front_local))


def grid(
    name_prefix: str,
    architecture_class: str,
    factory: Callable[[Mapping[str, float]], object],
    axes: Mapping[str, Sequence[float]],
) -> list[FactoryCandidate]:
    """Cartesian product of named parameter ``axes`` into ``FactoryCandidate``s (deterministic)."""
    names = list(axes)
    out: list[FactoryCandidate] = []
    for i, combo in enumerate(itertools.product(*(axes[n] for n in names))):
        params = dict(zip(names, combo, strict=True))
        out.append(FactoryCandidate(f"{name_prefix}_{i}", architecture_class, factory, params))
    return out
