"""Design candidates: parameterized recipes that build a MotorModel stack.

A candidate is a *named recipe* (architecture class + a factory closure over named scalar
parameters) that builds one :class:`MotorModel` — a bare motor, an :class:`Actuator`, or a full
:class:`PowerStage`. Studies sample, score, and compare candidates without knowing how the
parameters route into geometry/gearbox/electronics — the factory owns that wiring, which keeps the
robust and comparison studies topology-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from robotactuatormdo.motors.protocols import MotorModel

__all__ = ["DesignCandidate", "FactoryCandidate"]


@runtime_checkable
class DesignCandidate(Protocol):
    """A named recipe that builds a MotorModel from a parameter vector."""

    name: str
    architecture_class: str

    def build(self) -> MotorModel: ...

    def perturb(self, overrides: Mapping[str, float]) -> DesignCandidate: ...


@dataclass(frozen=True, slots=True)
class FactoryCandidate:
    """Concrete :class:`DesignCandidate`: a factory closure over a named-scalar parameter dict.

    ``factory(params)`` returns a MotorModel. ``perturb`` returns a new candidate with overridden
    parameters (the only hook robust studies need). ``cost_usd`` is optional/informational and is
    read by studies via ``getattr(candidate, "cost_usd", None)``.
    """

    name: str
    architecture_class: str
    factory: Callable[[Mapping[str, float]], MotorModel]
    params: Mapping[str, float] = field(default_factory=dict)
    cost_usd: float | None = None

    def build(self) -> MotorModel:
        return self.factory(self.params)

    def perturb(self, overrides: Mapping[str, float]) -> FactoryCandidate:
        merged = {**self.params, **overrides}
        return FactoryCandidate(
            self.name, self.architecture_class, self.factory, merged, self.cost_usd
        )
