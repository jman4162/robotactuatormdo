"""DC cable model (first-order resistive drop)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Cable"]


@dataclass(frozen=True, slots=True)
class Cable:
    """A DC supply cable. SI units."""

    resistance_ohm: float
    max_current_a: float = float("inf")
    mass_kg: float = 0.0

    def __post_init__(self) -> None:
        if self.resistance_ohm < 0.0:
            raise ValueError("resistance_ohm must be non-negative")
