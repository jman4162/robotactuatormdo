"""Battery / DC source model (first-order).

A Thevenin source: an open-circuit pack voltage (supplied at evaluation time, not stored here)
behind an internal resistance, with a current limit. State-of-charge and RC dynamics are not
modeled.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Battery"]


@dataclass(frozen=True, slots=True)
class Battery:
    """Thevenin battery. SI units."""

    internal_resistance_ohm: float
    max_current_a: float = float("inf")
    mass_kg: float = 0.0

    def __post_init__(self) -> None:
        if self.internal_resistance_ohm < 0.0:
            raise ValueError("internal_resistance_ohm must be non-negative")
