"""Conductor (winding) material records.

Resistivity rises with temperature at ``temp_coeff_per_c`` (~0.39 %/°C for copper), the dominant
reason continuous torque is thermally limited.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Conductor", "COPPER", "load"]


@dataclass(frozen=True, slots=True)
class Conductor:
    """A winding conductor. SI units; ``temp_coeff_per_c`` is a fraction per °C from 20 °C."""

    name: str
    resistivity_20_ohm_m: float
    temp_coeff_per_c: float
    density_kg_m3: float

    def resistivity_at(self, temp_c: float) -> float:
        """Resistivity [Ohm·m] at ``temp_c`` (linear model referenced to 20 °C)."""
        return self.resistivity_20_ohm_m * (1.0 + self.temp_coeff_per_c * (temp_c - 20.0))


COPPER = Conductor(
    name="Cu",
    resistivity_20_ohm_m=1.68e-8,
    temp_coeff_per_c=0.00393,
    density_kg_m3=8960.0,
)

_REGISTRY: dict[str, Conductor] = {COPPER.name: COPPER}


def load(name: str) -> Conductor:
    """Return the built-in conductor record named ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown conductor '{name}'; available: {sorted(_REGISTRY)}") from None
