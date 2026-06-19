"""Insulation-class thermal limits (IEC 60085).

The maximum winding temperature is set by the insulation class; ``CLASS_F`` (155 °C) matches the
sizing default.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["InsulationClass", "CLASS_B", "CLASS_F", "CLASS_H", "load"]


@dataclass(frozen=True, slots=True)
class InsulationClass:
    """A winding insulation class. ``max_temp_c`` is the rated hot-spot limit [°C]."""

    name: str
    max_temp_c: float


CLASS_B = InsulationClass("B", 130.0)
CLASS_F = InsulationClass("F", 155.0)
CLASS_H = InsulationClass("H", 180.0)

_REGISTRY: dict[str, InsulationClass] = {c.name: c for c in (CLASS_B, CLASS_F, CLASS_H)}


def load(name: str) -> InsulationClass:
    """Return the built-in insulation class named ``name`` (``"B"``/``"F"``/``"H"``)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown insulation class '{name}'; available: {sorted(_REGISTRY)}"
        ) from None
