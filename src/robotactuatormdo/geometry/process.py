"""Manufacturing/winding process options (first-order screening).

A ``WindingProcess`` maps a manufacturing choice (round-wire / hairpin / litz / PCB-coreless) to the
first-order quantities sizing already uses: slot fill, winding factor, achievable current density,
an AC-loss multiplier, and cost/manufacturability multipliers. ``ROUND_WIRE`` reproduces the
package's historical defaults exactly (a no-op), so passing it leaves sizing unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

__all__ = [
    "WindingProcess",
    "ROUND_WIRE",
    "HAIRPIN",
    "LITZ",
    "PCB_CORELESS",
    "geometry_for_process",
]

if TYPE_CHECKING:  # avoid a runtime import cycle with radial_flux
    from robotactuatormdo.geometry.radial_flux import RadialPMGeometry


@dataclass(frozen=True, slots=True)
class WindingProcess:
    """First-order manufacturing process record. SI; multipliers are dimensionless."""

    name: str
    slot_fill: float
    winding_factor: float
    max_current_density_a_per_mm2: float
    ac_loss_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    manufacturability_score: float = 1.0


# Baseline == the historical size_radial_pm defaults (no-op).
ROUND_WIRE = WindingProcess("round-wire", slot_fill=0.45, winding_factor=0.95,
                            max_current_density_a_per_mm2=6.0)
HAIRPIN = WindingProcess("hairpin", slot_fill=0.60, winding_factor=0.95,
                         max_current_density_a_per_mm2=7.0, ac_loss_multiplier=1.3,
                         cost_multiplier=1.2, manufacturability_score=0.8)
LITZ = WindingProcess("litz", slot_fill=0.35, winding_factor=0.95,
                      max_current_density_a_per_mm2=6.0, ac_loss_multiplier=0.6,
                      cost_multiplier=1.8, manufacturability_score=0.6)
PCB_CORELESS = WindingProcess("pcb-coreless", slot_fill=0.25, winding_factor=0.90,
                              max_current_density_a_per_mm2=5.0, ac_loss_multiplier=0.8,
                              cost_multiplier=1.5, manufacturability_score=0.7)


def geometry_for_process(geom: RadialPMGeometry, process: WindingProcess) -> RadialPMGeometry:
    """Return a copy of ``geom`` with slot fill and winding factor taken from ``process``."""
    return replace(geom, slot_fill=process.slot_fill, winding_factor=process.winding_factor)
