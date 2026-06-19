"""First-order BOM cost from the active-material mass breakdown.

A screening estimate: material cost = mass x $/kg per material, plus a processing term modeled as a
fraction of material cost (scaled by a process multiplier) plus a fixed assembly floor. Not a DFM
model — values are order-of-magnitude for ranking candidates, consistent with the package's
first-order stance.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotactuatormdo.materials.copper import Conductor
from robotactuatormdo.materials.electrical_steel import ElectricalSteel
from robotactuatormdo.materials.magnets import MagnetMaterial

__all__ = ["CostBreakdown", "CostModel", "bom_cost"]

_PROCESSING_FRACTION = 0.4  # processing ~ 40% of active-material cost (screening heuristic)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    magnet_material_usd: float
    copper_material_usd: float
    iron_material_usd: float
    processing_usd: float

    @property
    def material_usd(self) -> float:
        return self.magnet_material_usd + self.copper_material_usd + self.iron_material_usd

    @property
    def total_usd(self) -> float:
        return self.material_usd + self.processing_usd


@dataclass(frozen=True, slots=True)
class CostModel:
    process_cost_multiplier: float = 1.0
    assembly_usd: float = 0.0


_DEFAULT_COST_MODEL = CostModel()


def bom_cost(
    magnet_mass_kg: float,
    copper_mass_kg: float,
    iron_mass_kg: float,
    magnet: MagnetMaterial,
    steel: ElectricalSteel,
    copper: Conductor,
    model: CostModel | None = None,
) -> CostBreakdown:
    """First-order BOM cost from active-material masses and material $/kg."""
    model = model or _DEFAULT_COST_MODEL
    magnet_usd = magnet_mass_kg * magnet.cost_per_kg
    copper_usd = copper_mass_kg * copper.cost_per_kg
    iron_usd = iron_mass_kg * steel.cost_per_kg
    material = magnet_usd + copper_usd + iron_usd
    processing = (
        model.process_cost_multiplier * _PROCESSING_FRACTION * material + model.assembly_usd
    )
    return CostBreakdown(magnet_usd, copper_usd, iron_usd, processing)
