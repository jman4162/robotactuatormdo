"""Soft-magnetic-composite (SMC) material records.

SMC enables 3D flux paths (relevant for axial/transverse geometries) with low eddy loss but lower
saturation and a finite permeability vs laminations. The Steinmetz fields are layout-compatible
with :class:`~robotactuatormdo.materials.electrical_steel.ElectricalSteel`, so an ``SMCMaterial``
can be passed where a steel record is expected (it carries the Steinmetz fields sizing consumes).
``relative_permeability`` is stored for future MMF-drop refinement; sizing currently treats iron as
infinitely permeable.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SMCMaterial", "SMC_NOMINAL", "load"]


@dataclass(frozen=True, slots=True)
class SMCMaterial:
    """A soft-magnetic-composite grade. SI; Steinmetz coeffs as in ElectricalSteel."""

    name: str
    b_sat_t: float
    relative_permeability: float
    k_hyst: float
    alpha: float
    k_eddy: float
    density_kg_m3: float
    cost_per_kg: float = 0.0


# Nominal SMC (Somaloy-class): lower B_sat and mu_r, much lower eddy loss, higher hysteresis.
SMC_NOMINAL = SMCMaterial(
    name="SMC-nominal",
    b_sat_t=1.5,
    relative_permeability=500.0,
    k_hyst=0.04,
    alpha=1.8,
    k_eddy=1.0e-5,
    density_kg_m3=7400.0,
    cost_per_kg=10.0,
)

_REGISTRY: dict[str, SMCMaterial] = {SMC_NOMINAL.name: SMC_NOMINAL}


def load(name: str) -> SMCMaterial:
    """Return the built-in SMC record named ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown SMC '{name}'; available: {sorted(_REGISTRY)}") from None
