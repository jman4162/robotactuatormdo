"""Permanent-magnet material records.

First-order datasheet-backed records with the fields the radial sizing model needs. Remanence
``Br`` falls with temperature at ``temp_coeff_br_per_c`` (a negative %/°C for NdFeB); recoil
permeability ``recoil_mu_r`` closes the magnetic circuit. Values are nominal grade figures, not a
specific vendor lot — calibrate against a datasheet for design work.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MagnetMaterial", "NDFEB_N42", "load"]


@dataclass(frozen=True, slots=True)
class MagnetMaterial:
    """A permanent-magnet grade. SI units; ``temp_coeff_br_per_c`` is a fraction per °C."""

    name: str
    br_t: float
    temp_coeff_br_per_c: float
    recoil_mu_r: float
    max_op_temp_c: float
    density_kg_m3: float
    resistivity_ohm_m: float
    cost_per_kg: float = 0.0

    def br_at(self, temp_c: float) -> float:
        """Remanence [T] at ``temp_c`` (linear reversible model referenced to 20 °C)."""
        return self.br_t * (1.0 + self.temp_coeff_br_per_c * (temp_c - 20.0))


# Nominal sintered NdFeB ~N42 (Br ~1.30 T, ~-0.12 %/°C, recoil mu_r ~1.05).
NDFEB_N42 = MagnetMaterial(
    name="NdFeB-N42",
    br_t=1.30,
    temp_coeff_br_per_c=-0.0012,
    recoil_mu_r=1.05,
    max_op_temp_c=80.0,
    density_kg_m3=7500.0,
    resistivity_ohm_m=1.4e-6,
    cost_per_kg=80.0,  # nominal sintered NdFeB, order-of-magnitude
)

_REGISTRY: dict[str, MagnetMaterial] = {NDFEB_N42.name: NDFEB_N42}


def load(name: str) -> MagnetMaterial:
    """Return the built-in magnet record named ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown magnet '{name}'; available: {sorted(_REGISTRY)}") from None
