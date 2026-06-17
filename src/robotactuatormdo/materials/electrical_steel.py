"""Electrical-steel / lamination records.

Carries a saturation flux density (for first-order saturation flags) and Steinmetz core-loss
coefficients used by :func:`robotactuatormdo.losses.core.steinmetz_core_loss_w`:
``P = mass * (k_hyst * f * B**alpha + k_eddy * f**2 * B**2)``. Coefficients are order-of-magnitude
figures for thin non-grain-oriented laminations — calibrate against a datasheet loss curve for
design work.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ElectricalSteel", "M250_35A", "load"]


@dataclass(frozen=True, slots=True)
class ElectricalSteel:
    """A laminated electrical-steel grade. SI units; Steinmetz coeffs give loss in W when
    multiplied by mass [kg], with ``f`` in Hz and ``B`` in T."""

    name: str
    b_sat_t: float
    k_hyst: float
    alpha: float
    k_eddy: float
    density_kg_m3: float


# Nominal M250-35A non-grain-oriented lamination.
M250_35A = ElectricalSteel(
    name="M250-35A",
    b_sat_t=1.8,
    k_hyst=0.02,
    alpha=1.8,
    k_eddy=8.0e-5,
    density_kg_m3=7650.0,
)

_REGISTRY: dict[str, ElectricalSteel] = {M250_35A.name: M250_35A}


def load(name: str) -> ElectricalSteel:
    """Return the built-in steel record named ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown steel '{name}'; available: {sorted(_REGISTRY)}") from None
