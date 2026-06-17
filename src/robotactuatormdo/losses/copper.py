"""Copper (winding) loss and temperature-dependent phase resistance.

Convention: ``resistance_ohm`` is the **per-phase** resistance and ``i_rms_a`` the per-phase RMS
current, so a three-phase machine dissipates ``3 * I_rms^2 * R_phase``.
"""

from __future__ import annotations

__all__ = ["phase_resistance_at", "copper_loss_w"]


def phase_resistance_at(r_s_20_ohm: float, temp_coeff_per_c: float, temp_c: float) -> float:
    """Phase resistance [Ohm] at ``temp_c`` (linear model referenced to 20 °C)."""
    return r_s_20_ohm * (1.0 + temp_coeff_per_c * (temp_c - 20.0))


def copper_loss_w(i_rms_a: float, resistance_ohm: float, n_phases: int = 3) -> float:
    """Total copper loss [W] = ``n_phases * I_rms^2 * R_phase``."""
    return n_phases * i_rms_a**2 * resistance_ohm
