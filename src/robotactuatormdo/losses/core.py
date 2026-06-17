"""Steinmetz core-loss model.

``P_core = mass * (k_hyst * f * B^alpha + k_eddy * f^2 * B^2)`` — hysteresis term plus classical
eddy term, with ``f`` the electrical frequency [Hz] and ``B`` a representative peak flux density
[T]. This is a lumped first-order estimate over the iron mass, not a region-resolved model.
"""

from __future__ import annotations

__all__ = ["steinmetz_core_loss_w"]


def steinmetz_core_loss_w(
    mass_kg: float,
    freq_hz: float,
    b_peak_t: float,
    k_hyst: float,
    alpha: float,
    k_eddy: float,
) -> float:
    """Core loss [W] for ``mass_kg`` of iron at electrical frequency ``freq_hz``."""
    if freq_hz <= 0.0:
        return 0.0
    hysteresis = k_hyst * freq_hz * b_peak_t**alpha
    eddy = k_eddy * freq_hz**2 * b_peak_t**2
    return mass_kg * (hysteresis + eddy)
