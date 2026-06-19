"""Inverter conduction and switching loss (TOPOLOGY brief §10.4).

First-order estimates for a three-phase two-level voltage-source bridge:
- conduction ``P_cond = n_conducting * I_rms**2 * R_ds_on`` — at any instant each of the three
  phase currents flows through one device, so ``n_conducting = 3`` by default.
- switching ``P_sw = n_switches * 0.5 * V_link * I_pk * (t_r + t_f) * f_sw`` — six devices each
  switch once per PWM period; ``I_pk`` is the switched (peak phase) current. This is a conservative
  linear-transition estimate, not a datasheet energy-curve model.
"""

from __future__ import annotations

__all__ = ["conduction_loss_w", "switching_loss_w"]


def conduction_loss_w(i_rms_a: float, r_ds_on_ohm: float, n_conducting: int = 3) -> float:
    """Inverter conduction loss [W]."""
    return n_conducting * i_rms_a**2 * r_ds_on_ohm


def switching_loss_w(
    v_link_v: float,
    i_pk_a: float,
    switching_time_s: float,
    f_sw_hz: float,
    n_switches: int = 6,
) -> float:
    """Inverter switching loss [W] for a linear transition of duration ``switching_time_s``."""
    return n_switches * 0.5 * v_link_v * i_pk_a * switching_time_s * f_sw_hz
