"""Inverter device model and technology presets.

Wraps the loss formulas in :mod:`robotactuatormdo.losses.inverter` with a device record. ``R_ds_on``
is taken as a fixed (datasheet hot) value — there is no inverter thermal node yet (Phase 5), so the
device temperature is not coupled. Presets carry typical first-order figures per device family
(Si MOSFET / GaN / SiC); override for a specific part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robotactuatormdo.losses.inverter import conduction_loss_w, switching_loss_w

__all__ = ["Inverter", "si_mosfet", "gan", "sic"]

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class Inverter:
    """A three-phase two-level voltage-source inverter. SI units."""

    r_ds_on_ohm: float
    switching_time_s: float  # t_r + t_f
    f_sw_hz: float
    max_phase_current_a_rms: float = float("inf")
    n_switches: int = 6
    n_conducting: int = 3
    mass_kg: float = 0.0

    def __post_init__(self) -> None:
        if self.r_ds_on_ohm < 0.0 or self.switching_time_s < 0.0 or self.f_sw_hz <= 0.0:
            raise ValueError("inverter parameters must be non-negative (f_sw positive)")

    def conduction_loss_w(self, i_rms_a: float) -> float:
        return conduction_loss_w(i_rms_a, self.r_ds_on_ohm, self.n_conducting)

    def switching_loss_w(self, v_link_v: float, i_pk_a: float) -> float:
        return switching_loss_w(
            v_link_v, i_pk_a, self.switching_time_s, self.f_sw_hz, self.n_switches
        )

    def loss_w(self, i_rms_a: float, v_link_v: float) -> float:
        """Total inverter loss [W] at a phase RMS current and DC-link voltage."""
        i_pk = i_rms_a * _SQRT2
        return self.conduction_loss_w(i_rms_a) + self.switching_loss_w(v_link_v, i_pk)


def si_mosfet(max_phase_current_a_rms: float = float("inf")) -> Inverter:
    """Silicon MOSFET: low R_ds(on) at low voltage, slower switching."""
    return Inverter(
        r_ds_on_ohm=3e-3,
        switching_time_s=150e-9,
        f_sw_hz=20e3,
        max_phase_current_a_rms=max_phase_current_a_rms,
        mass_kg=0.05,
    )


def gan(max_phase_current_a_rms: float = float("inf")) -> Inverter:
    """GaN FET: low R_ds(on) and very fast switching, enabling higher f_sw."""
    return Inverter(
        r_ds_on_ohm=5e-3,
        switching_time_s=20e-9,
        f_sw_hz=60e3,
        max_phase_current_a_rms=max_phase_current_a_rms,
        mass_kg=0.04,
    )


def sic(max_phase_current_a_rms: float = float("inf")) -> Inverter:
    """SiC MOSFET: fast switching, suited to higher-voltage buses."""
    return Inverter(
        r_ds_on_ohm=8e-3,
        switching_time_s=40e-9,
        f_sw_hz=40e3,
        max_phase_current_a_rms=max_phase_current_a_rms,
        mass_kg=0.05,
    )
