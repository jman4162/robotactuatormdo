"""Stable result and property types shared across all motor/actuator backends.

Units discipline (enforced project-wide): every quantity stored here is **SI** and, for
electrical machine constants, expressed as *phase* quantities. Datasheet conventions
(line-to-line vs phase, peak vs RMS, trapezoidal vs sinusoidal Ke/Kt) are converted only at
the API boundary, never stored internally. Mixing, e.g., a line-line-RMS back-EMF constant with
a phase-current torque constant is a classic comparison bug this rule exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FeasibilityFlags",
    "MotorOperatingResult",
    "MassProperties",
    "TorqueSpeedEnvelope",
    "EfficiencyMap",
]


@dataclass(frozen=True, slots=True)
class FeasibilityFlags:
    """Per-constraint feasibility at a single operating point. ``True`` means *satisfied*."""

    voltage_ok: bool = True
    current_ok: bool = True
    winding_temp_ok: bool = True
    magnet_temp_ok: bool = True
    saturation_ok: bool = True
    demag_ok: bool = True

    @property
    def feasible(self) -> bool:
        """All modeled constraints satisfied."""
        return (
            self.voltage_ok
            and self.current_ok
            and self.winding_temp_ok
            and self.magnet_temp_ok
            and self.saturation_ok
            and self.demag_ok
        )


@dataclass(frozen=True, slots=True)
class MotorOperatingResult:
    """Outcome of evaluating a motor at one (torque, speed, bus voltage, ambient) point.

    All fields SI: torque [N·m], speed [rad/s], losses/power [W], current [A_rms, phase],
    voltage [V, phase], temperatures [°C].
    """

    torque_nm: float
    speed_rad_s: float
    phase_current_a_rms: float
    phase_voltage_v_rms: float
    copper_loss_w: float
    core_loss_w: float
    magnet_eddy_loss_w: float
    mechanical_loss_w: float
    winding_temp_c: float
    magnet_temp_c: float
    feasibility: FeasibilityFlags = field(default_factory=FeasibilityFlags)

    @property
    def mechanical_power_w(self) -> float:
        return self.torque_nm * self.speed_rad_s

    @property
    def total_loss_w(self) -> float:
        return (
            self.copper_loss_w
            + self.core_loss_w
            + self.magnet_eddy_loss_w
            + self.mechanical_loss_w
        )

    @property
    def efficiency(self) -> float:
        """Output/input efficiency in [0, 1]; 0 if no net power is delivered."""
        p_out = self.mechanical_power_w
        p_in = p_out + self.total_loss_w
        if p_in <= 0.0:
            return 0.0
        return max(0.0, min(1.0, p_out / p_in))


@dataclass(frozen=True, slots=True)
class MassProperties:
    """Mass and rotary inertia breakdown. Mass [kg], inertia [kg·m²] about the spin axis."""

    total_mass_kg: float
    rotor_inertia_kg_m2: float
    active_mass_kg: float | None = None
    magnet_mass_kg: float | None = None
    copper_mass_kg: float | None = None
    iron_mass_kg: float | None = None


@dataclass(frozen=True, slots=True)
class TorqueSpeedEnvelope:
    """Sampled feasible torque-speed boundary.

    ``speed_rad_s`` and ``peak_torque_nm``/``continuous_torque_nm`` are parallel 1-D arrays.
    """

    speed_rad_s: np.ndarray
    peak_torque_nm: np.ndarray
    continuous_torque_nm: np.ndarray


@dataclass(frozen=True, slots=True)
class EfficiencyMap:
    """Efficiency over a torque-speed grid.

    ``efficiency`` has shape ``(len(torque_nm), len(speed_rad_s))`` with values in [0, 1];
    ``np.nan`` marks infeasible cells.
    """

    speed_rad_s: np.ndarray
    torque_nm: np.ndarray
    efficiency: np.ndarray
