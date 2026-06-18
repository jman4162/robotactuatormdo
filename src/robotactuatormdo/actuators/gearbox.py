"""Gearbox model and type presets (TOPOLOGY brief §12.2).

Conventions (reduction ``G = omega_motor / omega_out``, direct drive ``G = 1``):
- output torque ``T_out = eta * G * (T_motor - T_drag_in)``
- input torque  ``T_motor = T_out / (eta * G) + T_drag_in``  (drag is input-referred)
- gear dissipation ``P_loss = T_motor * omega_motor - T_out * omega_out`` (clamped >= 0)
- output-referred inertia ``J_out = G**2 * (J_motor + J_gb_in)``
- first-order backdrive torque at the output ``T_bd = G * T_drag_in / eta_back`` (ignores motor
  cogging/detent, which the motor model does not represent).

The presets carry *typical first-order* figures for each transmission family; override fields for a
specific unit. They are screening values, not datasheet specifications.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotactuatormdo.actuators.reflected_inertia import reflected_inertia

__all__ = ["Gearbox", "direct_drive", "planetary", "harmonic", "cycloidal", "belt"]


@dataclass(frozen=True, slots=True)
class Gearbox:
    """A speed-reduction transmission. SI; ``inertia_kg_m2`` is referred to the input shaft."""

    ratio: float
    forward_efficiency: float
    mass_kg: float = 0.0
    inertia_kg_m2: float = 0.0
    backlash_rad: float = 0.0
    torsional_stiffness_nm_per_rad: float = 1e9
    max_output_torque_nm: float = float("inf")
    no_load_drag_torque_nm: float = 0.0
    backdrive_efficiency: float | None = None

    def __post_init__(self) -> None:
        if self.ratio <= 0.0:
            raise ValueError(f"ratio must be positive, got {self.ratio}")
        if not 0.0 < self.forward_efficiency <= 1.0:
            raise ValueError(f"forward_efficiency must be in (0, 1], got {self.forward_efficiency}")
        if self.backdrive_efficiency is None:
            object.__setattr__(self, "backdrive_efficiency", self.forward_efficiency)

    def input_speed(self, output_speed_rad_s: float) -> float:
        """Motor-shaft speed for a given output speed."""
        return self.ratio * output_speed_rad_s

    def input_torque(self, output_torque_nm: float) -> float:
        """Motor torque needed to deliver ``output_torque_nm`` at the output."""
        return (
            output_torque_nm / (self.forward_efficiency * self.ratio)
            + self.no_load_drag_torque_nm
        )

    def output_torque(self, motor_torque_nm: float) -> float:
        """Output torque produced by ``motor_torque_nm`` at the input."""
        return (
            self.forward_efficiency * self.ratio * (motor_torque_nm - self.no_load_drag_torque_nm)
        )

    def gear_loss_w(self, output_torque_nm: float, output_speed_rad_s: float) -> float:
        """Gearbox dissipation [W] at an output operating point (clamped >= 0)."""
        t_motor = self.input_torque(output_torque_nm)
        w_motor = self.input_speed(output_speed_rad_s)
        p_in = t_motor * w_motor
        p_out = output_torque_nm * output_speed_rad_s
        return max(p_in - p_out, 0.0)

    def backdrive_torque_nm(self) -> float:
        """Output torque required to backdrive the input (first-order, drag-limited)."""
        return self.ratio * self.no_load_drag_torque_nm / self.backdrive_efficiency

    def reflected_inertia_at_output(self, motor_inertia_kg_m2: float) -> float:
        """Output-referred inertia ``G**2 * (J_motor + J_gb_in)`` [kg·m²]."""
        return reflected_inertia(motor_inertia_kg_m2 + self.inertia_kg_m2, self.ratio)


def direct_drive(efficiency: float = 0.99) -> Gearbox:
    """No reduction (G = 1): the joint is the motor shaft."""
    return Gearbox(ratio=1.0, forward_efficiency=efficiency, torsional_stiffness_nm_per_rad=1e12)


def planetary(
    ratio: float,
    *,
    mass_kg: float = 0.15,
    max_output_torque_nm: float = float("inf"),
    efficiency: float = 0.97,
) -> Gearbox:
    """Single-stage planetary: high efficiency, low-moderate ratio, modest backlash."""
    return Gearbox(
        ratio=ratio,
        forward_efficiency=efficiency,
        mass_kg=mass_kg,
        inertia_kg_m2=2e-6,
        backlash_rad=3e-3,
        torsional_stiffness_nm_per_rad=2e4,
        max_output_torque_nm=max_output_torque_nm,
        no_load_drag_torque_nm=0.02,
    )


def harmonic(
    ratio: float,
    *,
    mass_kg: float = 0.30,
    max_output_torque_nm: float = float("inf"),
    efficiency: float = 0.80,
) -> Gearbox:
    """Strain-wave (harmonic) drive: high ratio, near-zero backlash, high stiffness, lower eta."""
    return Gearbox(
        ratio=ratio,
        forward_efficiency=efficiency,
        mass_kg=mass_kg,
        inertia_kg_m2=3e-6,
        backlash_rad=1e-4,
        torsional_stiffness_nm_per_rad=1e5,
        max_output_torque_nm=max_output_torque_nm,
        no_load_drag_torque_nm=0.05,
    )


def cycloidal(
    ratio: float,
    *,
    mass_kg: float = 0.35,
    max_output_torque_nm: float = float("inf"),
    efficiency: float = 0.85,
) -> Gearbox:
    """Cycloidal drive: high ratio, low backlash, high shock tolerance and stiffness."""
    return Gearbox(
        ratio=ratio,
        forward_efficiency=efficiency,
        mass_kg=mass_kg,
        inertia_kg_m2=4e-6,
        backlash_rad=3e-4,
        torsional_stiffness_nm_per_rad=8e4,
        max_output_torque_nm=max_output_torque_nm,
        no_load_drag_torque_nm=0.04,
    )


def belt(
    ratio: float,
    *,
    mass_kg: float = 0.10,
    max_output_torque_nm: float = float("inf"),
    efficiency: float = 0.95,
) -> Gearbox:
    """Belt/pulley reduction: efficient and quiet, but compliant (low stiffness)."""
    return Gearbox(
        ratio=ratio,
        forward_efficiency=efficiency,
        mass_kg=mass_kg,
        inertia_kg_m2=1e-6,
        backlash_rad=1e-3,
        torsional_stiffness_nm_per_rad=5e3,
        max_output_torque_nm=max_output_torque_nm,
        no_load_drag_torque_nm=0.01,
    )
