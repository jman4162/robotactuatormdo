"""Topology-agnostic evaluation: run a motor over a duty cycle, reduce to mission metrics.

This is the connective tissue of the framework. Given any :class:`MotorModel` and an output
:class:`DutyCycle`, it produces a single :class:`MissionResult` so radial, axial, and commercial
motors are compared on identical demand. The motor coupling here is an *ideal* gear ratio (η = 1);
real gearbox loss, backlash, and stiffness are modeled in a later phase.
"""

from __future__ import annotations

import numpy as np

from robotactuatormdo._numeric import trapezoid
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.requirements.duty_cycle import DutyCycle
from robotactuatormdo.results import MissionResult

__all__ = ["evaluate_over_duty_cycle"]


def evaluate_over_duty_cycle(
    motor: MotorModel,
    duty: DutyCycle,
    *,
    bus_voltage_v: float,
    ambient_temp_c: float = 25.0,
    gear_ratio: float = 1.0,
) -> MissionResult:
    """Evaluate ``motor`` at every duty-cycle sample and aggregate mission-level metrics.

    Output samples map to the motor shaft through an ideal reduction ``gear_ratio = G``:
    ``omega_motor = G * omega_out`` and ``tau_motor = tau_out / G``.

    Parameters
    ----------
    motor:
        Any object implementing :class:`MotorModel`.
    duty:
        Output (joint-shaft) duty cycle.
    bus_voltage_v:
        DC bus voltage applied at every operating point [V].
    ambient_temp_c:
        Ambient temperature for the thermal evaluation [°C].
    gear_ratio:
        Ideal reduction ratio ``G = omega_motor / omega_out`` (> 0); 1.0 is direct drive.
    """
    if gear_ratio <= 0.0:
        raise ValueError(f"gear_ratio must be positive, got {gear_ratio}")

    points = tuple(
        motor.evaluate_operating_point(
            torque_nm=float(tau) / gear_ratio,
            speed_rad_s=float(omega) * gear_ratio,
            bus_voltage_v=bus_voltage_v,
            ambient_temp_c=ambient_temp_c,
        )
        for tau, omega in zip(duty.torque_nm, duty.speed_rad_s, strict=True)
    )

    currents = np.array([p.phase_current_a_rms for p in points], dtype=float)
    losses = np.array([p.total_loss_w for p in points], dtype=float)
    feasible = np.array([p.feasibility.feasible for p in points], dtype=bool)

    rms_current = float(np.sqrt(trapezoid(currents**2, duty.time_s) / duty.duration_s))
    loss_energy = trapezoid(losses, duty.time_s)
    mech_energy = duty.mechanical_energy_j

    return MissionResult(
        rms_phase_current_a=rms_current,
        peak_phase_current_a=float(np.max(currents)),
        mechanical_energy_j=mech_energy,
        electrical_energy_j=mech_energy + loss_energy,
        loss_energy_j=loss_energy,
        peak_winding_temp_c=float(np.max([p.winding_temp_c for p in points])),
        peak_magnet_temp_c=float(np.max([p.magnet_temp_c for p in points])),
        all_feasible=bool(np.all(feasible)),
        fraction_feasible=float(np.mean(feasible)),
        points=points,
    )
