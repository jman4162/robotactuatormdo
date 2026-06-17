"""Robot joint requirement schema.

A :class:`JointRequirement` is the apples-to-apples specification every candidate architecture
is evaluated against, so radial-flux, axial-flux, and commercial motors all face the same duty,
envelope, and electrical/thermal limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["JointRequirement"]


@dataclass(frozen=True, slots=True)
class JointRequirement:
    """Requirements for a single robot joint/actuator. All fields SI.

    Parameters
    ----------
    peak_torque_nm:
        Required output peak torque [N·m] (magnetic/current limited).
    continuous_rms_torque_nm:
        Required output continuous RMS torque [N·m] (thermal limited).
    max_speed_rad_s:
        Maximum output speed [rad/s].
    bus_voltage_v:
        Nominal DC bus voltage [V].
    max_phase_current_a_rms:
        Inverter/winding phase current limit [A_rms].
    envelope_outer_diameter_m:
        Maximum actuator outer diameter [m].
    envelope_axial_length_m:
        Maximum actuator axial length [m].
    max_mass_kg:
        Mass budget for the actuator [kg].
    ambient_temp_c:
        Ambient temperature for thermal evaluation [°C].
    target_bandwidth_hz:
        Desired output torque bandwidth [Hz]; ``None`` if unconstrained.
    duty_cycle_time_s / duty_cycle_torque_nm / duty_cycle_speed_rad_s:
        Optional parallel arrays describing the output duty cycle used for mission-level
        (RMS current, energy, thermal-recovery) metrics.
    cooling_available:
        Free-form tag for the available cooling path (e.g. ``"passive"``, ``"forced_air"``,
        ``"liquid_jacket"``); interpreted by the thermal layer.
    production_volume:
        Annual production volume, used by manufacturability/cost scoring; ``None`` if N/A.
    """

    peak_torque_nm: float
    continuous_rms_torque_nm: float
    max_speed_rad_s: float
    bus_voltage_v: float
    max_phase_current_a_rms: float
    envelope_outer_diameter_m: float
    envelope_axial_length_m: float
    max_mass_kg: float
    ambient_temp_c: float = 25.0
    target_bandwidth_hz: float | None = None
    duty_cycle_time_s: np.ndarray | None = None
    duty_cycle_torque_nm: np.ndarray | None = None
    duty_cycle_speed_rad_s: np.ndarray | None = None
    cooling_available: str = "passive"
    production_volume: int | None = None
