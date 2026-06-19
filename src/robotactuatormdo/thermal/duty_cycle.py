"""Transient thermal integration of a motor/actuator over a duty cycle.

Two decoupled passes: (1) a **loss pass** that calls the (validated) motor model at each duty
sample to obtain per-channel losses, and (2) an **integration pass** that injects those losses at
the right thermal nodes and advances node temperatures in time with backward Euler.

First-order limitation (documented): losses are **quasi-static** — evaluated at each point's own
self-heated steady winding temperature — while only the node temperatures are integrated in time.
Per-timestep feedback of the transient temperature into the loss model is deferred.
"""

from __future__ import annotations

import numpy as np

from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.requirements.duty_cycle import DutyCycle
from robotactuatormdo.results import ThermalHistory
from robotactuatormdo.thermal.lumped_network import ThermalNetwork, solve_transient

__all__ = ["integrate"]


def _ambient_temp_c(network: ThermalNetwork) -> float:
    bt = [float(n.fixed_temp_c) for n in network.nodes if n.is_boundary]
    return float(np.mean(bt))


def integrate(
    motor: MotorModel,
    duty: DutyCycle,
    network: ThermalNetwork,
    *,
    bus_voltage_v: float,
    gear_ratio: float = 1.0,
    initial_temps_c: dict[str, float] | None = None,
) -> ThermalHistory:
    """Integrate node temperatures over ``duty``. Ambient is the network's boundary temperature.

    Loss channels map to nodes: copper -> ``winding``; core -> ``stator`` if present else
    ``winding``; magnet eddy -> ``magnet`` if present. The output duty cycle is mapped to the motor
    shaft through an ideal ``gear_ratio`` (matching :func:`evaluate_over_duty_cycle`).
    """
    if gear_ratio <= 0.0:
        raise ValueError(f"gear_ratio must be positive, got {gear_ratio}")
    free = set(network.free_node_names)
    core_node = "stator" if "stator" in free else "winding"
    ambient_c = _ambient_temp_c(network)

    power_series: list[dict[str, float]] = []
    for tau, omega in zip(duty.torque_nm, duty.speed_rad_s, strict=True):
        res = motor.evaluate_operating_point(
            torque_nm=float(tau) / gear_ratio,
            speed_rad_s=float(omega) * gear_ratio,
            bus_voltage_v=bus_voltage_v,
            ambient_temp_c=ambient_c,
        )
        power: dict[str, float] = {"winding": res.copper_loss_w}
        power[core_node] = power.get(core_node, 0.0) + res.core_loss_w
        if "magnet" in free:
            power["magnet"] = res.magnet_eddy_loss_w
        power_series.append(power)

    temps = solve_transient(network, duty.time_s, power_series, initial_temps_c)
    return ThermalHistory(
        time_s=np.asarray(duty.time_s, dtype=float),
        node_names=network.free_node_names,
        node_temps_c=temps,
    )
