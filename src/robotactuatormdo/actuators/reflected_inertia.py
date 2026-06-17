"""Gearbox-reflected inertia.

A gearbox of ratio ``G`` (motor-speed / output-speed) multiplies the motor's rotary inertia seen
at the output by ``G**2``. This is why high gear ratios hurt backdrivability and collision
behavior — a small motor inertia becomes large at the joint.
"""

from __future__ import annotations

__all__ = ["reflected_inertia"]


def reflected_inertia(motor_inertia_kg_m2: float, gear_ratio: float) -> float:
    """Motor inertia reflected to the gearbox output: ``G**2 * J_motor`` [kg·m²].

    Parameters
    ----------
    motor_inertia_kg_m2:
        Rotor inertia about the spin axis [kg·m²].
    gear_ratio:
        Reduction ratio ``G = omega_motor / omega_output`` (dimensionless, > 0).
    """
    if motor_inertia_kg_m2 < 0.0:
        raise ValueError(f"motor_inertia_kg_m2 must be non-negative, got {motor_inertia_kg_m2}")
    if gear_ratio <= 0.0:
        raise ValueError(f"gear_ratio must be positive, got {gear_ratio}")
    return gear_ratio**2 * motor_inertia_kg_m2
