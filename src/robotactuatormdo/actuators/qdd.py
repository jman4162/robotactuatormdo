"""Quasi-direct-drive (QDD) actuator assembly.

QDD couples a high-torque motor to a *low-ratio* (typically single-stage planetary) transmission,
trading a little torque multiplication for high backdrivability and bandwidth — the common modern
robotics actuator. This is a thin convenience over :class:`Actuator` + a planetary :class:`Gearbox`.
"""

from __future__ import annotations

from typing import Any

from robotactuatormdo.actuators.actuator import Actuator
from robotactuatormdo.actuators.gearbox import planetary
from robotactuatormdo.motors.protocols import MotorModel

__all__ = ["quasi_direct_drive"]


def quasi_direct_drive(motor: MotorModel, ratio: float = 6.0, **gearbox_kwargs: Any) -> Actuator:
    """Wrap ``motor`` in a low-ratio planetary gearbox to form a QDD actuator."""
    return Actuator(motor, planetary(ratio, **gearbox_kwargs))
