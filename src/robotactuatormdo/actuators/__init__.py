"""Actuator-level coupling: gearboxes, QDD, reflected inertia."""

from __future__ import annotations

from robotactuatormdo.actuators.actuator import Actuator, ActuatorProperties
from robotactuatormdo.actuators.gearbox import (
    Gearbox,
    belt,
    cycloidal,
    direct_drive,
    harmonic,
    planetary,
)
from robotactuatormdo.actuators.qdd import quasi_direct_drive
from robotactuatormdo.actuators.reflected_inertia import reflected_inertia

__all__ = [
    "Actuator",
    "ActuatorProperties",
    "Gearbox",
    "direct_drive",
    "planetary",
    "harmonic",
    "cycloidal",
    "belt",
    "quasi_direct_drive",
    "reflected_inertia",
]
