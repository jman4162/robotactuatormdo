"""Robot joint / duty-cycle / task requirement schemas."""

from __future__ import annotations

from robotactuatormdo.requirements.duty_cycle import DutyCycle
from robotactuatormdo.requirements.joint import JointRequirement
from robotactuatormdo.requirements.robot_task import RobotTask

__all__ = ["JointRequirement", "DutyCycle", "RobotTask"]
