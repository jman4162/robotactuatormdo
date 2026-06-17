"""Robot task: a labeled collection of per-joint duty cycles.

For now a task is simply the set of output duty cycles its joints must serve, supplied directly.
Generating joint duty cycles from a trajectory (inverse dynamics) needs a robot kinematic/dynamic
model and is deferred to a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robotactuatormdo.requirements.duty_cycle import DutyCycle

__all__ = ["RobotTask", "joint_duty_from_task"]


@dataclass(frozen=True, slots=True)
class RobotTask:
    """A named task and the duty cycle demanded of each joint.

    Parameters
    ----------
    name:
        Human-readable task label (e.g. ``"trot_2ms"``).
    joint_duty_cycles:
        Mapping of joint name to its :class:`DutyCycle`.
    """

    name: str
    joint_duty_cycles: dict[str, DutyCycle]

    def joint(self, name: str) -> DutyCycle:
        """Return the duty cycle for ``name`` (raises ``KeyError`` if absent)."""
        return self.joint_duty_cycles[name]


def joint_duty_from_task(task: Any) -> dict[str, DutyCycle]:
    """Derive per-joint duty cycles from a robot trajectory via inverse dynamics."""
    raise NotImplementedError(
        "planned: trajectory->duty needs a robot dynamics model (later phase)"
    )
