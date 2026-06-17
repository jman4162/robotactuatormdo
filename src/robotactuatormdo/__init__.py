"""robotactuatormdo — topology-neutral robot actuator architecture trade-study and MDO.

See the package README and design briefs for the motivating principles. The public surface
re-exported here is the *stable contract*: the motor protocol, shared result types, and the
joint requirement schema. Physics backends live in the submodules and are filled in over time.
"""

from __future__ import annotations

from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.requirements.joint import JointRequirement
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MotorModel",
    "JointRequirement",
    "MotorOperatingResult",
    "MassProperties",
    "TorqueSpeedEnvelope",
    "EfficiencyMap",
    "FeasibilityFlags",
]
