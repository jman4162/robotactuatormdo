"""robotactuatormdo — topology-neutral robot actuator architecture trade-study and MDO.

See the package README and design briefs for the motivating principles. The public surface
re-exported here is the *stable contract*: the motor protocol, shared result types, and the
joint requirement schema. Physics backends live in the submodules and are filled in over time.
"""

from __future__ import annotations

from robotactuatormdo.actuators.actuator import Actuator, ActuatorProperties
from robotactuatormdo.actuators.gearbox import Gearbox
from robotactuatormdo.actuators.qdd import quasi_direct_drive
from robotactuatormdo.evaluation import evaluate_over_duty_cycle
from robotactuatormdo.geometry.radial_flux import RadialPMGeometry, size_radial_pm
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.motors.radial_pmsm import RadialPMParameters, RadialPMSM
from robotactuatormdo.requirements.duty_cycle import DutyCycle
from robotactuatormdo.requirements.joint import JointRequirement
from robotactuatormdo.requirements.robot_task import RobotTask
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MissionResult,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MotorModel",
    "JointRequirement",
    "DutyCycle",
    "RobotTask",
    "MotorOperatingResult",
    "MassProperties",
    "TorqueSpeedEnvelope",
    "EfficiencyMap",
    "FeasibilityFlags",
    "MissionResult",
    "evaluate_over_duty_cycle",
    "RadialPMSM",
    "RadialPMParameters",
    "RadialPMGeometry",
    "size_radial_pm",
    "Gearbox",
    "Actuator",
    "ActuatorProperties",
    "quasi_direct_drive",
]
