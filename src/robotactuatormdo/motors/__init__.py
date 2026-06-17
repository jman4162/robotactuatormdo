"""Motor backends. All implement the :class:`MotorModel` protocol."""

from __future__ import annotations

from robotactuatormdo.motors.axial_adapter_axfluxmdo import AxFluxMDOAdapter
from robotactuatormdo.motors.commercial_catalog import CommercialMotor
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.motors.radial_bldc import RadialBLDC
from robotactuatormdo.motors.radial_pmsm import RadialPMSM

__all__ = [
    "MotorModel",
    "RadialBLDC",
    "RadialPMSM",
    "AxFluxMDOAdapter",
    "CommercialMotor",
]
