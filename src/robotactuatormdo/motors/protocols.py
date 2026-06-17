"""The motor backend contract.

Every motor model — radial BLDC, radial PMSM, an axial-flux adapter over ``axfluxmdo``, or a
commercial-catalog lookup — implements :class:`MotorModel` so the trade-study layer can treat
them interchangeably. This is what keeps the framework topology-neutral: the studies never know
whether they hold a radial or an axial machine.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from robotactuatormdo.results import (
    EfficiencyMap,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

__all__ = ["MotorModel"]


@runtime_checkable
class MotorModel(Protocol):
    """Common interface for a motor model.

    Implementations must accept and return SI phase quantities (see ``results`` for the
    units-discipline rule).
    """

    def evaluate_operating_point(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float,
    ) -> MotorOperatingResult:
        """Evaluate losses, currents, voltages, temperatures and feasibility at one point."""
        ...

    def mass_properties(self) -> MassProperties:
        """Return the motor's mass and rotary-inertia breakdown."""
        ...

    def torque_speed_envelope(self) -> TorqueSpeedEnvelope:
        """Return the peak and continuous torque-speed boundary."""
        ...

    def efficiency_map(self) -> EfficiencyMap:
        """Return efficiency over a torque-speed grid."""
        ...
