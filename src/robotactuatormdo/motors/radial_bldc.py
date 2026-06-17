"""Radial-flux PM machine driven with trapezoidal back-EMF / six-step (BLDC) commutation."""

from __future__ import annotations

from robotactuatormdo.results import (
    EfficiencyMap,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

_PLANNED = "planned: radial BLDC electrical model - see TOPOLOGY-TRADE-STUDY brief section 7"


class RadialBLDC:
    """Radial-flux PM machine driven with trapezoidal back-EMF / six-step (BLDC) commutation.

    Implements :class:`robotactuatormdo.motors.protocols.MotorModel`. Methods are stubbed
    pending implementation.
    """

    def evaluate_operating_point(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float,
    ) -> MotorOperatingResult:
        raise NotImplementedError(_PLANNED)

    def mass_properties(self) -> MassProperties:
        raise NotImplementedError(_PLANNED)

    def torque_speed_envelope(self) -> TorqueSpeedEnvelope:
        raise NotImplementedError(_PLANNED)

    def efficiency_map(self) -> EfficiencyMap:
        raise NotImplementedError(_PLANNED)
