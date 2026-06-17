"""Motor model populated from a commercial datasheet, used as a credible baseline."""

from __future__ import annotations

from robotactuatormdo.results import (
    EfficiencyMap,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

_PLANNED = "planned: commercial datasheet adapter - see BACKGROUND brief"


class CommercialMotor:
    """Motor model populated from a commercial datasheet, used as a credible baseline.

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
