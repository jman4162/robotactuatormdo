"""Axial-flux backend: adapts an ``axfluxmdo`` design to the :class:`MotorModel` protocol.

``axfluxmdo`` is an *optional* dependency. Install it with ``pip install
'robotactuatormdo[axial]'``. It is imported lazily so the radial/system layers stay usable
without it.
"""

from __future__ import annotations

from typing import Any

from robotactuatormdo.results import (
    EfficiencyMap,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

_PLANNED = "planned: axfluxmdo adapter - see BACKGROUND brief"


def _import_axfluxmdo() -> Any:
    try:
        import axfluxmdo
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The axial-flux backend requires 'axfluxmdo'. Install it with: "
            "pip install 'robotactuatormdo[axial]'"
        ) from exc
    return axfluxmdo


class AxFluxMDOAdapter:
    """Wrap an ``axfluxmdo`` axial-flux design as a :class:`MotorModel`."""

    def __init__(self, design: Any) -> None:
        self._axfluxmdo = _import_axfluxmdo()
        self._design = design

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
