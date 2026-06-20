"""Axial-flux backend: adapt an ``axfluxmdo`` design to the :class:`MotorModel` protocol.

``axfluxmdo`` is an optional dependency (``pip install 'robotactuatormdo[axial]'``). This adapter is
**duck-typed and defensive**: it pulls a small set of quantities from any design object that exposes
them, so it is fully testable with a fake backend and needs no import to construct. The real
``axfluxmdo`` attribute names are **unverified** — the name guesses live only in the private
``_QUANTITY_NAMES`` table below; verify and adjust them against an installed ``axfluxmdo>=0.7``.

Minimal contract the design object must expose (attribute or zero-arg method), per quantity:
``kt`` (N·m/A_rms), ``ke`` (phase peak V·s/rad, optional), ``r_phase`` (Ohm), ``mass_kg``,
``rotor_inertia_kg_m2``, ``max_phase_current_a_rms``, ``max_speed_rad_s``, ``rated_bus_voltage_v``.
Optional: ``efficiency_at(torque, speed)`` for a loss estimate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from robotactuatormdo.losses.copper import copper_loss_w
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

__all__ = ["AxFluxMDOAdapter"]

_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)
_MISSING = object()

# Real axfluxmdo names are unknown; each entry lists candidate attribute/method names to try.
_QUANTITY_NAMES: dict[str, tuple[str, ...]] = {
    "kt": ("kt_nm_per_a_rms", "kt", "torque_constant"),
    "ke": ("ke_v_s_per_rad", "ke", "back_emf_constant"),
    "r_phase": ("r_phase_ohm", "phase_resistance", "r_phase"),
    "mass_kg": ("mass_kg", "total_mass_kg", "mass"),
    "rotor_inertia_kg_m2": ("rotor_inertia_kg_m2", "inertia", "rotor_inertia"),
    "max_phase_current_a_rms": ("max_phase_current_a_rms", "i_max_rms", "max_current"),
    "max_speed_rad_s": ("max_speed_rad_s", "max_speed", "omega_max"),
    "rated_bus_voltage_v": ("rated_bus_voltage_v", "bus_voltage_v", "v_bus"),
}


def _import_axfluxmdo() -> Any:
    try:
        import axfluxmdo
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The axial-flux backend requires 'axfluxmdo'. Install it with: "
            "pip install 'robotactuatormdo[axial]'"
        ) from exc
    return axfluxmdo


def _q(design: Any, quantity: str, *, default: Any = _MISSING) -> Any:
    """Fetch a quantity from a duck-typed design, trying candidate names; call zero-arg methods."""
    for name in _QUANTITY_NAMES[quantity]:
        if hasattr(design, name):
            val = getattr(design, name)
            return val() if callable(val) else val
    if default is _MISSING:
        raise AttributeError(
            f"axial design is missing {quantity!r}; tried {_QUANTITY_NAMES[quantity]}"
        )
    return default


class _AxialShim:
    """Scalar duck-typed design derived from an axfluxmdo evaluation (or a fake)."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class AxFluxMDOAdapter:
    """Wrap an axial-flux design object as a :class:`MotorModel` (duck-typed).

    Pass an existing design object directly. Set ``require_axfluxmdo=True`` to assert the optional
    package is installed (production use); the default ``False`` keeps the adapter testable with a
    fake design and no dependency.
    """

    @classmethod
    def from_axfluxmdo(
        cls,
        design: Any,
        model: Any,
        operating_point: Any,
        *,
        max_phase_current_a_rms: float,
        max_speed_rad_s: float,
        rated_bus_voltage_v: float,
    ) -> AxFluxMDOAdapter:
        """Build an adapter from a real axfluxmdo design by evaluating it at a rated point.

        ``model.evaluate(design, operating_point)`` must return a result with ``torque_nm``,
        ``back_emf_v_rms``, ``mass_kg`` (and optionally ``phase_resistance_ohm``,
        ``mass_breakdown``). Kept fully duck-typed so a fake model/design/op exercises it without
        the optional ``axfluxmdo`` dependency. Rotor inertia (which axfluxmdo does not expose) is a
        first-order disk estimate from rotor mass and radii.
        """
        res = model.evaluate(design, operating_point)
        current = float(operating_point.current_rms)
        speed_rpm = float(operating_point.speed_rpm)
        omega_m = speed_rpm * 2.0 * math.pi / 60.0

        kt = res.torque_nm / current
        ke = res.back_emf_v_rms * _SQRT2 / omega_m if omega_m > 0.0 else 0.0
        mass = float(res.mass_kg)
        breakdown = getattr(res, "mass_breakdown", None) or {}
        rotor_mass = breakdown.get("magnets", 0.0) + breakdown.get("back_iron", 0.0) or 0.5 * mass
        r_o = float(getattr(design, "outer_radius", 0.0))
        r_i = float(getattr(design, "inner_radius", 0.0))
        inertia = 0.5 * rotor_mass * (r_o**2 + r_i**2)

        op_type = type(operating_point)

        def efficiency_at(torque_nm: float, speed_rad_s: float) -> float:
            op2 = op_type(
                speed_rpm=abs(speed_rad_s) * 60.0 / (2.0 * math.pi),
                current_rms=abs(torque_nm) / kt,
                dc_bus_voltage=rated_bus_voltage_v,
            )
            return float(model.evaluate(design, op2).efficiency)

        shim = _AxialShim(
            kt=kt,
            ke=ke,
            r_phase=float(getattr(res, "phase_resistance_ohm", 0.0)),
            mass_kg=mass,
            rotor_inertia_kg_m2=inertia,
            max_phase_current_a_rms=max_phase_current_a_rms,
            max_speed_rad_s=max_speed_rad_s,
            rated_bus_voltage_v=rated_bus_voltage_v,
            efficiency_at=efficiency_at,
        )
        return cls(shim)

    def __init__(self, design: Any, *, require_axfluxmdo: bool = False):
        if require_axfluxmdo:
            _import_axfluxmdo()
        self._design = design
        self.kt = float(_q(design, "kt"))
        self.ke = float(_q(design, "ke", default=0.0))
        self.r_phase = float(_q(design, "r_phase", default=0.0))
        self._mass = float(_q(design, "mass_kg", default=0.0))
        self._inertia = float(_q(design, "rotor_inertia_kg_m2", default=0.0))
        self.max_phase_current_a_rms = float(_q(design, "max_phase_current_a_rms"))
        self.max_speed_rad_s = float(_q(design, "max_speed_rad_s"))
        self.rated_bus_voltage_v = float(_q(design, "rated_bus_voltage_v"))

    def evaluate_operating_point(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float,
    ) -> MotorOperatingResult:
        i_rms = abs(torque_nm) / self.kt
        i_pk = i_rms * _SQRT2
        p_cu = copper_loss_w(i_rms, self.r_phase)
        # Optional efficiency-based loss top-up if the design provides it.
        p_other = 0.0
        eff_fn = getattr(self._design, "efficiency_at", None)
        if callable(eff_fn):
            eff = float(eff_fn(torque_nm, speed_rad_s))
            p_mech = torque_nm * speed_rad_s
            if 0.0 < eff < 1.0 and p_mech > 0.0:
                p_other = max(p_mech * (1.0 / eff - 1.0) - p_cu, 0.0)

        v_pk_max = bus_voltage_v / _SQRT3
        voltage_ok = True
        if self.ke > 0.0:
            v_pk = math.hypot(self.r_phase * i_pk, self.ke * abs(speed_rad_s))
            voltage_ok = v_pk <= v_pk_max * (1.0 + 1e-6)
        flags = FeasibilityFlags(
            voltage_ok=voltage_ok,
            current_ok=i_rms <= self.max_phase_current_a_rms * (1.0 + 1e-6),
            mechanical_ok=abs(speed_rad_s) <= self.max_speed_rad_s * (1.0 + 1e-6),
        )
        return MotorOperatingResult(
            torque_nm=torque_nm,
            speed_rad_s=speed_rad_s,
            phase_current_a_rms=i_rms,
            phase_voltage_v_rms=0.0,
            copper_loss_w=p_cu,
            core_loss_w=p_other,
            magnet_eddy_loss_w=0.0,
            mechanical_loss_w=0.0,
            winding_temp_c=ambient_temp_c,
            magnet_temp_c=ambient_temp_c,
            feasibility=flags,
        )

    def mass_properties(self) -> MassProperties:
        return MassProperties(total_mass_kg=self._mass, rotor_inertia_kg_m2=self._inertia)

    def torque_speed_envelope(self, n_speeds: int = 40) -> TorqueSpeedEnvelope:
        speeds = np.linspace(0.0, self.max_speed_rad_s, n_speeds)
        t_max = self.kt * self.max_phase_current_a_rms
        if self.ke > 0.0:
            w_base = (self.rated_bus_voltage_v / _SQRT3) / self.ke
            factor = np.where(speeds > w_base, w_base / np.maximum(speeds, 1e-9), 1.0)
        else:
            factor = np.ones_like(speeds)
        torque = t_max * factor
        return TorqueSpeedEnvelope(
            speed_rad_s=speeds, peak_torque_nm=torque, continuous_torque_nm=torque
        )

    def efficiency_map(self, n_speeds: int = 30, n_torques: int = 30) -> EfficiencyMap:
        env = self.torque_speed_envelope(n_speeds=n_speeds)
        speeds = env.speed_rad_s
        t_max = float(np.max(env.peak_torque_nm)) or 1.0
        torques = np.linspace(t_max / n_torques, t_max, n_torques)
        eff = np.full((n_torques, n_speeds), np.nan)
        for j, w in enumerate(speeds):
            for i, t in enumerate(torques):
                res = self.evaluate_operating_point(t, w, self.rated_bus_voltage_v, 25.0)
                if res.feasibility.feasible and res.mechanical_power_w > 0.0:
                    eff[i, j] = res.efficiency
        return EfficiencyMap(speed_rad_s=speeds, torque_nm=torques, efficiency=eff)
