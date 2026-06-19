"""Power stage: battery + cable + inverter feeding an inner motor/actuator.

``PowerStage`` implements :class:`MotorModel`, so it nests as
``PowerStage(Actuator(motor, gearbox))`` for the full battery->joint stack and drops into
:func:`evaluate_over_duty_cycle` unchanged. The
``bus_voltage_v`` argument is the battery **pack open-circuit voltage**; the stage solves the sagged
DC-link voltage under load:

    P_dc   = P_motor_in + P_inverter,   P_motor_in = inner.mechanical_power_w + inner.total_loss_w
    I_dc   = P_dc / V_link
    V_link = V_oc - I_dc * (R_internal + R_cable)

via a small fixed point, then feeds the sagged ``V_link`` to the inner model (whose voltage check
already uses ``V_link/sqrt(3)``). Inverter loss enters ``inverter_loss_w``; battery/cable/inverter
current limits and link collapse set ``source_ok``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robotactuatormdo.electronics.battery import Battery
from robotactuatormdo.electronics.cable import Cable
from robotactuatormdo.electronics.inverter import Inverter
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

__all__ = ["PowerStage", "PowerStageResult"]

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class PowerStageResult:
    """Source-side detail at one operating point (SI)."""

    dc_link_voltage_v: float
    dc_current_a: float
    inverter_loss_w: float


def _rated_pack_voltage(inner: MotorModel) -> float:
    v = getattr(inner, "rated_bus_voltage_v", None)
    if v is None:
        v = getattr(getattr(inner, "params", None), "rated_bus_voltage_v", None)
    return float(v) if v else 48.0


class PowerStage:
    """Battery + cable + inverter wrapping an inner :class:`MotorModel`."""

    def __init__(
        self,
        inner: MotorModel,
        inverter: Inverter,
        battery: Battery,
        cable: Cable | None = None,
        rated_pack_voltage_v: float | None = None,
    ):
        self.inner = inner
        self.inverter = inverter
        self.battery = battery
        self.cable = cable
        self.rated_pack_voltage_v = rated_pack_voltage_v or _rated_pack_voltage(inner)

    @property
    def _source_resistance(self) -> float:
        r = self.battery.internal_resistance_ohm
        if self.cable is not None:
            r += self.cable.resistance_ohm
        return r

    # ------------------------------------------------------------------ link solver
    def _solve_link(
        self, torque_nm: float, speed_rad_s: float, v_oc: float, ambient_temp_c: float
    ) -> tuple[MotorOperatingResult, float, float, float]:
        """Fixed-point solve for (inner result, V_link, I_dc, inverter loss)."""
        r_src = self._source_resistance
        v_link = v_oc
        res = self.inner.evaluate_operating_point(torque_nm, speed_rad_s, v_link, ambient_temp_c)
        i_dc = 0.0
        p_inv = 0.0
        for _ in range(40):
            res = self.inner.evaluate_operating_point(
                torque_nm, speed_rad_s, max(v_link, 1e-6), ambient_temp_c
            )
            i_rms = res.phase_current_a_rms
            p_inv = self.inverter.loss_w(i_rms, max(v_link, 0.0))
            p_dc = res.mechanical_power_w + res.total_loss_w + p_inv
            i_dc = p_dc / max(v_link, 1e-6)
            v_new = v_oc - i_dc * r_src
            if v_new <= 0.0 or not math.isfinite(v_new):
                v_link = 0.0
                break
            if abs(v_new - v_link) < 1e-7 * max(v_oc, 1.0):
                v_link = v_new
                break
            v_link = v_new
        return res, v_link, i_dc, p_inv

    # ------------------------------------------------------------------ operating point
    def evaluate_operating_point(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float,
    ) -> MotorOperatingResult:
        res, v_link, i_dc, p_inv = self._solve_link(
            torque_nm, speed_rad_s, bus_voltage_v, ambient_temp_c
        )
        i_rms = res.phase_current_a_rms
        cable_ok = self.cable is None or abs(i_dc) <= self.cable.max_current_a
        source_ok = (
            res.feasibility.source_ok
            and abs(i_dc) <= self.battery.max_current_a
            and i_rms <= self.inverter.max_phase_current_a_rms
            and cable_ok
            and v_link > 0.0
        )
        mf = res.feasibility
        flags = FeasibilityFlags(
            voltage_ok=mf.voltage_ok,
            current_ok=mf.current_ok,
            winding_temp_ok=mf.winding_temp_ok,
            magnet_temp_ok=mf.magnet_temp_ok,
            saturation_ok=mf.saturation_ok,
            demag_ok=mf.demag_ok,
            mechanical_ok=mf.mechanical_ok,
            source_ok=source_ok,
        )
        return MotorOperatingResult(
            torque_nm=torque_nm,
            speed_rad_s=speed_rad_s,
            phase_current_a_rms=i_rms,
            phase_voltage_v_rms=res.phase_voltage_v_rms,
            copper_loss_w=res.copper_loss_w,
            core_loss_w=res.core_loss_w,
            magnet_eddy_loss_w=res.magnet_eddy_loss_w,
            mechanical_loss_w=res.mechanical_loss_w,
            winding_temp_c=res.winding_temp_c,
            magnet_temp_c=res.magnet_temp_c,
            inverter_loss_w=p_inv,
            feasibility=flags,
        )

    def link_state(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float = 25.0,
    ) -> PowerStageResult:
        """DC-link voltage, DC current, and inverter loss at one operating point."""
        _, v_link, i_dc, p_inv = self._solve_link(
            torque_nm, speed_rad_s, bus_voltage_v, ambient_temp_c
        )
        return PowerStageResult(dc_link_voltage_v=v_link, dc_current_a=i_dc, inverter_loss_w=p_inv)

    # ------------------------------------------------------------------ mass
    def mass_properties(self) -> MassProperties:
        m = self.inner.mass_properties()
        extra = self.inverter.mass_kg + self.battery.mass_kg
        if self.cable is not None:
            extra += self.cable.mass_kg
        return MassProperties(
            total_mass_kg=m.total_mass_kg + extra,
            rotor_inertia_kg_m2=m.rotor_inertia_kg_m2,
            active_mass_kg=m.active_mass_kg,
            magnet_mass_kg=m.magnet_mass_kg,
            copper_mass_kg=m.copper_mass_kg,
            iron_mass_kg=m.iron_mass_kg,
        )

    # ------------------------------------------------------------------ envelope
    def torque_speed_envelope(self, n_speeds: int = 30) -> TorqueSpeedEnvelope:
        inner_env = self.inner.torque_speed_envelope()
        speeds = inner_env.speed_rad_s
        t_upper = float(np.max(inner_env.peak_torque_nm)) or 1.0
        bus = self.rated_pack_voltage_v
        peak = np.array([self._max_torque(w, bus, t_upper, thermal=False) for w in speeds])
        cont = np.array([self._max_torque(w, bus, t_upper, thermal=True) for w in speeds])
        return TorqueSpeedEnvelope(
            speed_rad_s=speeds, peak_torque_nm=peak, continuous_torque_nm=cont
        )

    def _max_torque(self, speed: float, bus: float, t_upper: float, *, thermal: bool) -> float:
        def feasible(t: float) -> bool:
            f = self.evaluate_operating_point(t, speed, bus, 25.0).feasibility
            ok = f.voltage_ok and f.current_ok and f.saturation_ok and f.demag_ok
            ok = ok and f.mechanical_ok and f.source_ok
            if thermal:
                ok = ok and f.winding_temp_ok and f.magnet_temp_ok
            return ok

        if not feasible(0.0):
            return 0.0
        lo, hi = 0.0, t_upper
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                lo = mid
            else:
                hi = mid
        return lo

    # ------------------------------------------------------------------ efficiency map
    def efficiency_map(self, n_speeds: int = 20, n_torques: int = 20) -> EfficiencyMap:
        env = self.torque_speed_envelope(n_speeds=n_speeds)
        speeds = env.speed_rad_s
        t_max = float(np.max(env.peak_torque_nm)) or 1.0
        torques = np.linspace(t_max / n_torques, t_max, n_torques)
        bus = self.rated_pack_voltage_v
        eff = np.full((n_torques, n_speeds), np.nan)
        for j, w in enumerate(speeds):
            for i, t in enumerate(torques):
                res = self.evaluate_operating_point(t, w, bus, 25.0)
                if res.feasibility.feasible and res.mechanical_power_w > 0.0:
                    eff[i, j] = res.efficiency
        return EfficiencyMap(speed_rad_s=speeds, torque_nm=torques, efficiency=eff)
