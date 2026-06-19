"""Commercial-catalog motor: a MotorModel built from datasheet scalars (no geometry).

A credible baseline to beat. All quantities are SI *phase* values (see the units-discipline note
in ``results.py``). With no geometry there is nothing to judge saturation/demag against, so those
flags stay ``True`` (documented). Core/windage loss is a lumped speed-dependent term, not a
physical Steinmetz model. ``ke_v_s_per_rad`` is a phase peak back-EMF constant on **mechanical**
speed; set it 0 to skip the voltage-headroom check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robotactuatormdo.losses.copper import copper_loss_w, phase_resistance_at
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

__all__ = ["CommercialMotorSpec", "CommercialMotor"]

_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True, slots=True)
class CommercialMotorSpec:
    """Datasheet scalars for a commercial motor. SI, phase quantities."""

    kt_nm_per_a_rms: float
    r_phase_ohm: float
    pole_pairs: int
    max_phase_current_a_rms: float
    rated_torque_nm: float
    peak_torque_nm: float
    max_speed_rad_s: float
    rated_bus_voltage_v: float
    ke_v_s_per_rad: float = 0.0
    l_phase_h: float = 0.0
    core_loss_coeff_w_per_rad: float = 0.0
    windage_loss_coeff_w_per_rad2: float = 0.0
    r_th_winding_ambient_c_w: float = 0.0
    winding_temp_limit_c: float = 155.0
    copper_temp_coeff_per_c: float = 0.00393
    total_mass_kg: float = 0.0
    rotor_inertia_kg_m2: float = 0.0

    def __post_init__(self) -> None:
        for name in ("kt_nm_per_a_rms", "max_phase_current_a_rms", "peak_torque_nm",
                     "max_speed_rad_s", "rated_bus_voltage_v"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


class CommercialMotor:
    """A datasheet-scalar :class:`MotorModel`."""

    def __init__(self, spec: CommercialMotorSpec):
        self.spec = spec
        self.params = spec  # for Actuator/PowerStage composition (rated_bus_voltage_v)

    def evaluate_operating_point(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float,
    ) -> MotorOperatingResult:
        s = self.spec
        i_rms = abs(torque_nm) / s.kt_nm_per_a_rms
        i_pk = i_rms * _SQRT2
        w_e = s.pole_pairs * speed_rad_s
        p_core = s.core_loss_coeff_w_per_rad * abs(speed_rad_s)
        p_mech = s.windage_loss_coeff_w_per_rad2 * speed_rad_s**2

        t_w = ambient_temp_c
        r_s = phase_resistance_at(s.r_phase_ohm, s.copper_temp_coeff_per_c, t_w)
        if s.r_th_winding_ambient_c_w > 0.0:
            for _ in range(20):
                r_s = phase_resistance_at(s.r_phase_ohm, s.copper_temp_coeff_per_c, t_w)
                p_cu = copper_loss_w(i_rms, r_s)
                t_w_new = ambient_temp_c + s.r_th_winding_ambient_c_w * (p_cu + p_core + p_mech)
                if abs(t_w_new - t_w) < 1e-4:
                    t_w = t_w_new
                    break
                t_w = t_w_new
        p_cu = copper_loss_w(i_rms, r_s)

        v_pk_max = bus_voltage_v / _SQRT3
        if s.ke_v_s_per_rad > 0.0:
            e_pk = s.ke_v_s_per_rad * abs(speed_rad_s)
            v_pk = math.hypot(r_s * i_pk, e_pk + w_e * s.l_phase_h * i_pk)
            voltage_ok = v_pk <= v_pk_max * (1.0 + 1e-6)
        else:
            v_pk = r_s * i_pk
            voltage_ok = True
        v_rms = v_pk / _SQRT2

        flags = FeasibilityFlags(
            voltage_ok=voltage_ok,
            current_ok=(i_rms <= s.max_phase_current_a_rms * (1.0 + 1e-6))
            and (abs(torque_nm) <= s.peak_torque_nm * (1.0 + 1e-6)),
            winding_temp_ok=t_w <= s.winding_temp_limit_c,
            mechanical_ok=abs(speed_rad_s) <= s.max_speed_rad_s * (1.0 + 1e-6),
        )
        return MotorOperatingResult(
            torque_nm=torque_nm,
            speed_rad_s=speed_rad_s,
            phase_current_a_rms=i_rms,
            phase_voltage_v_rms=v_rms,
            copper_loss_w=p_cu,
            core_loss_w=p_core,
            magnet_eddy_loss_w=0.0,
            mechanical_loss_w=p_mech,
            winding_temp_c=t_w,
            magnet_temp_c=ambient_temp_c,
            feasibility=flags,
        )

    def mass_properties(self) -> MassProperties:
        return MassProperties(
            total_mass_kg=self.spec.total_mass_kg,
            rotor_inertia_kg_m2=self.spec.rotor_inertia_kg_m2,
        )

    def torque_speed_envelope(self, n_speeds: int = 40) -> TorqueSpeedEnvelope:
        s = self.spec
        speeds = np.linspace(0.0, s.max_speed_rad_s, n_speeds)
        if s.ke_v_s_per_rad > 0.0:
            w_base = (s.rated_bus_voltage_v / _SQRT3) / s.ke_v_s_per_rad
        else:
            w_base = float("inf")
        factor = np.where(speeds > w_base, w_base / np.maximum(speeds, 1e-9), 1.0)
        return TorqueSpeedEnvelope(
            speed_rad_s=speeds,
            peak_torque_nm=s.peak_torque_nm * factor,
            continuous_torque_nm=s.rated_torque_nm * factor,
        )

    def efficiency_map(self, n_speeds: int = 30, n_torques: int = 30) -> EfficiencyMap:
        s = self.spec
        env = self.torque_speed_envelope(n_speeds=n_speeds)
        speeds = env.speed_rad_s
        t_max = float(np.max(env.peak_torque_nm)) or 1.0
        torques = np.linspace(t_max / n_torques, t_max, n_torques)
        eff = np.full((n_torques, n_speeds), np.nan)
        for j, w in enumerate(speeds):
            for i, t in enumerate(torques):
                res = self.evaluate_operating_point(t, w, s.rated_bus_voltage_v, 25.0)
                if res.feasibility.feasible and res.mechanical_power_w > 0.0:
                    eff[i, j] = res.efficiency
        return EfficiencyMap(speed_rad_s=speeds, torque_nm=torques, efficiency=eff)
