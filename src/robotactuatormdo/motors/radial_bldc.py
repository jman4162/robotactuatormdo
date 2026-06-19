"""Radial PM machine driven six-step (BLDC) — the same machine as the PMSM, different drive.

"BLDC vs PMSM" is a *drive* difference, not a topology: identical magnetics, masses, thermal, and
limits. This backend composes a :class:`RadialPMSM` over the same :class:`RadialPMParameters` and
applies first-order six-step effects: torque-per-amp is lower than ideal sinusoidal FOC (more
current, hence ~1/kt^2 more copper loss for the same torque), and there is no smooth MTPA field
weakening (``i_d = 0`` by default), so the high-speed corner rolls off earlier. Mean torque is
honored; commutation torque *ripple* is out of scope for a steady-state point (no ripple field is
invented). With ``kt_ratio_bldc_to_foc=1`` and ``allow_field_weakening=True`` it reproduces the
PMSM exactly (regression guard).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robotactuatormdo.losses.copper import copper_loss_w, phase_resistance_at
from robotactuatormdo.losses.core import steinmetz_core_loss_w
from robotactuatormdo.motors.radial_pmsm import (
    RadialPMParameters,
    RadialPMSM,
    _field_weakening_id,
    _network_temps,
)
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

__all__ = ["RadialBLDCParameters", "RadialBLDC"]

_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)
_FEAS_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class RadialBLDCParameters:
    """Six-step drive parameters over a radial PM machine."""

    pmsm: RadialPMParameters
    kt_ratio_bldc_to_foc: float = 0.95
    allow_field_weakening: bool = False
    commutation_loss_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.kt_ratio_bldc_to_foc <= 1.0:
            raise ValueError("kt_ratio_bldc_to_foc must be in (0, 1]")


class RadialBLDC:
    """A six-step (BLDC) :class:`MotorModel` over a radial PM machine."""

    def __init__(self, params: RadialBLDCParameters):
        self.params = params
        self._pmsm = RadialPMSM(params.pmsm)

    def evaluate_operating_point(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float,
    ) -> MotorOperatingResult:
        p = self.params.pmsm
        kt = self.params.kt_ratio_bldc_to_foc
        w_e = p.pole_pairs * speed_rad_s
        w_e_abs = abs(w_e)
        f_e = w_e_abs / (2.0 * math.pi)
        v_pk_max = bus_voltage_v / _SQRT3

        # Six-step needs more current per torque than ideal FOC (the kt derate).
        i_q = torque_nm / (1.5 * p.pole_pairs * p.flux_linkage_wb) / kt
        if self.params.allow_field_weakening:
            i_d = _field_weakening_id(i_q, w_e_abs, v_pk_max, p.l_d_h, p.l_q_h, p.flux_linkage_wb)
        else:
            i_d = 0.0
        i_pk = math.hypot(i_d, i_q)
        i_rms = i_pk / _SQRT2

        p_core = steinmetz_core_loss_w(
            p.iron_mass_kg, f_e, p.core_b_peak_t, p.k_hyst, p.steinmetz_alpha, p.k_eddy
        )
        q = p_core * (1.0 + self.params.commutation_loss_fraction)  # lump commutation into core
        net = p.thermal_network
        t_w = ambient_temp_c
        r_s = p.r_s_ohm_20
        for _ in range(20):
            r_s = phase_resistance_at(p.r_s_ohm_20, p.copper_temp_coeff_per_c, t_w)
            p_cu = copper_loss_w(i_rms, r_s) * p.ac_loss_multiplier
            if net is None:
                t_w_new = ambient_temp_c + p.r_th_winding_ambient_c_w * (p_cu + q)
            else:
                t_w_new, _ = _network_temps(net, ambient_temp_c, p_cu, q)
            if abs(t_w_new - t_w) < 1e-4:
                t_w = t_w_new
                break
            t_w = t_w_new
        p_cu = copper_loss_w(i_rms, r_s) * p.ac_loss_multiplier
        if net is None:
            t_mag = ambient_temp_c + p.r_th_magnet_ambient_c_w * (p_cu + q)
        else:
            t_w, t_mag = _network_temps(net, ambient_temp_c, p_cu, q)

        v_d = r_s * i_d - w_e * p.l_q_h * i_q
        v_q = r_s * i_q + w_e * (p.l_d_h * i_d + p.flux_linkage_wb)
        v_pk = math.hypot(v_d, v_q)

        flags = FeasibilityFlags(
            voltage_ok=v_pk <= v_pk_max * (1.0 + _FEAS_TOL),
            current_ok=i_pk <= p.max_phase_current_a_pk * (1.0 + _FEAS_TOL),
            winding_temp_ok=t_w <= p.winding_temp_limit_c,
            magnet_temp_ok=t_mag <= p.magnet_temp_limit_c,
            saturation_ok=max(p.b_tooth_t, p.b_yoke_t) <= p.b_sat_t,
            demag_ok=abs(i_d) <= p.demag_id_limit_a_pk * (1.0 + _FEAS_TOL),
        )
        return MotorOperatingResult(
            torque_nm=torque_nm,
            speed_rad_s=speed_rad_s,
            phase_current_a_rms=i_rms,
            phase_voltage_v_rms=v_pk / _SQRT2,
            copper_loss_w=p_cu,
            core_loss_w=q,
            magnet_eddy_loss_w=0.0,
            mechanical_loss_w=0.0,
            winding_temp_c=t_w,
            magnet_temp_c=t_mag,
            feasibility=flags,
        )

    def mass_properties(self) -> MassProperties:
        return self._pmsm.mass_properties()

    def torque_speed_envelope(self, n_speeds: int = 40) -> TorqueSpeedEnvelope:
        p = self.params.pmsm
        bus = p.rated_bus_voltage_v
        w_m_max = self._pmsm._no_load_speed(bus)
        speeds = np.linspace(0.0, w_m_max, n_speeds)
        t_cl = (
            1.5 * p.pole_pairs * p.flux_linkage_wb * p.max_phase_current_a_pk
            * self.params.kt_ratio_bldc_to_foc
        )
        peak = np.array([self._max_torque(w, bus, t_cl, thermal=False) for w in speeds])
        cont = np.array([self._max_torque(w, bus, t_cl, thermal=True) for w in speeds])
        return TorqueSpeedEnvelope(
            speed_rad_s=speeds, peak_torque_nm=peak, continuous_torque_nm=cont
        )

    def _max_torque(self, speed: float, bus: float, t_upper: float, *, thermal: bool) -> float:
        def feasible(t: float) -> bool:
            f = self.evaluate_operating_point(t, speed, bus, 25.0).feasibility
            ok = f.voltage_ok and f.current_ok and f.saturation_ok and f.demag_ok
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

    def efficiency_map(self, n_speeds: int = 30, n_torques: int = 30) -> EfficiencyMap:
        bus = self.params.pmsm.rated_bus_voltage_v
        env = self.torque_speed_envelope(n_speeds=n_speeds)
        speeds = env.speed_rad_s
        t_max = float(np.max(env.peak_torque_nm)) or 1.0
        torques = np.linspace(t_max / n_torques, t_max, n_torques)
        eff = np.full((n_torques, n_speeds), np.nan)
        for j, w in enumerate(speeds):
            for i, t in enumerate(torques):
                res = self.evaluate_operating_point(t, w, bus, 25.0)
                if res.feasibility.feasible and res.mechanical_power_w > 0.0:
                    eff[i, j] = res.efficiency
        return EfficiencyMap(speed_rad_s=speeds, torque_nm=torques, efficiency=eff)
