"""Radial-flux PMSM modeled in dq axes for field-oriented control (TOPOLOGY brief §8).

Conventions (project units-discipline applies):
- Amplitude-invariant Park transform; dq are **peak phase** quantities. ``I_rms = I_pk/sqrt(2)``
  with ``I_pk = sqrt(i_d**2 + i_q**2)``.
- Torque ``T = (3/2) p [psi_m i_q + (L_d - L_q) i_d i_q]``; surface-PM ``L_d ≈ L_q`` reduces this
  to ``T = (3/2) p psi_m i_q``.
- Steady-state voltage ``v_d = R_s i_d - w_e L_q i_q``, ``v_q = R_s i_q + w_e (L_d i_d + psi_m)``,
  ``w_e = p w_m``; ``V_pk = sqrt(v_d**2 + v_q**2)``.
- Inverter phase-voltage ceiling (space-vector PWM): ``V_pk_max = V_bus / sqrt(3)``.
- MTPA below base speed (``i_d = 0``). Above base speed, field weakening injects ``i_d < 0`` from
  the R_s-neglected voltage ellipse, after which the actual voltage and loss are recomputed with
  R_s. The motoring quadrant is the modeled case.

This model is **exact given its lumped parameters** (:class:`RadialPMParameters`); first-order
sizing assumptions live in :func:`robotactuatormdo.geometry.radial_flux.size_radial_pm`.

First-order limitations: saturation is judged from the (load-independent) PM tooth/yoke flux
densities; magnet-eddy and mechanical losses are zero; the thermal model is single-node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robotactuatormdo.losses.copper import copper_loss_w, phase_resistance_at
from robotactuatormdo.losses.core import steinmetz_core_loss_w
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)
from robotactuatormdo.thermal.lumped_network import ThermalNetwork, solve_steady_state

__all__ = ["RadialPMParameters", "RadialPMSM"]

_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)
_FEAS_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class RadialPMParameters:
    """Lumped electrical/thermal/mass parameters of a radial PMSM. SI; psi_m is phase-peak."""

    pole_pairs: int
    flux_linkage_wb: float
    r_s_ohm_20: float
    l_d_h: float
    l_q_h: float
    max_phase_current_a_rms: float
    rated_bus_voltage_v: float
    # losses / thermal
    iron_mass_kg: float
    core_b_peak_t: float
    k_hyst: float
    steinmetz_alpha: float
    k_eddy: float
    copper_temp_coeff_per_c: float
    r_th_winding_ambient_c_w: float
    r_th_magnet_ambient_c_w: float
    winding_temp_limit_c: float
    magnet_temp_limit_c: float
    # feasibility limits
    b_tooth_t: float
    b_yoke_t: float
    b_sat_t: float
    demag_id_limit_a_pk: float
    # mass / inertia
    total_mass_kg: float
    rotor_inertia_kg_m2: float
    copper_mass_kg: float
    magnet_mass_kg: float
    # optional multi-node thermal network (None => single-node r_th_* model)
    thermal_network: ThermalNetwork | None = None
    # AC copper-loss multiplier from the winding process (1.0 => DC/baseline)
    ac_loss_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.pole_pairs < 1:
            raise ValueError("pole_pairs must be >= 1")
        for name in ("flux_linkage_wb", "r_s_ohm_20", "l_d_h", "l_q_h",
                     "max_phase_current_a_rms", "rated_bus_voltage_v", "b_sat_t"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.thermal_network is not None:
            free = set(self.thermal_network.free_node_names)
            missing = {"winding", "magnet"} - free
            if missing:
                raise ValueError(f"thermal_network must have free nodes {missing}")

    @property
    def max_phase_current_a_pk(self) -> float:
        return self.max_phase_current_a_rms * _SQRT2


class RadialPMSM:
    """A radial PMSM :class:`MotorModel` driven by FOC, built on a :class:`RadialPMParameters`."""

    def __init__(self, params: RadialPMParameters):
        self.params = params

    # ------------------------------------------------------------------ operating point
    def evaluate_operating_point(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float,
    ) -> MotorOperatingResult:
        p = self.params
        w_e = p.pole_pairs * speed_rad_s
        w_e_abs = abs(w_e)
        f_e = w_e_abs / (2.0 * math.pi)
        v_pk_max = bus_voltage_v / _SQRT3

        # MTPA current from torque (surface-PM: i_d = 0 baseline).
        i_q = torque_nm / (1.5 * p.pole_pairs * p.flux_linkage_wb)
        i_d = _field_weakening_id(i_q, w_e_abs, v_pk_max, p.l_d_h, p.l_q_h, p.flux_linkage_wb)

        i_pk = math.hypot(i_d, i_q)
        i_rms = i_pk / _SQRT2

        # Core loss is independent of winding temperature; copper loss couples through R_s(T_w).
        p_core = steinmetz_core_loss_w(
            p.iron_mass_kg, f_e, p.core_b_peak_t, p.k_hyst, p.steinmetz_alpha, p.k_eddy
        )
        net = p.thermal_network
        t_w = ambient_temp_c
        r_s = p.r_s_ohm_20
        for _ in range(20):
            r_s = phase_resistance_at(p.r_s_ohm_20, p.copper_temp_coeff_per_c, t_w)
            p_cu = copper_loss_w(i_rms, r_s) * p.ac_loss_multiplier
            if net is None:
                t_w_new = ambient_temp_c + p.r_th_winding_ambient_c_w * (p_cu + p_core)
            else:
                t_w_new, _ = _network_temps(net, ambient_temp_c, p_cu, p_core)
            if abs(t_w_new - t_w) < 1e-4:
                t_w = t_w_new
                break
            t_w = t_w_new
        p_cu = copper_loss_w(i_rms, r_s) * p.ac_loss_multiplier
        p_loss = p_cu + p_core
        if net is None:
            t_mag = ambient_temp_c + p.r_th_magnet_ambient_c_w * p_loss
        else:
            t_w, t_mag = _network_temps(net, ambient_temp_c, p_cu, p_core)

        # Voltage with final R_s.
        v_d = r_s * i_d - w_e * p.l_q_h * i_q
        v_q = r_s * i_q + w_e * (p.l_d_h * i_d + p.flux_linkage_wb)
        v_pk = math.hypot(v_d, v_q)
        v_rms = v_pk / _SQRT2

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
            phase_voltage_v_rms=v_rms,
            copper_loss_w=p_cu,
            core_loss_w=p_core,
            magnet_eddy_loss_w=0.0,
            mechanical_loss_w=0.0,
            winding_temp_c=t_w,
            magnet_temp_c=t_mag,
            feasibility=flags,
        )

    # ------------------------------------------------------------------ mass
    def mass_properties(self) -> MassProperties:
        p = self.params
        return MassProperties(
            total_mass_kg=p.total_mass_kg,
            rotor_inertia_kg_m2=p.rotor_inertia_kg_m2,
            active_mass_kg=p.copper_mass_kg + p.iron_mass_kg + p.magnet_mass_kg,
            magnet_mass_kg=p.magnet_mass_kg,
            copper_mass_kg=p.copper_mass_kg,
            iron_mass_kg=p.iron_mass_kg,
        )

    # ------------------------------------------------------------------ envelope
    def torque_speed_envelope(self, n_speeds: int = 40) -> TorqueSpeedEnvelope:
        p = self.params
        bus = p.rated_bus_voltage_v
        w_m_max = self._no_load_speed(bus)
        speeds = np.linspace(0.0, w_m_max, n_speeds)
        t_current_limited = 1.5 * p.pole_pairs * p.flux_linkage_wb * p.max_phase_current_a_pk

        peak = np.array(
            [self._max_feasible_torque(w, bus, t_current_limited, thermal=False) for w in speeds]
        )
        cont = np.array(
            [self._max_feasible_torque(w, bus, t_current_limited, thermal=True) for w in speeds]
        )
        return TorqueSpeedEnvelope(
            speed_rad_s=speeds, peak_torque_nm=peak, continuous_torque_nm=cont
        )

    # ------------------------------------------------------------------ efficiency map
    def efficiency_map(self, n_speeds: int = 30, n_torques: int = 30) -> EfficiencyMap:
        p = self.params
        bus = p.rated_bus_voltage_v
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

    # ------------------------------------------------------------------ helpers
    def _no_load_speed(self, bus_voltage_v: float) -> float:
        """First-order top speed: where back-EMF alone reaches the voltage ceiling."""
        p = self.params
        v_pk_max = bus_voltage_v / _SQRT3
        w_e = v_pk_max / p.flux_linkage_wb
        return (w_e / p.pole_pairs) * 1.2  # a little past base speed to show FW roll-off

    def _max_feasible_torque(
        self, speed_rad_s: float, bus_voltage_v: float, t_upper: float, *, thermal: bool
    ) -> float:
        """Bisection for the largest torque that satisfies the modeled constraints at this speed."""

        def feasible(t: float) -> bool:
            res = self.evaluate_operating_point(t, speed_rad_s, bus_voltage_v, 25.0)
            f = res.feasibility
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


def _network_temps(
    net: ThermalNetwork, ambient_c: float, p_cu: float, p_core: float
) -> tuple[float, float]:
    """Winding and magnet temperatures from a thermal network, referenced to ``ambient_c``.

    Copper loss injects at ``winding``; core loss at ``stator`` if present else ``winding``; the
    magnet node carries only magnet-eddy loss (zero in this first-order model). The network is
    linear with a single ambient boundary, so node temperatures shift uniformly with ambient:
    ``T = ambient_c + (T_solved - T_boundary)``.
    """
    free = set(net.free_node_names)
    core_node = "stator" if "stator" in free else "winding"
    power = {"winding": p_cu}
    power[core_node] = power.get(core_node, 0.0) + p_core
    temps = solve_steady_state(net, power)
    t_boundary = temps[net.boundary_node_names[0]]
    t_w = ambient_c + (temps["winding"] - t_boundary)
    t_mag = ambient_c + (temps["magnet"] - t_boundary)
    return t_w, t_mag


def _field_weakening_id(
    i_q: float, w_e_abs: float, v_pk_max: float, l_d: float, l_q: float, psi_m: float
) -> float:
    """Field-weakening d-axis current (<= 0) from the R_s-neglected voltage ellipse.

    Returns 0 below base speed (no weakening needed). If even ``i_d`` cannot bring the voltage
    under the ceiling, returns 0 (best effort); infeasibility is then flagged via ``voltage_ok``.
    """
    if w_e_abs < 1e-9:
        return 0.0
    budget = (v_pk_max / w_e_abs) ** 2 - (l_q * i_q) ** 2
    if budget <= 0.0:
        return 0.0
    needed_flux = math.sqrt(budget)
    if needed_flux >= psi_m:
        return 0.0
    return (needed_flux - psi_m) / l_d
