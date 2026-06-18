"""Actuator = motor + transmission, presented at the output (joint) shaft.

``Actuator`` itself implements :class:`MotorModel`, but all of its inputs and outputs are at the
**output shaft**: it maps an output operating point to the motor shaft through the gearbox, calls
the inner motor, and folds gear loss back into the result. Because it is a ``MotorModel``, it drops
straight into :func:`robotactuatormdo.evaluate_over_duty_cycle` and any study with no changes
(call the reducer with ``gear_ratio=1.0`` — the actuator already owns the gearing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotactuatormdo.actuators.gearbox import Gearbox
from robotactuatormdo.motors.protocols import MotorModel
from robotactuatormdo.results import (
    EfficiencyMap,
    FeasibilityFlags,
    MassProperties,
    MotorOperatingResult,
    TorqueSpeedEnvelope,
)

__all__ = ["Actuator", "ActuatorProperties"]


@dataclass(frozen=True, slots=True)
class ActuatorProperties:
    """Static actuator-level figures of merit (SI; torque density in N·m/kg)."""

    gear_ratio: float
    total_mass_kg: float
    reflected_inertia_kg_m2: float
    backdrive_torque_nm: float
    peak_output_torque_nm: float
    continuous_output_torque_nm: float
    output_torque_density_nm_per_kg: float


class Actuator:
    """A motor coupled to a :class:`Gearbox`, evaluated at the output shaft."""

    def __init__(
        self, motor: MotorModel, gearbox: Gearbox, rated_bus_voltage_v: float | None = None
    ):
        self.motor = motor
        self.gearbox = gearbox
        if rated_bus_voltage_v is None:
            params = getattr(motor, "params", None)
            rated_bus_voltage_v = getattr(params, "rated_bus_voltage_v", 48.0)
        self.rated_bus_voltage_v = rated_bus_voltage_v

    # ------------------------------------------------------------------ operating point
    def evaluate_operating_point(
        self,
        torque_nm: float,
        speed_rad_s: float,
        bus_voltage_v: float,
        ambient_temp_c: float,
    ) -> MotorOperatingResult:
        gb = self.gearbox
        t_motor = gb.input_torque(torque_nm)
        w_motor = gb.input_speed(speed_rad_s)
        r = self.motor.evaluate_operating_point(t_motor, w_motor, bus_voltage_v, ambient_temp_c)
        gear_loss = gb.gear_loss_w(torque_nm, speed_rad_s)

        mf = r.feasibility
        flags = FeasibilityFlags(
            voltage_ok=mf.voltage_ok,
            current_ok=mf.current_ok,
            winding_temp_ok=mf.winding_temp_ok,
            magnet_temp_ok=mf.magnet_temp_ok,
            saturation_ok=mf.saturation_ok,
            demag_ok=mf.demag_ok,
            mechanical_ok=mf.mechanical_ok and abs(torque_nm) <= gb.max_output_torque_nm,
        )
        return MotorOperatingResult(
            torque_nm=torque_nm,
            speed_rad_s=speed_rad_s,
            phase_current_a_rms=r.phase_current_a_rms,
            phase_voltage_v_rms=r.phase_voltage_v_rms,
            copper_loss_w=r.copper_loss_w,
            core_loss_w=r.core_loss_w,
            magnet_eddy_loss_w=r.magnet_eddy_loss_w,
            mechanical_loss_w=r.mechanical_loss_w + gear_loss,
            winding_temp_c=r.winding_temp_c,
            magnet_temp_c=r.magnet_temp_c,
            feasibility=flags,
        )

    # ------------------------------------------------------------------ mass / inertia
    def mass_properties(self) -> MassProperties:
        m = self.motor.mass_properties()
        return MassProperties(
            total_mass_kg=m.total_mass_kg + self.gearbox.mass_kg,
            rotor_inertia_kg_m2=self.gearbox.reflected_inertia_at_output(m.rotor_inertia_kg_m2),
            active_mass_kg=m.active_mass_kg,
            magnet_mass_kg=m.magnet_mass_kg,
            copper_mass_kg=m.copper_mass_kg,
            iron_mass_kg=m.iron_mass_kg,
        )

    def reflected_inertia_at_output(self) -> float:
        j_motor = self.motor.mass_properties().rotor_inertia_kg_m2
        return self.gearbox.reflected_inertia_at_output(j_motor)

    def backdrive_torque_nm(self) -> float:
        return self.gearbox.backdrive_torque_nm()

    # ------------------------------------------------------------------ envelope
    def torque_speed_envelope(self) -> TorqueSpeedEnvelope:
        gb = self.gearbox
        env = self.motor.torque_speed_envelope()
        speed_out = env.speed_rad_s / gb.ratio
        peak_out = np.clip(
            gb.forward_efficiency * gb.ratio * (env.peak_torque_nm - gb.no_load_drag_torque_nm),
            0.0,
            gb.max_output_torque_nm,
        )
        cont_out = np.clip(
            gb.forward_efficiency
            * gb.ratio
            * (env.continuous_torque_nm - gb.no_load_drag_torque_nm),
            0.0,
            gb.max_output_torque_nm,
        )
        return TorqueSpeedEnvelope(
            speed_rad_s=speed_out, peak_torque_nm=peak_out, continuous_torque_nm=cont_out
        )

    # ------------------------------------------------------------------ efficiency map
    def efficiency_map(self, n_speeds: int = 30, n_torques: int = 30) -> EfficiencyMap:
        bus = self.rated_bus_voltage_v
        env = self.torque_speed_envelope()
        speeds = np.linspace(0.0, float(np.max(env.speed_rad_s)), n_speeds)
        t_max = float(np.max(env.peak_torque_nm)) or 1.0
        torques = np.linspace(t_max / n_torques, t_max, n_torques)
        eff = np.full((n_torques, n_speeds), np.nan)
        for j, w in enumerate(speeds):
            for i, t in enumerate(torques):
                res = self.evaluate_operating_point(t, w, bus, 25.0)
                if res.feasibility.feasible and res.mechanical_power_w > 0.0:
                    eff[i, j] = res.efficiency
        return EfficiencyMap(speed_rad_s=speeds, torque_nm=torques, efficiency=eff)

    # ------------------------------------------------------------------ summary
    def properties(self) -> ActuatorProperties:
        env = self.torque_speed_envelope()
        mass = self.mass_properties()
        peak = float(np.max(env.peak_torque_nm))
        cont = float(np.max(env.continuous_torque_nm))
        density = peak / mass.total_mass_kg if mass.total_mass_kg else 0.0
        return ActuatorProperties(
            gear_ratio=self.gearbox.ratio,
            total_mass_kg=mass.total_mass_kg,
            reflected_inertia_kg_m2=mass.rotor_inertia_kg_m2,
            backdrive_torque_nm=self.backdrive_torque_nm(),
            peak_output_torque_nm=peak,
            continuous_output_torque_nm=cont,
            output_torque_density_nm_per_kg=density,
        )
