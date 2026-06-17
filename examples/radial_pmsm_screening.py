"""Screening example: size a radial PMSM and evaluate it over a duty cycle.

Demonstrates the Phase 1 + Phase 2 vertical slice end to end:
geometry + materials -> RadialPMParameters -> RadialPMSM -> evaluate_over_duty_cycle
-> MissionResult.

Run with:  .venv/bin/python examples/radial_pmsm_screening.py
"""

from __future__ import annotations

import numpy as np

from robotactuatormdo import (
    DutyCycle,
    RadialPMGeometry,
    RadialPMSM,
    evaluate_over_duty_cycle,
    size_radial_pm,
)
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42


def main() -> None:
    geom = RadialPMGeometry(
        air_gap_radius_m=0.045,
        stack_length_m=0.040,
        outer_radius_m=0.070,
        pole_pairs=7,
        slots=24,
        magnet_thickness_m=0.004,
        air_gap_m=0.0008,
        turns_per_phase=40,
    )
    params = size_radial_pm(geom, NDFEB_N42, M250_35A, COPPER, rated_bus_voltage_v=48.0)
    motor = RadialPMSM(params)

    mass = motor.mass_properties()
    env = motor.torque_speed_envelope()
    peak_torque = float(np.max(env.peak_torque_nm))

    print("=== sized radial PMSM ===")
    print(f"  flux linkage psi_m : {params.flux_linkage_wb*1e3:.2f} mWb")
    print(f"  phase R (20C)      : {params.r_s_ohm_20*1e3:.2f} mOhm")
    print(f"  L_s                : {params.l_d_h*1e6:.1f} uH")
    print(f"  max phase current  : {params.max_phase_current_a_rms:.1f} A_rms")
    print(f"  total mass         : {mass.total_mass_kg*1e3:.0f} g")
    td = peak_torque / mass.total_mass_kg
    print(f"  peak torque        : {peak_torque:.2f} N*m  ({td:.1f} N*m/kg)")

    # A simple stance/swing duty cycle at the joint output (direct drive here).
    duty = DutyCycle.from_segments(
        [
            (0.4, 0.6 * peak_torque, 8.0),   # stance: high torque, low speed
            (0.3, 0.1 * peak_torque, 30.0),  # swing: low torque, higher speed
        ]
    )
    mission = evaluate_over_duty_cycle(motor, duty, bus_voltage_v=48.0, ambient_temp_c=30.0)

    print("=== mission over duty cycle ===")
    print(f"  RMS phase current  : {mission.rms_phase_current_a:.1f} A")
    print(f"  peak phase current : {mission.peak_phase_current_a:.1f} A")
    print(f"  mechanical energy  : {mission.mechanical_energy_j:.1f} J")
    print(f"  electrical energy  : {mission.electrical_energy_j:.1f} J")
    print(f"  average efficiency : {mission.average_efficiency*100:.1f} %")
    print(f"  peak winding temp  : {mission.peak_winding_temp_c:.1f} C")
    print(f"  all feasible       : {mission.all_feasible}")


if __name__ == "__main__":
    main()
