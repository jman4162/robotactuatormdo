"""Low-voltage vs higher-voltage bus on the same actuator (Phase 4).

Shows the brief's core point: at a low pack voltage the DC link sags, phase/DC current rise, and
the actuator becomes current/voltage-limited. The same QDD actuator and duty cycle are evaluated
through a PowerStage at 24 V vs 48 V.

Run with:  PYTHONPATH=src .venv/bin/python examples/low_voltage_current_limit.py
"""

from __future__ import annotations

from robotactuatormdo import (
    Battery,
    Cable,
    DutyCycle,
    PowerStage,
    RadialPMGeometry,
    RadialPMSM,
    evaluate_over_duty_cycle,
    quasi_direct_drive,
    size_radial_pm,
)
from robotactuatormdo.electronics.inverter import gan
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42


def main() -> None:
    geom = RadialPMGeometry(
        air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
        pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
        turns_per_phase=40,
    )
    motor = RadialPMSM(size_radial_pm(geom, NDFEB_N42, M250_35A, COPPER))
    actuator = quasi_direct_drive(motor, ratio=6.0)

    inverter = gan(max_phase_current_a_rms=60.0)
    battery = Battery(internal_resistance_ohm=0.05, max_current_a=120.0, mass_kg=0.5)
    cable = Cable(resistance_ohm=0.01)

    # Demanding output point + a representative duty cycle.
    op = (40.0, 6.0)  # torque [N*m], output speed [rad/s]
    duty = DutyCycle.from_segments([(0.5, 40.0, 4.0), (0.5, 10.0, 18.0)])

    for pack_v in (24.0, 48.0):
        stage = PowerStage(actuator, inverter, battery, cable)
        link = stage.link_state(*op, bus_voltage_v=pack_v, ambient_temp_c=30.0)
        op_res = stage.evaluate_operating_point(*op, bus_voltage_v=pack_v, ambient_temp_c=30.0)
        mission = evaluate_over_duty_cycle(stage, duty, bus_voltage_v=pack_v, ambient_temp_c=30.0)
        sag = pack_v - link.dc_link_voltage_v
        fb = op_res.feasibility
        print(f"--- {pack_v:.0f} V pack ---")
        print(f"  at {op[0]:.0f} N*m / {op[1]:.0f} rad/s:")
        print(f"    DC-link voltage : {link.dc_link_voltage_v:.1f} V (sag {sag:.1f} V)")
        print(f"    DC current      : {link.dc_current_a:.1f} A")
        print(f"    phase current   : {op_res.phase_current_a_rms:.1f} A_rms")
        print(f"    inverter loss   : {link.inverter_loss_w:.1f} W")
        print(f"    feasible        : {fb.feasible} "
              f"(voltage_ok={fb.voltage_ok}, source_ok={fb.source_ok})")
        print(f"  mission: eff {mission.average_efficiency*100:.1f} %, "
              f"all_feasible {mission.all_feasible}")


if __name__ == "__main__":
    main()
