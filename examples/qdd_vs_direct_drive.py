"""Compare a direct-drive vs a QDD (low-ratio planetary) actuator on the same motor.

Shows the Phase 3 actuator-level tradeoff: gearing multiplies output torque but adds reflected
inertia, backdrive torque, and gear loss. Both actuators are evaluated over the same high-torque /
low-speed output duty cycle through the Phase 1 reducer.

Run with:  PYTHONPATH=src .venv/bin/python examples/qdd_vs_direct_drive.py
"""

from __future__ import annotations

from robotactuatormdo import (
    Actuator,
    DutyCycle,
    RadialPMGeometry,
    RadialPMSM,
    evaluate_over_duty_cycle,
    quasi_direct_drive,
    size_radial_pm,
)
from robotactuatormdo.actuators.gearbox import direct_drive
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42


def _report(name: str, actuator, duty: DutyCycle) -> None:
    props = actuator.properties()
    mission = evaluate_over_duty_cycle(actuator, duty, bus_voltage_v=48.0, ambient_temp_c=30.0)
    print(f"--- {name} ---")
    print(f"  gear ratio          : {props.gear_ratio:.1f}")
    print(f"  peak output torque  : {props.peak_output_torque_nm:.1f} N*m")
    print(f"  output torque dens. : {props.output_torque_density_nm_per_kg:.1f} N*m/kg")
    print(f"  reflected inertia   : {props.reflected_inertia_kg_m2*1e4:.2f} x1e-4 kg*m^2")
    print(f"  backdrive torque    : {props.backdrive_torque_nm:.3f} N*m")
    print(f"  total mass          : {props.total_mass_kg*1e3:.0f} g")
    print(f"  mission efficiency  : {mission.average_efficiency*100:.1f} %")
    print(f"  all feasible        : {mission.all_feasible}")


def main() -> None:
    geom = RadialPMGeometry(
        air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
        pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
        turns_per_phase=40,
    )
    motor = RadialPMSM(size_radial_pm(geom, NDFEB_N42, M250_35A, COPPER))

    # High-torque, low-speed joint demand (where reduction earns its keep).
    duty = DutyCycle.from_segments([(0.5, 25.0, 3.0), (0.5, 6.0, 12.0)])

    _report("direct drive", Actuator(motor, direct_drive()), duty)
    _report("QDD planetary (6:1)", quasi_direct_drive(motor, ratio=6.0), duty)


if __name__ == "__main__":
    main()
