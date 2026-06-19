"""Shared helpers for the joint benchmark examples (brief §8).

Builds architecture-class candidates (radial direct-drive, radial+planetary QDD, radial+harmonic,
and a commercial baseline) wired to the real builders, with a first-order BOM cost attached so the
cost axis participates. Run the joint examples with ``PYTHONPATH=src``.
"""

from __future__ import annotations

from robotactuatormdo import (
    Actuator,
    RadialPMGeometry,
    RadialPMSM,
    size_radial_pm,
)
from robotactuatormdo.actuators.gearbox import harmonic, planetary
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.cost import bom_cost
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.motors.commercial_catalog import CommercialMotor, CommercialMotorSpec
from robotactuatormdo.studies.candidate import FactoryCandidate

BASE_GEOM = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
    turns_per_phase=40,
)


def _sized():
    return size_radial_pm(BASE_GEOM, NDFEB_N42, M250_35A, COPPER)


def _motor_cost_usd() -> float:
    p = _sized()
    cb = bom_cost(p.magnet_mass_kg, p.copper_mass_kg, p.iron_mass_kg, NDFEB_N42, M250_35A, COPPER)
    return cb.total_usd


def architecture_candidates(qdd_ratio: float = 6.0, harmonic_ratio: float = 50.0):
    """The architecture classes compared at each joint."""
    motor_cost = _motor_cost_usd()
    commercial = CommercialMotorSpec(
        kt_nm_per_a_rms=0.6, r_phase_ohm=0.08, pole_pairs=10, max_phase_current_a_rms=50.0,
        rated_torque_nm=14.0, peak_torque_nm=35.0, max_speed_rad_s=45.0, rated_bus_voltage_v=48.0,
        total_mass_kg=1.8, rotor_inertia_kg_m2=1.2e-3,
    )
    return [
        FactoryCandidate("radial direct-drive", "radial_direct_drive",
                         lambda p: RadialPMSM(_sized()), cost_usd=motor_cost),
        FactoryCandidate(f"radial + planetary {qdd_ratio:g}:1", "radial_planetary_qdd",
                         lambda p: Actuator(RadialPMSM(_sized()), planetary(qdd_ratio)),
                         cost_usd=motor_cost + 30.0),
        FactoryCandidate(f"radial + harmonic {harmonic_ratio:g}:1", "radial_harmonic",
                         lambda p: Actuator(RadialPMSM(_sized()), harmonic(harmonic_ratio)),
                         cost_usd=motor_cost + 120.0),
        FactoryCandidate("commercial geared baseline", "commercial_baseline",
                         lambda p: CommercialMotor(commercial), cost_usd=250.0),
    ]


def print_comparison(title, comparison) -> None:
    print(f"=== {title} ===")
    print(f"  feasible classes: {', '.join(comparison.feasible_classes) or '(none)'}")
    for row in comparison.rows:
        status = "feasible" if row.feasible else "INFEASIBLE"
        print(f"    {row.architecture_class:26s} {status}")
    print("  per-axis winners:")
    for obj, who in comparison.winners.items():
        print(f"    {obj.value:26s}: {who}")
