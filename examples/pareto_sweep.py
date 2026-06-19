"""Pareto sweep: mass vs torque density over a radial-PMSM geometry grid.

Demonstrates the studies grid generator + non-dominated sort. Run with:
PYTHONPATH=src .venv/bin/python examples/pareto_sweep.py
"""

from __future__ import annotations

from dataclasses import replace

from robotactuatormdo import (
    DutyCycle,
    JointRequirement,
    RadialPMGeometry,
    RadialPMSM,
    size_radial_pm,
)
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.studies.pareto import grid, pareto_front
from robotactuatormdo.studies.scoring import Objective, score_candidate

BASE = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
    turns_per_phase=40,
)


def _motor(params):
    geom = replace(
        BASE,
        stack_length_m=params["stack_length_m"],
        magnet_thickness_m=params["magnet_thickness_m"],
    )
    return RadialPMSM(size_radial_pm(geom, NDFEB_N42, M250_35A, COPPER))


def main() -> None:
    req = JointRequirement(
        peak_torque_nm=3.0, continuous_rms_torque_nm=0.5, max_speed_rad_s=20.0,
        bus_voltage_v=48.0, max_phase_current_a_rms=60.0, envelope_outer_diameter_m=0.1,
        envelope_axial_length_m=0.1, max_mass_kg=10.0, ambient_temp_c=30.0,
        duty_cycle=DutyCycle.constant(2.0, 8.0, 1.0),
    )
    cands = grid(
        "radial", "radial_direct_drive", _motor,
        {"stack_length_m": [0.02, 0.03, 0.04, 0.05], "magnet_thickness_m": [0.003, 0.004, 0.005]},
    )
    objs = (Objective.MASS_KG, Objective.TORQUE_DENSITY)
    scores = [score_candidate(c, req, objs) for c in cands]
    front = pareto_front(scores, objs)

    print(f"=== geometry sweep: {len(scores)} candidates, {len(front.front)} on the front ===")
    for s in front.front:
        print(f"  mass {s.objectives[Objective.MASS_KG]*1e3:5.0f} g  |  "
              f"torque density {s.objectives[Objective.TORQUE_DENSITY]:5.1f} N*m/kg")


if __name__ == "__main__":
    main()
