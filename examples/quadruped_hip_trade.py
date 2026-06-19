"""Quadruped hip trade study: compare actuator architecture classes (brief §8).

Moderate torque, higher speed and backdrivability matter — a sweet spot for low-ratio QDD. Run
with:  PYTHONPATH=src .venv/bin/python examples/quadruped_hip_trade.py
"""

from __future__ import annotations

from _trade_common import architecture_candidates, print_comparison

from robotactuatormdo import DutyCycle, JointRequirement, compare_topologies


def main() -> None:
    req = JointRequirement(
        peak_torque_nm=24.0, continuous_rms_torque_nm=8.0, max_speed_rad_s=25.0,
        bus_voltage_v=48.0, max_phase_current_a_rms=60.0, envelope_outer_diameter_m=0.11,
        envelope_axial_length_m=0.05, max_mass_kg=5.0, ambient_temp_c=30.0,
        duty_cycle=DutyCycle.from_segments([(0.4, 18.0, 10.0), (0.6, 5.0, 22.0)]),
    )
    comp = compare_topologies(req, architecture_candidates(qdd_ratio=5.0),
                              requirement_name="quadruped_hip")
    print_comparison("quadruped hip", comp)


if __name__ == "__main__":
    main()
