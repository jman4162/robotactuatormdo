"""Humanoid knee trade study: compare actuator architecture classes (brief §8).

High torque, moderate speed — a regime where direct drive cannot meet the torque and a low-ratio
QDD earns its keep. Run with:  PYTHONPATH=src .venv/bin/python examples/humanoid_knee_trade.py
"""

from __future__ import annotations

from _trade_common import architecture_candidates, print_comparison

from robotactuatormdo import DutyCycle, JointRequirement, compare_topologies


def main() -> None:
    req = JointRequirement(
        peak_torque_nm=40.0, continuous_rms_torque_nm=15.0, max_speed_rad_s=20.0,
        bus_voltage_v=48.0, max_phase_current_a_rms=60.0, envelope_outer_diameter_m=0.12,
        envelope_axial_length_m=0.06, max_mass_kg=5.0, ambient_temp_c=30.0,
        duty_cycle=DutyCycle.from_segments([(0.6, 35.0, 6.0), (0.4, 8.0, 16.0)]),
    )
    comp = compare_topologies(req, architecture_candidates(), requirement_name="humanoid_knee")
    print_comparison("humanoid knee", comp)


if __name__ == "__main__":
    main()
