"""Collaborative-robot elbow trade study: compare architecture classes (brief §8).

Lower torque, low speed, transparency/backdrivability prized — direct drive and low-ratio QDD are
both viable, so the trade is about inertia, efficiency, and cost. Run with:
PYTHONPATH=src .venv/bin/python examples/cobot_elbow_trade.py
"""

from __future__ import annotations

from _trade_common import architecture_candidates, print_comparison

from robotactuatormdo import DutyCycle, JointRequirement, compare_topologies


def main() -> None:
    req = JointRequirement(
        peak_torque_nm=8.0, continuous_rms_torque_nm=2.5, max_speed_rad_s=10.0,
        bus_voltage_v=48.0, max_phase_current_a_rms=60.0, envelope_outer_diameter_m=0.10,
        envelope_axial_length_m=0.05, max_mass_kg=5.0, ambient_temp_c=30.0,
        duty_cycle=DutyCycle.from_segments([(0.5, 5.0, 4.0), (0.5, 1.5, 8.0)]),
    )
    comp = compare_topologies(req, architecture_candidates(qdd_ratio=4.0),
                              requirement_name="cobot_elbow")
    print_comparison("cobot elbow", comp)


if __name__ == "__main__":
    main()
