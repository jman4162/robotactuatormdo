"""Transient thermal: heat build-up vs recovery over repeated tasks (Phase 5).

Sizes a radial PMSM, builds a 4-node thermal network, and integrates winding temperature over a
repeated stance/idle duty cycle. A well-cooled design recovers between cycles; an under-cooled one
ratchets toward its limit.

Run with:  PYTHONPATH=src .venv/bin/python examples/thermal_transient.py
"""

from __future__ import annotations

import numpy as np

from robotactuatormdo import (
    DutyCycle,
    RadialPMGeometry,
    RadialPMSM,
    integrate_thermal,
    radial_pm_network,
    size_radial_pm,
)
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

    # One task = 3 s hard push + 3 s idle; tile it into 10 repeats.
    task = DutyCycle.from_segments([(3.0, 7.0, 8.0), (3.0, 0.5, 8.0)])
    n_cycles = 10
    period = task.duration_s
    times = np.concatenate([task.time_s + i * period for i in range(n_cycles)])
    torque = np.tile(task.torque_nm, n_cycles)
    speed = np.tile(task.speed_rad_s, n_cycles)
    repeated = DutyCycle(time_s=_strictly_increasing(times), torque_nm=torque, speed_rad_s=speed)

    network = radial_pm_network(
        r_winding_stator_c_w=0.15, r_stator_ambient_c_w=0.4, r_magnet_ambient_c_w=2.0,
        ambient_c=30.0,
    )
    history = integrate_thermal(motor, repeated, network, bus_voltage_v=48.0)

    winding = history.temps_of("winding")
    print(f"=== winding temperature over {n_cycles} stance/idle cycles ===")
    print(f"  peak winding temp : {history.peak_temps_c['winding']:.1f} C")
    print(f"  final winding temp: {history.final_temps_c['winding']:.1f} C")
    # End-of-cycle temperature trend (recovery if flat, ratchet if rising).
    end_idx = [(i + 1) * task.time_s.size - 1 for i in range(n_cycles)]
    ends = [winding[j] for j in end_idx]
    print("  end-of-cycle winding temps: " + ", ".join(f"{t:.1f}" for t in ends))
    print(f"  per-cycle ratchet (last - first): {ends[-1] - ends[0]:+.1f} C")


def _strictly_increasing(t: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    out = t.astype(float).copy()
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


if __name__ == "__main__":
    main()
