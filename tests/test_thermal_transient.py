"""Tests for transient duty-cycle thermal integration."""

from __future__ import annotations

import numpy as np
import pytest

from robotactuatormdo.geometry.radial_flux import RadialPMGeometry, size_radial_pm
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.motors.radial_pmsm import RadialPMSM
from robotactuatormdo.requirements.duty_cycle import DutyCycle
from robotactuatormdo.results import ThermalHistory
from robotactuatormdo.thermal.duty_cycle import integrate
from robotactuatormdo.thermal.lumped_network import radial_pm_network

GEOM = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008,
    turns_per_phase=40,
)


def make_motor() -> RadialPMSM:
    return RadialPMSM(size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER))


def _network(ambient_c=30.0):
    return radial_pm_network(
        r_winding_stator_c_w=0.15, r_stator_ambient_c_w=0.4, r_magnet_ambient_c_w=2.0,
        ambient_c=ambient_c,
    )


def test_history_shape_and_names():
    motor = make_motor()
    net = _network()
    duty = DutyCycle.constant(torque_nm=5.0, speed_rad_s=8.0, duration_s=20.0)
    hist = integrate(motor, duty, net, bus_voltage_v=48.0)
    assert isinstance(hist, ThermalHistory)
    assert hist.node_names == net.free_node_names
    assert hist.node_temps_c.shape == (len(net.free_node_names), duty.time_s.size)


def test_constant_load_reaches_steady():
    from dataclasses import replace

    net = _network(ambient_c=30.0)
    t_op, w_op = 5.0, 8.0
    # The transient under a long constant load must converge to the motor's own steady network
    # temperature (same network, same losses).
    params = size_radial_pm(GEOM, NDFEB_N42, M250_35A, COPPER)
    motor = RadialPMSM(params)
    motor_net = RadialPMSM(replace(params, thermal_network=net))
    steady_t = motor_net.evaluate_operating_point(t_op, w_op, 48.0, 30.0).winding_temp_c

    times = np.linspace(0.0, 4000.0, 800)
    duty = DutyCycle(
        time_s=times, torque_nm=np.full_like(times, t_op), speed_rad_s=np.full_like(times, w_op)
    )
    hist = integrate(motor, duty, net, bus_voltage_v=48.0)
    assert hist.temps_of("winding")[-1] == pytest.approx(steady_t, rel=1e-2)


def test_idle_segment_cools():
    motor = make_motor()
    net = _network()
    # push hard, then idle; sample finely so the transient is resolved.
    push = np.linspace(0.0, 60.0, 120)
    idle = np.linspace(60.0, 600.0, 200)[1:]
    times = np.concatenate([push, idle])
    torque = np.concatenate([np.full(push.size, 7.0), np.full(idle.size, 0.2)])
    speed = np.full(times.size, 8.0)
    duty = DutyCycle(time_s=times, torque_nm=torque, speed_rad_s=speed)
    w = integrate(motor, duty, net, bus_voltage_v=48.0).temps_of("winding")
    end_of_push = w[push.size - 1]
    end_of_idle = w[-1]
    assert end_of_idle < end_of_push  # it cools during idle


def test_undercooled_design_ratchets():
    motor = make_motor()
    hot = radial_pm_network(
        r_winding_stator_c_w=0.5, r_stator_ambient_c_w=3.0, r_magnet_ambient_c_w=5.0, ambient_c=30.0
    )
    task = DutyCycle.from_segments([(3.0, 7.0, 8.0), (3.0, 0.5, 8.0)])
    n = 8
    period = task.duration_s
    times = np.concatenate([task.time_s + i * period for i in range(n)])
    # nudge duplicate boundaries
    for i in range(1, times.size):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + 1e-9
    duty = DutyCycle(
        time_s=times,
        torque_nm=np.tile(task.torque_nm, n),
        speed_rad_s=np.tile(task.speed_rad_s, n),
    )
    w = integrate(motor, duty, hot, bus_voltage_v=48.0).temps_of("winding")
    end_idx = [(i + 1) * task.time_s.size - 1 for i in range(n)]
    ends = [w[j] for j in end_idx]
    assert ends[-1] > ends[0]  # heat accumulates cycle over cycle
