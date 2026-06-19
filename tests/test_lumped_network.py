"""Tests for the multi-node lumped thermal network (steady + transient)."""

from __future__ import annotations

import numpy as np
import pytest

from robotactuatormdo.thermal.lumped_network import (
    ThermalEdge,
    ThermalNetwork,
    ThermalNode,
    boundary_outflow_w,
    min_time_constant_s,
    solve_steady_state,
    solve_transient,
)


def _single_node(r=0.5, c=100.0, amb=25.0):
    return ThermalNetwork(
        nodes=(ThermalNode("w", c), ThermalNode("amb", fixed_temp_c=amb)),
        edges=(ThermalEdge.from_resistance("w", "amb", r),),
    )


def test_single_node_matches_analytic():
    net = _single_node(r=0.5, amb=25.0)
    temps = solve_steady_state(net, {"w": 100.0})
    assert temps["w"] == pytest.approx(25.0 + 0.5 * 100.0, abs=1e-12)


def test_two_node_series():
    # w --R1-- h --R2-- amb, power injected at w only.
    net = ThermalNetwork(
        nodes=(
            ThermalNode("w", 100.0),
            ThermalNode("h", 200.0),
            ThermalNode("amb", fixed_temp_c=20.0),
        ),
        edges=(
            ThermalEdge.from_resistance("w", "h", 0.3),
            ThermalEdge.from_resistance("h", "amb", 0.7),
        ),
    )
    temps = solve_steady_state(net, {"w": 50.0})
    assert temps["h"] == pytest.approx(20.0 + 0.7 * 50.0)
    assert temps["w"] == pytest.approx(20.0 + (0.3 + 0.7) * 50.0)


def test_energy_balance():
    net = ThermalNetwork(
        nodes=(
            ThermalNode("w", 100.0),
            ThermalNode("h", 200.0),
            ThermalNode("amb", fixed_temp_c=20.0),
        ),
        edges=(
            ThermalEdge.from_resistance("w", "h", 0.3),
            ThermalEdge.from_resistance("h", "amb", 0.7),
        ),
    )
    temps = solve_steady_state(net, {"w": 50.0})
    assert boundary_outflow_w(net, temps) == pytest.approx(50.0)


def test_transient_reaches_steady_state():
    net = _single_node(r=0.5, c=100.0, amb=25.0)
    tau = 0.5 * 100.0  # RC = 50 s
    times = np.linspace(0.0, 10 * tau, 400)
    power = [{"w": 100.0}] * times.size
    traj = solve_transient(net, times, power, initial_temps_c={"w": 25.0})
    steady = solve_steady_state(net, {"w": 100.0})["w"]
    assert traj[0, -1] == pytest.approx(steady, rel=1e-3)


def test_rc_step_response():
    r, c, p, amb = 0.5, 100.0, 100.0, 25.0
    net = _single_node(r=r, c=c, amb=amb)
    tau = r * c
    times = np.linspace(0.0, 3 * tau, 600)  # small dt = tau/200
    traj = solve_transient(net, times, [{"w": p}] * times.size, initial_temps_c={"w": amb})
    analytic = amb + r * p * (1.0 - np.exp(-times / tau))
    assert np.max(np.abs(traj[0] - analytic)) < 0.02 * r * p


def test_min_time_constant():
    net = _single_node(r=0.5, c=100.0)
    assert min_time_constant_s(net) == pytest.approx(0.5 * 100.0)


def test_conductance_round_trip_and_validation():
    assert ThermalEdge.from_resistance("a", "b", 0.25).conductance_w_per_k == pytest.approx(4.0)
    with pytest.raises(ValueError):
        ThermalEdge.from_resistance("a", "b", 0.0)
    with pytest.raises(ValueError):
        ThermalEdge("a", "a", 1.0)  # self-edge
    with pytest.raises(ValueError):  # duplicate node names
        ThermalNetwork(nodes=(ThermalNode("w", 1.0), ThermalNode("w", fixed_temp_c=25.0)), edges=())
    with pytest.raises(ValueError):  # unknown node referenced
        ThermalNetwork(
            nodes=(ThermalNode("w", 1.0), ThermalNode("amb", fixed_temp_c=25.0)),
            edges=(ThermalEdge.from_resistance("w", "x", 1.0),),
        )
    with pytest.raises(ValueError):  # floating free node (no boundary path) -> singular
        ThermalNetwork(
            nodes=(ThermalNode("w", 1.0), ThermalNode("amb", fixed_temp_c=25.0)),
            edges=(),
        )


def test_assembled_matrix_symmetric_spd():
    from robotactuatormdo.thermal.lumped_network import _assemble

    net = ThermalNetwork(
        nodes=(
            ThermalNode("w", 100.0),
            ThermalNode("h", 200.0),
            ThermalNode("amb", fixed_temp_c=20.0),
        ),
        edges=(
            ThermalEdge.from_resistance("w", "h", 0.3),
            ThermalEdge.from_resistance("h", "amb", 0.7),
        ),
    )
    _, _, g, _, _ = _assemble(net)
    assert np.allclose(g, g.T)
    assert np.all(np.linalg.eigvalsh(g) > 0.0)
