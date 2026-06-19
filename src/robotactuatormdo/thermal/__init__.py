"""Lumped-parameter thermal network and duty-cycle thermal integration."""

from __future__ import annotations

from robotactuatormdo.thermal.duty_cycle import integrate
from robotactuatormdo.thermal.lumped_network import (
    ThermalEdge,
    ThermalNetwork,
    ThermalNode,
    boundary_outflow_w,
    min_time_constant_s,
    radial_pm_network,
    single_node_network,
    solve_steady_state,
    solve_transient,
)

__all__ = [
    "ThermalNode",
    "ThermalEdge",
    "ThermalNetwork",
    "solve_steady_state",
    "solve_transient",
    "min_time_constant_s",
    "boundary_outflow_w",
    "single_node_network",
    "radial_pm_network",
    "integrate",
]
