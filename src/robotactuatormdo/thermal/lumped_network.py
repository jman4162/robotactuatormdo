"""Multi-node lumped-parameter thermal network (TOPOLOGY brief §11).

A network is a set of nodes connected by conductive edges. A node carries a heat capacity
``C`` [J/K]; a node with ``fixed_temp_c`` set is a **boundary** (ambient/coolant) held at constant
temperature — so a "node-to-ambient resistance" is just an ordinary edge to a boundary node, which
keeps the assembly free of double-counting. Temperatures are in °C (identical to K for the
*differences* this linear, conduction-only model uses; there are no absolute-temperature/radiative
terms).

Steady state, for every free node i:  ``P_i + sum_j G_ij (T_j - T_i) = 0``.
Assembled over free nodes:  ``G_free @ T_free = P + b_boundary``  where each edge adds ``+g`` to
both endpoint diagonals and ``-g`` to free-free off-diagonals, and a boundary edge adds ``+g`` to
the free diagonal and ``+g*T_boundary`` to the RHS. ``G_free`` is symmetric positive-definite
whenever every free node has a conductive path to a boundary (else the solve raises).

Transient:  ``C dT/dt = P(t) - G T + b_boundary``, integrated with backward (implicit) Euler
``(C/dt + G) T^{n+1} = (C/dt) T^n + b^{n+1}`` — unconditionally stable for this stiff system, and
its fixed point is exactly the steady solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
]


@dataclass(frozen=True, slots=True)
class ThermalNode:
    """A lumped thermal node. ``fixed_temp_c`` set => boundary (held constant)."""

    name: str
    heat_capacity_j_per_k: float = 0.0
    fixed_temp_c: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("node name must be non-empty")
        if self.fixed_temp_c is None and self.heat_capacity_j_per_k <= 0.0:
            raise ValueError(f"free node {self.name!r} needs positive heat_capacity_j_per_k")

    @property
    def is_boundary(self) -> bool:
        return self.fixed_temp_c is not None


@dataclass(frozen=True, slots=True)
class ThermalEdge:
    """Conductive coupling between two nodes; stores conductance G = 1/R [W/K]."""

    node_a: str
    node_b: str
    conductance_w_per_k: float

    def __post_init__(self) -> None:
        if self.node_a == self.node_b:
            raise ValueError("self-edges are not allowed")
        if self.conductance_w_per_k <= 0.0:
            raise ValueError("conductance_w_per_k must be positive")

    @classmethod
    def from_resistance(cls, node_a: str, node_b: str, resistance_k_per_w: float) -> ThermalEdge:
        if resistance_k_per_w <= 0.0:
            raise ValueError("resistance_k_per_w must be positive")
        return cls(node_a, node_b, 1.0 / resistance_k_per_w)


@dataclass(frozen=True, slots=True)
class ThermalNetwork:
    """A lumped thermal network. Ambient/coolant are nodes with ``fixed_temp_c`` set."""

    nodes: tuple[ThermalNode, ...]
    edges: tuple[ThermalEdge, ...]

    def __post_init__(self) -> None:
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("duplicate node names")
        name_set = set(names)
        for e in self.edges:
            for nm in (e.node_a, e.node_b):
                if nm not in name_set:
                    raise ValueError(f"edge references unknown node {nm!r}")
        if not any(not n.is_boundary for n in self.nodes):
            raise ValueError("network has no free nodes to solve")
        if not any(n.is_boundary for n in self.nodes):
            raise ValueError("network has no boundary node")
        # Validate solvability (every free node has a path to a boundary => SPD).
        _, _, g, _, _ = _assemble(self)
        try:
            np.linalg.cholesky(g)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - exercised by a test
            raise ValueError(
                "thermal network is singular (a free node may lack a path to a boundary)"
            ) from exc

    @property
    def free_node_names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes if not n.is_boundary)

    @property
    def boundary_node_names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes if n.is_boundary)

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes)


def _assemble(
    network: ThermalNetwork,
) -> tuple[tuple[str, ...], dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Return (free_names, free_index, G_free, C_free, b_boundary)."""
    free = [n for n in network.nodes if not n.is_boundary]
    free_names = tuple(n.name for n in free)
    idx = {name: i for i, name in enumerate(free_names)}
    boundary_temp = {n.name: float(n.fixed_temp_c) for n in network.nodes if n.is_boundary}

    n = len(free)
    g = np.zeros((n, n))
    b = np.zeros(n)
    for edge in network.edges:
        cond = edge.conductance_w_per_k
        a, bb = edge.node_a, edge.node_b
        a_free, b_free = a in idx, bb in idx
        if a_free and b_free:
            ia, ib = idx[a], idx[bb]
            g[ia, ia] += cond
            g[ib, ib] += cond
            g[ia, ib] -= cond
            g[ib, ia] -= cond
        elif a_free:  # bb is boundary
            ia = idx[a]
            g[ia, ia] += cond
            b[ia] += cond * boundary_temp[bb]
        elif b_free:  # a is boundary
            ib = idx[bb]
            g[ib, ib] += cond
            b[ib] += cond * boundary_temp[a]
        # both boundary: no effect on the free system
    c = np.array([node.heat_capacity_j_per_k for node in free], dtype=float)
    return free_names, idx, g, c, b


def _power_vector(
    free_names: tuple[str, ...], idx: dict[str, int], power_w: dict[str, float]
) -> np.ndarray:
    p = np.zeros(len(free_names))
    for name, watts in power_w.items():
        if name not in idx:
            raise ValueError(f"power injected at non-free or unknown node {name!r}")
        p[idx[name]] += watts
    return p


def solve_steady_state(network: ThermalNetwork, power_w: dict[str, float]) -> dict[str, float]:
    """Steady-state node temperatures [°C] for the given per-node injected power [W]."""
    free_names, idx, g, _c, b = _assemble(network)
    p = _power_vector(free_names, idx, power_w)
    t_free = np.linalg.solve(g, p + b)
    temps = {name: float(t_free[i]) for i, name in enumerate(free_names)}
    for node in network.nodes:
        if node.is_boundary:
            temps[node.name] = float(node.fixed_temp_c)
    return temps


def solve_transient(
    network: ThermalNetwork,
    times_s: np.ndarray,
    power_w_series: list[dict[str, float]],
    initial_temps_c: dict[str, float] | None = None,
) -> np.ndarray:
    """Backward-Euler transient solve. Returns free-node temps, shape ``(n_free, n_time)`` [°C]."""
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times_s must be a 1-D array with >= 2 samples")
    if len(power_w_series) != times.size:
        raise ValueError("power_w_series must have one entry per time sample")
    free_names, idx, g, c, b = _assemble(network)
    n = len(free_names)

    t = np.zeros((n, times.size))
    if initial_temps_c is None:
        # default: start every free node at the (mean) boundary temperature
        bt = [float(node.fixed_temp_c) for node in network.nodes if node.is_boundary]
        t0 = np.full(n, float(np.mean(bt)))
    else:
        t0 = np.array([initial_temps_c.get(name, 0.0) for name in free_names], dtype=float)
    t[:, 0] = t0

    for k in range(1, times.size):
        dt = times[k] - times[k - 1]
        if dt <= 0.0:
            raise ValueError("times_s must be strictly increasing")
        p = _power_vector(free_names, idx, power_w_series[k])
        a_mat = np.diag(c / dt) + g
        rhs = (c / dt) * t[:, k - 1] + p + b
        t[:, k] = np.linalg.solve(a_mat, rhs)
    return t


def min_time_constant_s(network: ThermalNetwork) -> float:
    """Smallest free-node thermal time constant ``min_i C_i / G_ii`` [s] (for sizing dt)."""
    _free, _idx, g, c, _b = _assemble(network)
    return float(np.min(c / np.diag(g)))


def boundary_outflow_w(network: ThermalNetwork, temps: dict[str, float]) -> float:
    """Total heat leaving through boundary edges [W] (for energy-balance checks)."""
    total = 0.0
    boundary = {n.name for n in network.nodes if n.is_boundary}
    for edge in network.edges:
        a, b = edge.node_a, edge.node_b
        if (a in boundary) ^ (b in boundary):
            free_node, bnd_node = (b, a) if a in boundary else (a, b)
            total += edge.conductance_w_per_k * (temps[free_node] - temps[bnd_node])
    return total


def single_node_network(
    r_th_winding_ambient_c_w: float,
    r_th_magnet_ambient_c_w: float,
    *,
    c_winding_j_per_k: float = 400.0,
    c_magnet_j_per_k: float = 200.0,
    ambient_c: float = 25.0,
) -> ThermalNetwork:
    """Two free nodes (winding, magnet) each tied to ambient — reproduces the single-node model."""
    return ThermalNetwork(
        nodes=(
            ThermalNode("winding", c_winding_j_per_k),
            ThermalNode("magnet", c_magnet_j_per_k),
            ThermalNode("ambient", fixed_temp_c=ambient_c),
        ),
        edges=(
            ThermalEdge.from_resistance("winding", "ambient", r_th_winding_ambient_c_w),
            ThermalEdge.from_resistance("magnet", "ambient", r_th_magnet_ambient_c_w),
        ),
    )


def radial_pm_network(
    r_winding_stator_c_w: float,
    r_stator_ambient_c_w: float,
    r_magnet_ambient_c_w: float,
    *,
    c_winding_j_per_k: float = 300.0,
    c_stator_j_per_k: float = 800.0,
    c_magnet_j_per_k: float = 200.0,
    ambient_c: float = 25.0,
) -> ThermalNetwork:
    """Minimal 4-node radial-PMSM network: winding -> stator -> ambient, and magnet -> ambient."""
    return ThermalNetwork(
        nodes=(
            ThermalNode("winding", c_winding_j_per_k),
            ThermalNode("stator", c_stator_j_per_k),
            ThermalNode("magnet", c_magnet_j_per_k),
            ThermalNode("ambient", fixed_temp_c=ambient_c),
        ),
        edges=(
            ThermalEdge.from_resistance("winding", "stator", r_winding_stator_c_w),
            ThermalEdge.from_resistance("stator", "ambient", r_stator_ambient_c_w),
            ThermalEdge.from_resistance("magnet", "ambient", r_magnet_ambient_c_w),
        ),
    )
