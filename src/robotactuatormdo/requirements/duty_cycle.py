"""Duty cycle: a time series of output (joint-shaft) torque and speed demand.

The duty cycle is the demand side of the trade study. It is expressed at the **output/joint**
shaft in SI units; the evaluation layer maps it to a motor shaft through a gear ratio. Its
reductions (RMS torque, peak torque, mechanical energy) are the basis for continuous-vs-peak
sizing and mission-energy metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from robotactuatormdo._numeric import trapezoid

__all__ = ["DutyCycle"]


@dataclass(frozen=True, slots=True)
class DutyCycle:
    """A sampled output duty cycle. ``time_s``, ``torque_nm``, ``speed_rad_s`` are parallel arrays.

    Time must be strictly increasing with at least two samples. Integrals use the trapezoidal
    rule over ``time_s``.
    """

    time_s: np.ndarray
    torque_nm: np.ndarray
    speed_rad_s: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=float)
        tau = np.asarray(self.torque_nm, dtype=float)
        omega = np.asarray(self.speed_rad_s, dtype=float)
        object.__setattr__(self, "time_s", t)
        object.__setattr__(self, "torque_nm", tau)
        object.__setattr__(self, "speed_rad_s", omega)

        if not (t.ndim == tau.ndim == omega.ndim == 1):
            raise ValueError("time_s, torque_nm, speed_rad_s must be 1-D arrays")
        if not (t.size == tau.size == omega.size):
            raise ValueError("time_s, torque_nm, speed_rad_s must have equal length")
        if t.size < 2:
            raise ValueError("a duty cycle needs at least 2 samples")
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("time_s must be strictly increasing")

    # ------------------------------------------------------------------ constructors
    @classmethod
    def constant(cls, torque_nm: float, speed_rad_s: float, duration_s: float) -> DutyCycle:
        """A flat duty cycle holding ``torque_nm`` at ``speed_rad_s`` for ``duration_s``."""
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        return cls(
            time_s=np.array([0.0, duration_s]),
            torque_nm=np.array([torque_nm, torque_nm]),
            speed_rad_s=np.array([speed_rad_s, speed_rad_s]),
        )

    @classmethod
    def from_segments(cls, segments: Sequence[tuple[float, float, float]]) -> DutyCycle:
        """Build a piecewise-constant duty cycle from ``(duration_s, torque_nm, speed_rad_s)``.

        Each segment is held constant for its duration. Segment boundaries are emitted as two
        samples (start and end of the segment) so the trapezoidal integral reproduces the
        piecewise-constant demand exactly; coincident boundary times are nudged to stay strictly
        increasing.
        """
        if not segments:
            raise ValueError("at least one segment is required")
        times: list[float] = []
        torques: list[float] = []
        speeds: list[float] = []
        t = 0.0
        for i, (dur, tau, omega) in enumerate(segments):
            if dur <= 0.0:
                raise ValueError(f"segment {i} duration must be positive, got {dur}")
            times.append(t)
            torques.append(tau)
            speeds.append(omega)
            t += dur
            times.append(t)
            torques.append(tau)
            speeds.append(omega)
        return cls(
            time_s=_make_strictly_increasing(np.asarray(times, dtype=float)),
            torque_nm=np.asarray(torques, dtype=float),
            speed_rad_s=np.asarray(speeds, dtype=float),
        )

    # ------------------------------------------------------------------ reductions
    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def peak_torque_nm(self) -> float:
        return float(np.max(np.abs(self.torque_nm)))

    @property
    def max_speed_rad_s(self) -> float:
        return float(np.max(np.abs(self.speed_rad_s)))

    @property
    def rms_torque_nm(self) -> float:
        """Time-RMS torque: sqrt( integral(T^2 dt) / duration ). Thermal-equivalent continuous."""
        mean_sq = trapezoid(self.torque_nm**2, self.time_s) / self.duration_s
        return float(np.sqrt(mean_sq))

    @property
    def mechanical_energy_j(self) -> float:
        """Output mechanical energy demand: integral of T*omega dt [J]."""
        return trapezoid(self.torque_nm * self.speed_rad_s, self.time_s)


def _make_strictly_increasing(t: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Nudge equal/decreasing successive times up by ``eps`` so they are strictly increasing."""
    out = t.copy()
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out
