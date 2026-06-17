"""Small internal numerics shared across the package."""

from __future__ import annotations

import numpy as np

__all__ = ["trapezoid"]


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over abscissa ``x`` as a Python float.

    Wraps ``numpy.trapezoid`` (falling back to the older ``numpy.trapz`` name) so callers get a
    single, deprecation-proof entry point.
    """
    integ = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(integ(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))
