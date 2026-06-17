"""Air-gap tangential shear stress: the central fair-comparison primitive.

A motor's torque is the integral of tangential shear stress over the active air-gap surface,
weighted by radius. Reducing both topologies to the *same* shear-stress figure prevents the most
common trade-study error: crediting "the topology" for what is really a packaging choice (a
larger radius or a longer stack). All quantities are SI: torque [N·m], radii/length [m], shear
stress [Pa].

Radial flux (cylindrical gap of radius ``r_g`` and stack length ``L_stk``):

    T = sigma_t * r_g * A_g = 2*pi * sigma_t * r_g**2 * L_stk
    sigma_t = T / (2*pi * r_g**2 * L_stk)

    => torque scales with r_g**2 * L_stk

Axial flux (annular gap between ``r_i`` and ``r_o``, integrating r * sigma_t over the annulus):

    T = (2*pi*sigma_t/3) * (r_o**3 - r_i**3)
    sigma_t = 3*T / (2*pi * (r_o**3 - r_i**3))

    => torque scales with (r_o**3 - r_i**3)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "area_radial",
    "area_axial",
    "shear_stress_radial",
    "torque_radial",
    "shear_stress_axial",
    "torque_axial",
]


def area_radial(r_g: float, l_stk: float) -> float:
    """Cylindrical air-gap surface area ``2*pi*r_g*L_stk`` [m²]."""
    _require_positive(r_g=r_g, l_stk=l_stk)
    return 2.0 * np.pi * r_g * l_stk


def area_axial(r_o: float, r_i: float) -> float:
    """Annular air-gap area ``pi*(r_o**2 - r_i**2)`` [m²]."""
    _require_axial_radii(r_o, r_i)
    return np.pi * (r_o**2 - r_i**2)


def shear_stress_radial(torque_nm: float, r_g: float, l_stk: float) -> float:
    """Mean tangential shear stress [Pa] for a radial-flux machine."""
    _require_positive(r_g=r_g, l_stk=l_stk)
    return torque_nm / (2.0 * np.pi * r_g**2 * l_stk)


def torque_radial(sigma_t: float, r_g: float, l_stk: float) -> float:
    """Torque [N·m] from radial shear stress: ``2*pi*sigma_t*r_g**2*L_stk``."""
    _require_positive(r_g=r_g, l_stk=l_stk)
    return 2.0 * np.pi * sigma_t * r_g**2 * l_stk


def shear_stress_axial(torque_nm: float, r_o: float, r_i: float) -> float:
    """Mean tangential shear stress [Pa] for an axial-flux machine."""
    _require_axial_radii(r_o, r_i)
    return 3.0 * torque_nm / (2.0 * np.pi * (r_o**3 - r_i**3))


def torque_axial(sigma_t: float, r_o: float, r_i: float) -> float:
    """Torque [N·m] from axial shear stress: ``(2*pi*sigma_t/3)*(r_o**3 - r_i**3)``."""
    _require_axial_radii(r_o, r_i)
    return (2.0 * np.pi * sigma_t / 3.0) * (r_o**3 - r_i**3)


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")


def _require_axial_radii(r_o: float, r_i: float) -> None:
    if r_i < 0.0:
        raise ValueError(f"r_i must be non-negative, got {r_i}")
    if r_o <= r_i:
        raise ValueError(f"r_o ({r_o}) must be greater than r_i ({r_i})")
