"""Helpers for the radial-BLDC vs axial-flux case study notebook.

Keeps the notebook cells thin: motor builders (real axfluxmdo with a labeled literature fallback),
a consistent matplotlib style, and a figure saver. All models are first-order; see the notebook
prose for assumptions and sources.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from robotactuatormdo import RadialPMGeometry, size_radial_pm
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42
from robotactuatormdo.motors.axial_adapter_axfluxmdo import AxFluxMDOAdapter
from robotactuatormdo.motors.radial_bldc import RadialBLDC, RadialBLDCParameters

RADIAL_COLOR = "#1f77b4"
AXIAL_COLOR = "#d62728"
NEUTRAL = "#444444"

_FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.constrained_layout.use": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 11,
        "axes.prop_cycle": plt.cycler(color=[RADIAL_COLOR, AXIAL_COLOR, "#2ca02c", "#9467bd"]),
    })


def save_fig(fig, name: str) -> None:
    os.makedirs(_FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(_FIG_DIR, name), bbox_inches="tight")


# --- radial BLDC (first-order sizing of our own model) -------------------------------------------

RADIAL_GEOM = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008, turns_per_phase=40,
)


def build_radial_bldc(bus_voltage_v: float = 48.0):
    """Return (RadialBLDC, RadialPMParameters) sized from RADIAL_GEOM."""
    params = size_radial_pm(RADIAL_GEOM, NDFEB_N42, M250_35A, COPPER,
                            rated_bus_voltage_v=bus_voltage_v)
    return RadialBLDC(RadialBLDCParameters(params)), params


# --- axial flux (real axfluxmdo, literature-scalar fallback) -------------------------------------

# Literature anchors for the fallback (order-of-magnitude, robotics joint scale):
#   axial-flux PM ~15-25 N*m/kg peak, 92-95% efficiency (YASA-class, IEEE Spectrum;
#   PatSnap humanoid-joint synthesis). Sources are cited in the notebook prose.
_LIT_AXIAL = dict(
    kt=0.42, ke=0.42, r_phase=0.04, mass_kg=1.2, rotor_inertia_kg_m2=8.0e-4,
    max_phase_current_a_rms=60.0, max_speed_rad_s=250.0, rated_bus_voltage_v=48.0,
)


class _LiteratureAxial:
    """Duck-typed axial design from published scalars (clearly-labeled fallback)."""

    def __init__(self, peak_efficiency: float = 0.93, **kw):
        self.__dict__.update(kw)
        self._eff = peak_efficiency

    def efficiency_at(self, torque_nm: float, speed_rad_s: float) -> float:
        return self._eff


def build_axial(bus_voltage_v: float = 48.0, outer_radius_m: float = 0.060,
                inner_radius_m: float = 0.036, pole_pairs: int = 20, turns_per_phase: int = 40):
    """Return (AxFluxMDOAdapter, used_real_axfluxmdo: bool).

    Tries real axfluxmdo (sizing an AxialFluxMotor and deriving the max phase current from its
    current-density limit). Falls back to labeled literature scalars if axfluxmdo is unavailable.
    """
    try:
        from axfluxmdo import AxialFluxMotor, OperatingPoint
        from axfluxmdo.models import AnalyticalModel

        model = AnalyticalModel()
        motor = AxialFluxMotor(
            outer_radius=outer_radius_m, inner_radius=inner_radius_m, air_gap=0.001,
            pole_pairs=pole_pairs, turns_per_phase=turns_per_phase,
        )
        probe_i = 30.0
        probe_op = OperatingPoint(speed_rpm=400, current_rms=probe_i, dc_bus_voltage=bus_voltage_v)
        probe = model.evaluate(motor, probe_op)
        j = getattr(probe, "current_density_a_mm2", None)
        max_i = (10.0 * probe_i / j) if j else 60.0  # axfluxmdo current-density limit ~10 A/mm^2
        adapter = AxFluxMDOAdapter.from_axfluxmdo(
            motor, model, probe_op,
            max_phase_current_a_rms=max_i, max_speed_rad_s=250.0,
            rated_bus_voltage_v=bus_voltage_v,
        )
        return adapter, True
    except Exception:  # pragma: no cover - exercised only when axfluxmdo is absent/incompatible
        lit = dict(_LIT_AXIAL)
        lit["rated_bus_voltage_v"] = bus_voltage_v
        return AxFluxMDOAdapter(_LiteratureAxial(**lit)), False
