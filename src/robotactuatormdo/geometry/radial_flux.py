"""First-order radial-flux PMSM sizing (TOPOLOGY brief §3/§6).

Maps a physical :class:`RadialPMGeometry` (an inrunner surface-PM machine) plus material records
to the lumped :class:`RadialPMParameters` the operating-point model consumes. Every relation is a
documented first-order textbook form (Hanselman/Lipo style); the goal is *credible architecture
screening*, not datasheet accuracy. Constant-factor error (winding factor, pole-arc, leakage) is
expected and is why the tests assert scaling laws rather than absolute torque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robotactuatormdo.materials.copper import Conductor
from robotactuatormdo.materials.electrical_steel import ElectricalSteel
from robotactuatormdo.materials.magnets import MagnetMaterial
from robotactuatormdo.motors.radial_pmsm import RadialPMParameters

__all__ = ["RadialPMGeometry", "size_radial_pm"]

_MU0 = 4.0e-7 * math.pi


_B_TOOTH_TARGET_T = 1.6
_B_YOKE_TARGET_T = 1.5


@dataclass(frozen=True, slots=True)
class RadialPMGeometry:
    """Geometry of an inrunner surface-PM radial machine. SI (metres). ``pole_arc_fraction`` and
    ``slot_fill`` are in [0, 1]; ``turns_per_phase`` is series turns.

    ``tooth_width_m`` and ``yoke_thickness_m`` default to ``None``, in which case sizing derives
    them from flux continuity (so the iron is not accidentally driven into saturation); pass
    explicit values to override.
    """

    air_gap_radius_m: float
    stack_length_m: float
    outer_radius_m: float
    pole_pairs: int
    slots: int
    magnet_thickness_m: float
    air_gap_m: float
    turns_per_phase: int
    slot_fill: float = 0.45
    pole_arc_fraction: float = 0.85
    winding_factor: float = 0.95
    tooth_width_m: float | None = None
    yoke_thickness_m: float | None = None
    rotor_back_iron_thickness_m: float = 0.005


def size_radial_pm(
    geom: RadialPMGeometry,
    magnet: MagnetMaterial,
    steel: ElectricalSteel,
    copper: Conductor,
    *,
    rated_bus_voltage_v: float = 48.0,
    max_current_density_a_per_mm2: float = 6.0,
    winding_temp_limit_c: float = 155.0,
    h_conv_w_m2k: float = 25.0,
) -> RadialPMParameters:
    """Derive lumped PMSM parameters from geometry + materials (first-order)."""
    r_g = geom.air_gap_radius_m
    l_stk = geom.stack_length_m
    p = geom.pole_pairs
    l_m = geom.magnet_thickness_m
    air_gap = geom.air_gap_m
    n_ph = geom.turns_per_phase
    k_w = geom.winding_factor
    alpha_p = geom.pole_arc_fraction

    g_eff = air_gap + l_m / magnet.recoil_mu_r
    tau_p = math.pi * r_g / p  # pole pitch arc length at the bore

    # --- magnetic loading ---
    b_g = magnet.br_t * l_m / (l_m + magnet.recoil_mu_r * air_gap)
    b_g1 = (4.0 / math.pi) * b_g * math.sin(alpha_p * math.pi / 2.0)  # fundamental peak
    phi_p = 2.0 * b_g1 * r_g * l_stk / p  # fundamental flux per pole
    psi_m = k_w * n_ph * phi_p  # phase-peak PM flux linkage

    # Teeth/yoke: use given dimensions, else size from flux continuity at a target density so the
    # iron is not accidentally saturated.
    slot_pitch = 2.0 * math.pi * r_g / geom.slots
    tooth_width = geom.tooth_width_m or (b_g * slot_pitch / _B_TOOTH_TARGET_T)
    yoke_thickness = geom.yoke_thickness_m or (b_g * alpha_p * tau_p / (2.0 * _B_YOKE_TARGET_T))

    # --- winding resistance & current capability ---
    d_slot = max(geom.outer_radius_m - r_g - yoke_thickness, 1e-4)
    band_area = math.pi * ((r_g + d_slot) ** 2 - r_g**2)
    tooth_area = geom.slots * tooth_width * d_slot
    slot_area = max(band_area - tooth_area, 1e-9)
    a_cu_total = geom.slot_fill * slot_area
    a_cond = (a_cu_total / 3.0) / (2.0 * n_ph)  # one conductor cross-section
    l_end = tau_p
    r_s_20 = copper.resistivity_20_ohm_m * (n_ph * 2.0 * (l_stk + l_end)) / a_cond
    j_max = max_current_density_a_per_mm2 * 1e6  # A/m^2
    i_max_rms = j_max * a_cond

    # --- synchronous inductance (magnetizing + 10% leakage), surface-PM L_d=L_q ---
    l_md = 1.5 * (4.0 / math.pi) * _MU0 * (k_w * n_ph) ** 2 * (r_g * l_stk) / (p**2 * g_eff)
    l_s = 1.1 * l_md

    # --- flux densities for saturation flags / core loss ---
    b_tooth = b_g * slot_pitch / tooth_width
    b_yoke = b_g * alpha_p * tau_p / (2.0 * yoke_thickness)

    # --- masses ---
    tooth_vol = geom.slots * tooth_width * d_slot * l_stk
    yoke_vol = (
        math.pi
        * (geom.outer_radius_m**2 - (geom.outer_radius_m - yoke_thickness) ** 2)
        * l_stk
    )
    stator_iron_mass = (tooth_vol + yoke_vol) * steel.density_kg_m3

    r_mag = r_g - air_gap - l_m / 2.0
    magnet_vol = alpha_p * 2.0 * math.pi * r_mag * l_m * l_stk
    magnet_mass = magnet_vol * magnet.density_kg_m3

    rotor_iron_outer = r_g - air_gap - l_m
    rotor_iron_inner = max(rotor_iron_outer - geom.rotor_back_iron_thickness_m, 1e-4)
    rotor_iron_vol = math.pi * (rotor_iron_outer**2 - rotor_iron_inner**2) * l_stk
    rotor_iron_mass = rotor_iron_vol * steel.density_kg_m3

    copper_vol = a_cu_total * (l_stk + 2.0 * l_end)
    copper_mass = copper_vol * copper.density_kg_m3

    total_mass = stator_iron_mass + rotor_iron_mass + magnet_mass + copper_mass

    # rotor inertia (annulus of magnets + rotor iron)
    r_rot_o = r_g - air_gap
    m_rot = magnet_mass + rotor_iron_mass
    rotor_inertia = 0.5 * m_rot * (r_rot_o**2 + rotor_iron_inner**2)

    # --- thermal (crude size-scaled single-node) ---
    a_surf = 2.0 * math.pi * geom.outer_radius_m * l_stk + 2.0 * math.pi * geom.outer_radius_m**2
    r_th_wa = 1.0 / (h_conv_w_m2k * a_surf)
    r_th_ma = 2.0 * r_th_wa  # rotor magnets harder to cool

    return RadialPMParameters(
        pole_pairs=p,
        flux_linkage_wb=psi_m,
        r_s_ohm_20=r_s_20,
        l_d_h=l_s,
        l_q_h=l_s,
        max_phase_current_a_rms=i_max_rms,
        rated_bus_voltage_v=rated_bus_voltage_v,
        iron_mass_kg=stator_iron_mass,
        core_b_peak_t=b_tooth,
        k_hyst=steel.k_hyst,
        steinmetz_alpha=steel.alpha,
        k_eddy=steel.k_eddy,
        copper_temp_coeff_per_c=copper.temp_coeff_per_c,
        r_th_winding_ambient_c_w=r_th_wa,
        r_th_magnet_ambient_c_w=r_th_ma,
        winding_temp_limit_c=winding_temp_limit_c,
        magnet_temp_limit_c=magnet.max_op_temp_c,
        b_tooth_t=b_tooth,
        b_yoke_t=b_yoke,
        b_sat_t=steel.b_sat_t,
        demag_id_limit_a_pk=1.5 * i_max_rms * math.sqrt(2.0),
        total_mass_kg=total_mass,
        rotor_inertia_kg_m2=rotor_inertia,
        copper_mass_kg=copper_mass,
        magnet_mass_kg=magnet_mass,
    )
