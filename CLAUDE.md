# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repository is **pre-implementation**. It contains only two design briefs (research Q&A
with cited sources) for a package called `robotactuatormdo` that has not yet been built. There is
no `pyproject.toml`/`setup.py`, no source tree, no tests, and no build tooling. When asked to
"build", "test", or "lint", there is nothing to run — the first task is scaffolding the package,
not discovering existing commands.

The two briefs are the canonical source of intent; read them before making structural decisions.
The `-GITIGNORE` suffix signals they are meant to be kept local / git-ignored, not used as the README.

- `BACKGROUND-GITIGNORE.md` — the *why* and the *packaging decision*: scope, three-level metrics, separate-package rationale, target layout, phased roadmap, first benchmark study.
- `TOPOLOGY-TRADE-STUDY-BACKGROUND-GITIGNORE.md` — the *theory layer*: governing equations, the radial-vs-axial fair-comparison framing, materials database, loss/thermal/mechanical models, and a more detailed module layout (this one supersedes the layout in the first brief).

## What this project is

`robotactuatormdo` is intended as a **topology-neutral, system-level robot actuator architecture
trade-study and MDO framework** — NOT another motor optimizer. It compares full actuator
architectures (motor topology + drive/control + inverter/bus + gearbox + thermal path + joint
packaging) under a specific robot duty cycle, and shows *when* a topology wins rather than
advocating for one.

Framing principle: do not ask "is axial flux better than radial BLDC?" Ask "which motor + control
+ inverter + voltage + gearbox + cooling + packaging architecture gives the best Pareto front for
a given robot joint's duty cycle?" The optimizer should return **architecture classes** (e.g.
radial outrunner + low-ratio planetary; axial double-gap/YASA + low-ratio gearbox; commercial
baseline), each with binding constraints — never a single scalar "winner."

## Architectural decisions already made

1. **Separate package depending on `axfluxmdo`** (external: github.com/jman4162/axfluxmdo) — do
   NOT extend `axfluxmdo` in place. `axfluxmdo` stays the focused axial-flux PM motor physics +
   MDO engine and becomes *one motor backend among several*. This keeps the trade-study tool
   credibly topology-neutral and lets the fast-churning system-level APIs evolve without
   destabilizing the motor package.

2. **Separate topology from drive/control — this is a deliberate anti-category-error.** "BLDC" is
   a drive/control style (trapezoidal back-EMF, six-step commutation), NOT a flux topology.
   Model these as independent axes:
   - Flux topology: radial vs axial
   - Rotor/stator architecture: inrunner, outrunner, single-rotor axial, double-rotor axial, YASA/yokeless
   - Drive/control: six-step BLDC, sinusoidal PMSM/FOC, field weakening
   A radial PM motor can be driven BLDC *or* FOC; high-bandwidth robotics usually favors FOC.
   Radial BLDC/PMSM analytical models are a **first-class comparator**, not a side note.

## Core modeling contracts to honor

- **Four shared model layers**: (1) common PM machine equations → (2) topology-specific geometry →
  (3) drive/control equations → (4) actuator-system equations. The point is to attribute any
  apparent advantage to its real cause (diameter, gear ratio, pole count, bus voltage, cooling,
  current limit) rather than to "the topology."
- **Energy consistency**: torque and back-EMF must derive from the *same* flux-linkage model
  (mirrors `axfluxmdo`'s approach).
- **Shear-stress fair comparison is central.** Compare σ_t = T / ∫r dA across topologies
  (radial ≈ T/(2π r_g² L_stk); axial ≈ 3T/(2π(r_o³−r_i³))) so a packaging advantage (bigger
  radius, longer stack) is never mistaken for better electromagnetics.
- **Units discipline for motor constants**: store SI *phase* quantities internally; convert to
  datasheet conventions (phase/line-line, peak/RMS, Ke/Kt variants) only at the API boundary.
  Mixing line-line-RMS back-EMF with phase-current torque constants is a common comparison bug.
- **Optimize torque under constraints, never torque alone**: tooth/yoke B ≤ B_sat, J_cu ≤ J_max,
  T_winding ≤ insulation limit, T_magnet ≤ demag limit, plus voltage/current margins
  (√(v_d²+v_q²) ≤ V_phase,max, √(i_d²+i_q²) ≤ I_phase,max).
- **Metrics at three levels** — motor-only (EM screening), actuator-level (the layer that matters
  most: reflected inertia G²J_m, backdrive torque, torque bandwidth, backlash/stiffness),
  mission-level (energy/trajectory, RMS/peak current, thermal recovery over repeated cycles,
  distal mass penalty). Compare architectures at actuator and mission level, not motor-only.
- **Robust, not nominal**: evaluate under uncertainty (magnet Br/temp coeff, copper fill, air-gap
  stack-up, thermal contact, gearbox efficiency, duty cycle); produce robust Pareto fronts
  (e.g. 95th-percentile feasible actuator).
- **Materials are datasheet-backed records with uncertainty ranges**, not hard-coded "universal"
  constants (magnets, electrical steel, SMC, copper, structural/thermal).

## Intended package layout (target, not yet built)

```
robotactuatormdo/
  requirements/  joint.py, duty_cycle.py, robot_task.py     # JointRequirement: torque/speed/duty/envelope/voltage/cooling/mass
  motors/        protocols.py, radial_bldc.py, radial_pmsm.py, axial_adapter_axfluxmdo.py, commercial_catalog.py
  controls/      trapezoidal_bldc.py, sinusoidal_foc.py, field_weakening.py
  geometry/      radial_flux.py, axial_flux.py, shear_stress.py
  materials/     magnets.py, electrical_steel.py, smc.py, copper.py, insulation.py, structural.py
  losses/        copper.py, core.py, magnet_eddy.py, inverter.py, mechanical.py
  thermal/       lumped_network.py, duty_cycle.py
  actuators/     gearbox.py, qdd.py, reflected_inertia.py
  studies/       pareto.py, robust.py, compare_topologies.py
  examples/      humanoid_knee_trade.py, quadruped_hip_trade.py, cobot_elbow_trade.py
```

Shared motor protocol all backends implement (axial adapter calls `axfluxmdo`; radial models
start analytical, later FEA-calibrated):

```python
class MotorModel:
    def evaluate_operating_point(self, torque_nm, speed_rad_s, bus_voltage_v, ambient_temp_c) -> MotorOperatingResult: ...
    def mass_properties(self) -> MassProperties: ...
    def torque_speed_envelope(self) -> TorqueSpeedEnvelope: ...
    def efficiency_map(self) -> EfficiencyMap: ...
```

## Conventions

- Python package. Suggested runtime deps when scaffolding: `axfluxmdo`, `numpy`, `scipy`,
  `pandas`, with `pymoo`/`openmdao` optional (mirrors `axfluxmdo`'s optimization stack).
- First integration target (per the briefs): compare architecture classes across three joints
  (manipulator elbow, quadruped knee, humanoid hip/shoulder) over defined torque-speed-time duty
  cycles — radial outrunner + internal planetary QDD, radial inrunner + external
  planetary/cycloidal, double-rotor axial + low-ratio gearbox, coreless/PCB or yokeless axial
  direct/QDD, plus a commercial-motor baseline.

## Source rigor

Both briefs are research-backed with inline citations (Horizon Technology, TI/NXP/MathWorks on
BLDC vs PMSM control, MIT/Patterson on fair axial-vs-radial comparison, OSTI/ResearchGate AFPM
reviews, Arnold Magnetics / Höganäs material notes, arXiv QDD actuator papers). When extending the
design or making engineering claims, maintain that standard — cite reputable sources rather than
asserting motor-design tradeoffs from memory.
```
