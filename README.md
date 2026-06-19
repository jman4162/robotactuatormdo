# robotactuatormdo

Topology-neutral robot-actuator architecture trade studies. `robotactuatormdo` evaluates full
actuator stacks — motor + drive + gearbox + power electronics + thermal path — against a robot
joint's duty cycle, and reports which architecture class wins on each axis (mass, torque,
reflected inertia, efficiency, cost). It uses [`axfluxmdo`](https://github.com/jman4162/axfluxmdo)
as one of several interchangeable motor backends.

The models are first-order and analytical, sized for early architecture screening rather than
detailed design or FEA-grade accuracy.

## How it fits together

Every motor and every wrapper implements one `MotorModel` protocol
(`evaluate_operating_point`, `mass_properties`, `torque_speed_envelope`, `efficiency_map`), so the
stack composes:

```
PowerStage( Actuator( RadialPMSM + ThermalNetwork, Gearbox ), Inverter, Battery, Cable )
```

Each layer is itself a `MotorModel` expressed at the next shaft out, so the duty-cycle reducer and
the trade-study layer operate on any combination without special cases.

- **Motors:** radial PMSM (sinusoidal FOC), radial BLDC (six-step), a commercial-catalog baseline
  from datasheet scalars, and an axial adapter over `axfluxmdo`.
- **Actuator:** a `Gearbox` (planetary / harmonic / cycloidal / belt / direct-drive presets) with
  efficiency, reflected inertia, and backdrive torque.
- **Power electronics:** inverter conduction + switching loss, battery internal resistance, and
  cable drop, solved for the sagged DC-link voltage.
- **Thermal:** a multi-node lumped network with steady-state and transient (duty-cycle) solvers.
- **Studies:** non-dominated (Pareto) sorting, Monte-Carlo robust scoring, and per-joint topology
  comparison.

Topology, rotor/stator architecture, and drive/control are independent axes. Radial and axial are
flux topologies; six-step BLDC and sinusoidal PMSM/FOC are drive styles that can run on the same
radial machine. The framework keeps them separate so a comparison attributes an advantage to the
right cause. A shared air-gap shear-stress metric (`σ_t = T / ∫ r dA`) keeps radial-vs-axial
comparisons honest about packaging (radius, stack length) versus electromagnetics.

## Example: compare architecture classes for one joint

```python
from robotactuatormdo import (
    Actuator, DutyCycle, FactoryCandidate, JointRequirement, RadialPMGeometry,
    RadialPMSM, compare_topologies, size_radial_pm,
)
from robotactuatormdo.actuators.gearbox import planetary
from robotactuatormdo.materials.copper import COPPER
from robotactuatormdo.materials.electrical_steel import M250_35A
from robotactuatormdo.materials.magnets import NDFEB_N42

geom = RadialPMGeometry(
    air_gap_radius_m=0.045, stack_length_m=0.040, outer_radius_m=0.070,
    pole_pairs=7, slots=24, magnet_thickness_m=0.004, air_gap_m=0.0008, turns_per_phase=40,
)
def motor(_):
    return RadialPMSM(size_radial_pm(geom, NDFEB_N42, M250_35A, COPPER))

req = JointRequirement(
    peak_torque_nm=24.0, continuous_rms_torque_nm=8.0, max_speed_rad_s=25.0,
    bus_voltage_v=48.0, max_phase_current_a_rms=60.0, envelope_outer_diameter_m=0.11,
    envelope_axial_length_m=0.05, max_mass_kg=5.0,
    duty_cycle=DutyCycle.from_segments([(0.4, 18.0, 10.0), (0.6, 5.0, 22.0)]),
)
candidates = [
    FactoryCandidate("direct", "radial_direct_drive", motor),
    FactoryCandidate("qdd", "radial_planetary_qdd", lambda p: Actuator(motor(p), planetary(5.0))),
]
comp = compare_topologies(req, candidates)
print(comp.feasible_classes)   # which classes meet the requirement
print(comp.winners)            # best class per objective axis
```

Runnable benchmark scripts live in `examples/` (humanoid knee, quadruped hip, cobot elbow, and a
Pareto sweep).

## Install

```bash
pip install robotactuatormdo                 # core (radial + system layers)
pip install "robotactuatormdo[axial]"        # adds the axfluxmdo axial backend
pip install "robotactuatormdo[dev]"          # tests + lint
```

From a checkout:

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

## Status

| Layer | State |
| --- | --- |
| Schema + duty-cycle reducer (`MissionResult`) | done |
| Radial PMSM motor + first-order sizing | done |
| Gearbox / Actuator / QDD | done |
| Inverter / battery / cable (`PowerStage`) | done |
| Multi-node thermal network (steady + transient) | done |
| Studies: Pareto / robust / topology comparison | done |
| Backends: BLDC, commercial catalog, axial adapter | done |
| Materials, winding process, BOM cost | done |

## First-order limitations

- Analytical first-order models for screening; no FEA. Sizing carries known constant-factor error,
  so tests assert scaling laws rather than absolute torque.
- Magnet-eddy and mechanical (windage/bearing) losses are not modeled (0 W).
- FOC field weakening neglects stator resistance in the d-axis current solve.
- The thermal network is linear and conduction-only (no radiation); losses are quasi-static over a
  duty cycle (winding temperature integrates in time, but per-step loss feedback is not modeled).
- BOM cost uses order-of-magnitude material prices and a processing fraction, for ranking only.
- The axial adapter's `axfluxmdo` attribute names are unverified against the installed package and
  are isolated in one lookup table to verify later.

## License

MIT — see [LICENSE](LICENSE).
