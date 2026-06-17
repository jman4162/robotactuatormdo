# robotactuatormdo

**Topology-neutral robot actuator architecture trade-study and MDO framework.**

`robotactuatormdo` compares *full actuator architectures* — motor topology + drive/control +
inverter/bus + gearbox + thermal path + joint packaging — under a specific robot duty cycle, and
shows **when** a topology wins rather than advocating for one. It uses
[`axfluxmdo`](https://github.com/jman4162/axfluxmdo) as one of several interchangeable motor
backends.

> Status: **pre-alpha**. The package layout and the stable contracts (the `MotorModel` protocol,
> result dataclasses, and the shear-stress fair-comparison equations) are in place. Most physics
> modules are typed stubs that raise `NotImplementedError` and are being filled in.

## Design principles

- **Separate topology from control.** "BLDC" is a drive/control style (trapezoidal back-EMF,
  six-step commutation), *not* a flux topology. Flux topology (radial vs axial), rotor/stator
  architecture, and drive/control (six-step BLDC vs sinusoidal PMSM/FOC) are independent axes.
- **Fair comparison via air-gap shear stress.** Compare `σ_t = T / ∫ r dA` across topologies so a
  packaging advantage (larger radius, longer stack) is never mistaken for better electromagnetics.
- **Optimize torque under constraints, never torque alone** (saturation, current density, winding
  and magnet temperature, voltage/current margins).
- **Report at three levels:** motor-only, actuator-level, and robot-mission-level.
- **Robust, not nominal** — evaluate under manufacturing and duty-cycle uncertainty.

## Install

```bash
# core (radial + system layers)
pip install robotactuatormdo

# with the axial-flux backend (pulls in axfluxmdo)
pip install "robotactuatormdo[axial]"

# development
pip install "robotactuatormdo[dev]"
```

Or with `uv` from a checkout:

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
