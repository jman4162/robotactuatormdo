"""The core package must import and expose its contract without any optional deps."""

from __future__ import annotations

import robotactuatormdo as r


def test_version_present():
    assert isinstance(r.__version__, str)
    assert r.__version__


def test_public_contract_exported():
    for name in [
        "MotorModel",
        "JointRequirement",
        "MotorOperatingResult",
        "MassProperties",
        "TorqueSpeedEnvelope",
        "EfficiencyMap",
        "FeasibilityFlags",
    ]:
        assert hasattr(r, name), name


def test_axial_adapter_imports_without_axfluxmdo():
    # Importing the module must not require the optional extra; only *using* it should.
    from robotactuatormdo.motors.axial_adapter_axfluxmdo import AxFluxMDOAdapter

    assert AxFluxMDOAdapter is not None
