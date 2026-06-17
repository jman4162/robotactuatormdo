"""Inverter conduction + switching loss (MOSFET/GaN/SiC)."""

from __future__ import annotations


def inverter_loss(current_a_rms, bus_voltage_v, params):
    raise NotImplementedError("planned: inverter loss model")
