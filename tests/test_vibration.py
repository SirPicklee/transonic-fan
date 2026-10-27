"""Tests for the blade vibration model against exact solutions."""

import math

import numpy as np
import pytest

from fan.vibration import TI64, BladeStructure, section_properties, rotor67_structure

L, A, I, J, IP = 0.15, 4e-4, 1e-9, 3e-9, 2e-7


def uniform_beam(n=41):
    r = np.linspace(0.0, L, n)
    one = np.ones(n)
    return BladeStructure(r, A * one, I * one, 10 * I * one, J * one, IP * one, TI64)


def test_rectangle_section_properties():
    b, h = 0.08, 0.004
    x = np.array([-b / 2, b / 2, b / 2, -b / 2])
    y = np.array([-h / 2, -h / 2, h / 2, h / 2])
    a, i_min, i_max, i_p = section_properties(x, y)
    assert a == pytest.approx(b * h)
    assert i_min == pytest.approx(b * h ** 3 / 12)
    assert i_max == pytest.approx(h * b ** 3 / 12)
    assert i_p == pytest.approx(b * h ** 3 / 12 + h * b ** 3 / 12)


def test_first_bending_frequency_of_a_uniform_cantilever():
    exact = 1.87510 ** 2 / (2 * math.pi * L ** 2) * math.sqrt(
        TI64["E"] * I / (TI64["rho"] * A))
    assert uniform_beam().frequencies(0.0)["1F"] == pytest.approx(exact, rel=1e-3)


def test_first_torsion_frequency_of_a_uniform_cantilever():
    exact = 1.0 / (4 * L) * math.sqrt(TI64["G"] * J / (TI64["rho"] * IP))
    assert uniform_beam().frequencies(0.0)["1T"] == pytest.approx(exact, rel=1e-3)


def test_southwell_coefficient_of_a_rotating_uniform_cantilever():
    """f^2 = f0^2 + S (rpm/60)^2 with S = 1.1734 for the first flap mode
    (zero hub radius)."""
    beam, rpm = uniform_beam(), 20000.0
    f0, f = beam.frequencies(0.0)["1F"], beam.frequencies(rpm)["1F"]
    assert (f ** 2 - f0 ** 2) / (rpm / 60.0) ** 2 == pytest.approx(1.1734, rel=0.02)


def test_rotor67_blade_modes_are_ordered_and_stiffen_with_speed():
    blade, *_ = rotor67_structure(n_stations=11)
    f0, fd = blade.frequencies(0.0), blade.frequencies(16043.0)
    assert f0["1F"] < f0["2F"]
    assert f0["1F"] < f0["1E"]
    assert fd["1F"] > f0["1F"]