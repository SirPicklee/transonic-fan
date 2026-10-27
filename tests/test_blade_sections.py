"""Tests for the blade section geometry."""

import math

import numpy as np
import pytest

from fan.blade_sections import chord_from_solidity, circular_arc_section


def test_chord_from_solidity():
    assert chord_from_solidity(1.29, 0.25, 22) == pytest.approx(1.29 * 2 * math.pi * 0.25 / 22)


def test_flat_plate_without_camber():
    x, y = circular_arc_section(0.0, -30.0, 0.1, 0.05)["camber"]
    slope = np.degrees(np.arctan2(np.diff(y), np.diff(x)))
    assert slope == pytest.approx(-30.0)


@pytest.mark.parametrize("camber, stagger", [(20.0, -40.0), (60.0, -10.0), (5.0, -65.0)])
def test_camber_line_matches_metal_angles(camber, stagger):
    x, y = circular_arc_section(camber, stagger, 0.1, 0.05, n_points=2001)["camber"]
    k_le = math.degrees(math.atan2(y[1] - y[0], x[1] - x[0]))
    k_te = math.degrees(math.atan2(y[-1] - y[-2], x[-1] - x[-2]))
    assert k_le == pytest.approx(stagger - camber / 2, abs=0.05)
    assert k_te == pytest.approx(stagger + camber / 2, abs=0.05)


def test_chord_length():
    x, y = circular_arc_section(40.0, -30.0, 0.1, 0.05)["camber"]
    assert math.hypot(x[-1] - x[0], y[-1] - y[0]) == pytest.approx(0.1)


def test_maximum_thickness():
    sec = circular_arc_section(40.0, -30.0, 0.1, 0.05)
    (xu, yu), (xl, yl) = sec["upper"], sec["lower"]
    assert np.hypot(xu - xl, yu - yl).max() == pytest.approx(0.005, rel=1e-3)


def test_section_is_centred_at_mid_chord():
    x, y = circular_arc_section(40.0, -30.0, 0.1, 0.05)["camber"]
    assert 0.5 * (x[0] + x[-1]) == pytest.approx(0.0, abs=1e-12)
    assert 0.5 * (y[0] + y[-1]) == pytest.approx(0.0, abs=1e-12)