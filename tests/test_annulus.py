"""Tests for annulus sizing and the NASA Rotor 67 inlet check."""

import math

import pytest

from fan.annulus import (Q_CHOKE, ROTOR67, Annulus, axial_mach, axial_velocity,
                         flow_function, inlet_triangle, mach_from_flow_function,
                         size_annulus)
from fan.velocity_triangles import GAMMA, R_AIR


def test_flow_function_known_choke_value():
    # gamma = 1.4: Q(1) = sqrt(1.4) * 1.2^-3 = 0.6847
    assert Q_CHOKE == pytest.approx(0.6847, abs=1e-4)
    assert flow_function(0.0) == 0.0


def test_flow_function_peaks_at_sonic():
    assert flow_function(0.9) < Q_CHOKE
    assert flow_function(1.1) < Q_CHOKE


@pytest.mark.parametrize("mach", [0.1, 0.4, 0.6, 0.9])
def test_mach_round_trip(mach):
    assert mach_from_flow_function(flow_function(mach)) == pytest.approx(mach, rel=1e-8)


def test_choked_annulus_is_rejected():
    tiny = Annulus(r_tip=0.1, r_hub=0.05)
    with pytest.raises(ValueError):
        axial_mach(tiny, mass_flow=33.25)


def test_mean_radius_splits_area_in_two():
    ann = Annulus(r_tip=0.25, r_hub=0.10)
    inner = math.pi * (ann.r_mean ** 2 - ann.r_hub ** 2)
    assert inner == pytest.approx(ann.area / 2.0)


def test_axial_velocity_gives_back_its_mach_number():
    t0, mach = 288.15, 0.55
    v = axial_velocity(mach, t0)
    t = t0 - v ** 2 / (2.0 * GAMMA * R_AIR / (GAMMA - 1.0))
    assert v / math.sqrt(GAMMA * R_AIR * t) == pytest.approx(mach)


def test_sizing_is_the_inverse_of_axial_mach():
    ann = size_annulus(mass_flow=50.0, axial_mach=0.55, hub_tip_ratio=0.35)
    assert ann.hub_tip_ratio == pytest.approx(0.35)
    assert axial_mach(ann, 50.0) == pytest.approx(0.55, rel=1e-8)


def test_more_flow_needs_a_bigger_fan():
    small = size_annulus(30.0, 0.55, 0.35)
    big = size_annulus(60.0, 0.55, 0.35)
    assert big.r_tip == pytest.approx(small.r_tip * math.sqrt(2.0))


def test_rotor67_geometry():
    ann = ROTOR67["annulus_inlet"]
    assert ann.hub_tip_ratio == pytest.approx(0.375, abs=0.002)


def test_rotor67_tip_relative_mach_within_three_percent_of_nasa():
    """One-dimensional uniform inflow reproduces the published tip relative Mach."""
    ann = ROTOR67["annulus_inlet"]
    tip = inlet_triangle(ann, ann.r_tip, ROTOR67["rpm"], ROTOR67["mass_flow"])
    assert tip.mach_rel(288.15) == pytest.approx(ROTOR67["tip_relative_mach"], rel=0.03)