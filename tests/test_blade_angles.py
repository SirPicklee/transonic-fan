"""Tests for incidence, deviation and blade metal angles."""

import numpy as np
import pytest

from fan.blade_angles import carter_m, metal_angles


def test_carter_rule_is_satisfied():
    beta1, beta2, sigma = np.array([-60.0]), np.array([-40.0]), np.array([1.5])
    k1, k2, camber, _, dev = metal_angles(beta1, beta2, sigma)
    assert dev == pytest.approx(carter_m(k2) * camber / np.sqrt(sigma), rel=1e-9)


def test_blade_turns_more_than_the_flow():
    beta1, beta2, sigma = np.array([-60.0]), np.array([-40.0]), np.array([1.5])
    _, _, camber, _, dev = metal_angles(beta1, beta2, sigma)
    assert dev > 0
    assert camber > beta2 - beta1


def test_no_turning_needs_no_camber():
    k1, k2, camber, _, dev = metal_angles(np.array([-50.0]), np.array([-50.0]),
                                          np.array([1.5]))
    assert camber == pytest.approx(0.0, abs=1e-12)
    assert dev == pytest.approx(0.0, abs=1e-12)


def test_higher_solidity_guides_the_flow_better():
    args = (np.array([-60.0, -60.0]), np.array([-40.0, -40.0]))
    _, _, _, _, dev = metal_angles(*args, solidity=np.array([1.0, 3.0]))
    assert dev[1] < dev[0]


def test_incidence_shifts_the_inlet_metal_angle():
    k1, *_ = metal_angles(np.array([-60.0]), np.array([-40.0]), np.array([1.5]),
                          incidence=3.0)
    assert k1 == pytest.approx(-57.0)