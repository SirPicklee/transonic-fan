"""Tests for the tip clearance loss model."""

import numpy as np
import pytest

from fan.tip_clearance import denton_leakage_loss, distribute, efficiency_penalty

TIP = dict(w1=466.0, w2=329.0, cx2=167.0, d_w_theta=119.0, solidity=1.29, span=0.1435)


def test_no_gap_no_loss():
    assert denton_leakage_loss(gap=0.0, **TIP) == 0.0


def test_unloaded_blade_does_not_leak():
    args = dict(TIP, d_w_theta=0.0)
    assert denton_leakage_loss(gap=0.001, **args) == pytest.approx(0.0, abs=1e-15)


def test_loss_is_proportional_to_gap():
    one = denton_leakage_loss(gap=0.001, **TIP)
    assert denton_leakage_loss(gap=0.002, **TIP) == pytest.approx(2.0 * one)


def test_rotor67_tip_penalty_is_of_the_usual_size():
    """Rule of thumb: roughly 1-2 efficiency points per 1 % of clearance/span."""
    gap = 0.01 * TIP["span"]
    zeta = denton_leakage_loss(gap=gap, **TIP)
    d_eta = efficiency_penalty(zeta, TIP["w2"], work=46.9e3)
    assert 0.005 < d_eta < 0.03


def test_distribution_keeps_the_mass_average():
    span = np.linspace(0.0, 1.0, 101)
    mass = 1.0 + span                       # any positive mass weighting
    local = distribute(span, mass, total=0.01, width=0.3)
    mean = np.trapezoid(local * mass, span) / np.trapezoid(mass, span)
    assert mean == pytest.approx(0.01)
    assert np.all(local[span < 0.7] == 0.0)
    assert local[-1] == local.max()