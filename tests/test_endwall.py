"""Tests for the hub endwall loss models."""

import numpy as np
import pytest

from fan.endwall import (distribute_from_hub, endwall_dissipation_penalty,
                         howell_secondary_loss)


def test_lift_coefficient_by_hand():
    # tan(-60) = -1.7321, tan(-40) = -0.8391, tan(beta_m) = -1.2856, cos = 0.6140
    # C_L = 2 / 1.5 * 0.6140 * 0.8930 = 0.731
    _, cl = howell_secondary_loss(-60.0, -40.0, 1.5)
    assert cl == pytest.approx(0.731, rel=2e-3)


def test_no_turning_no_secondary_loss():
    zeta, cl = howell_secondary_loss(-50.0, -50.0, 1.5)
    assert cl == 0.0 and zeta == 0.0


def test_secondary_loss_grows_with_the_square_of_the_lift():
    """Same mean angle, twice the turning: C_L doubles, the loss rises about four
    times (cos^2 beta1 changes a little as beta1 moves)."""
    small, cl_s = howell_secondary_loss(-55.0, -45.0, 1.5)
    large, cl_l = howell_secondary_loss(-60.0, -40.0, 1.5)
    assert cl_l > 1.9 * cl_s
    assert large > 3.0 * small


def test_friction_is_proportional_to_the_dissipation_coefficient():
    args = dict(w1=240.0, w2=180.0, d_w_theta=250.0, solidity=3.1, pitch=0.03,
                axial_chord=0.09, rho=1.1, mass_flow_passage=1.5, work=47e3)
    one = endwall_dissipation_penalty(cd=0.002, **args)
    assert one > 0.0
    assert endwall_dissipation_penalty(cd=0.004, **args) == pytest.approx(2.0 * one)


def test_hub_distribution_keeps_the_mass_average():
    span = np.linspace(0.0, 1.0, 101)
    mass = 1.0 + span
    local = distribute_from_hub(span, mass, total=0.01, width=0.3)
    mean = np.trapezoid(local * mass, span) / np.trapezoid(mass, span)
    assert mean == pytest.approx(0.01)
    assert np.all(local[span > 0.3] == 0.0)
    assert local[0] == local.max()