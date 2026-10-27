"""Tests for the rotor loss model."""

import numpy as np
import pytest

from fan.annulus import ROTOR67, Annulus
from fan.losses import (HubStallError, analyze, efficiency_from_loss,
                        equivalent_diffusion_ratio, mass_averaged_efficiency,
                        normal_shock_p0_ratio, profile_loss, rotor67_solidity,
                        shock_loss)
from fan.radial_equilibrium import design_for_pressure_ratio, solve_exit

INLET = ROTOR67["annulus_inlet"]
EXIT = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
RPM, MDOT = ROTOR67["rpm"], ROTOR67["mass_flow"]


@pytest.fixture(scope="module")
def free_vortex():
    sol = solve_exit(INLET, EXIT, RPM, MDOT, c_theta_m=156.5, n=-1, efficiency=0.92)
    return sol


def test_normal_shock_table_values():
    # Gas tables, gamma = 1.4: p02/p01 = 1 at M = 1, 0.9582 at M = 1.4, 0.7209 at M = 2
    assert normal_shock_p0_ratio(np.array([1.0, 1.4, 2.0])) == pytest.approx(
        [1.0, 0.9582, 0.7209], abs=1e-4)


def test_no_shock_loss_when_subsonic():
    assert np.all(shock_loss(np.array([0.3, 0.8, 1.0])) == 0.0)


def test_shock_loss_grows_with_mach():
    m = np.linspace(1.05, 1.6, 12)
    assert np.all(np.diff(shock_loss(m)) > 0)


def test_zero_loss_gives_isentropic_rotor(free_vortex):
    """Rothalpy, Euler work and the isentropic relations close exactly:
    without loss every streamline has unit efficiency, despite the radius change."""
    eta = efficiency_from_loss(free_vortex, np.zeros_like(free_vortex.r2))
    assert eta == pytest.approx(1.0, abs=1e-10)


def test_loss_lowers_efficiency(free_vortex):
    w = np.full_like(free_vortex.r2, 0.05)
    assert np.all(efficiency_from_loss(free_vortex, w) < 1.0)
    assert np.all(efficiency_from_loss(free_vortex, 2 * w) < efficiency_from_loss(free_vortex, w))


def test_equivalent_diffusion_ratio_exceeds_lieblein_floor(free_vortex):
    sigma = rotor67_solidity(free_vortex.span)
    assert np.all(equivalent_diffusion_ratio(free_vortex, sigma)
                  > 1.12 * free_vortex.w1 / free_vortex.w2)


def test_more_solidity_lowers_diffusion_but_adds_wetted_surface(free_vortex):
    low, high = np.full(101, 1.3), np.full(101, 2.5)
    assert np.all(equivalent_diffusion_ratio(free_vortex, high)
                  < equivalent_diffusion_ratio(free_vortex, low))
    assert profile_loss(free_vortex, high).shape == free_vortex.r2.shape


def test_free_vortex_rotor67_prediction():
    guess = design_for_pressure_ratio(INLET, EXIT, RPM, MDOT, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (EXIT.r_mean / guess.r2[0]) ** -1
    sol, eta, w_p, w_s = analyze(INLET, EXIT, RPM, MDOT, c_theta_m, -1)
    assert 0.90 < mass_averaged_efficiency(sol) < 0.97
    assert w_s[0] == 0.0 and w_s[-1] > 0.05        # shock only in the outer span
    assert eta[-1] < eta[len(eta) // 2]            # the tip pays for the shock


def test_constant_swirl_stalls_the_hub():
    guess = design_for_pressure_ratio(INLET, EXIT, RPM, MDOT, 1.63, 0, 0.92)
    with pytest.raises(HubStallError):
        analyze(INLET, EXIT, RPM, MDOT, guess.c_theta2[0], 0)