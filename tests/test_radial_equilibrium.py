"""Tests for simple radial equilibrium at the rotor exit."""

import math

import numpy as np
import pytest

from fan.annulus import ROTOR67, Annulus
from fan.radial_equilibrium import design_for_pressure_ratio, solve_exit
from fan.velocity_triangles import CP, R_AIR

INLET = ROTOR67["annulus_inlet"]
EXIT = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
RPM, MDOT = ROTOR67["rpm"], ROTOR67["mass_flow"]
OMEGA = RPM * 2.0 * math.pi / 60.0


@pytest.fixture(scope="module", params=[-1, 0, 1], ids=["free", "constant", "forced"])
def solution(request):
    return design_for_pressure_ratio(INLET, EXIT, RPM, MDOT, 1.63, request.param, 0.92)


def test_free_vortex_has_uniform_axial_velocity_and_work():
    sol = solve_exit(INLET, EXIT, RPM, MDOT, c_theta_m=150.0, n=-1, efficiency=0.92)
    assert np.ptp(sol.cx2) < 1e-6 * sol.cx2.mean()
    assert np.ptp(sol.t02) < 1e-9 * sol.t02.mean()


def test_mass_flow_is_conserved(solution):
    rho_cx = solution.rho_cx2_r / solution.r2
    flow = np.trapezoid(rho_cx * 2.0 * math.pi * solution.r2, solution.r2)
    assert flow == pytest.approx(MDOT, rel=1e-6)


def test_work_follows_euler_equation(solution):
    dh0 = OMEGA * solution.r2 * solution.c_theta2
    assert CP * (solution.t02 - solution.t01) == pytest.approx(dh0, rel=1e-12)


def test_mass_averaged_work_matches_pressure_ratio_target(solution):
    target = CP * 288.15 * (1.63 ** (1 / 3.5) - 1.0) / 0.92
    assert solution.work == pytest.approx(target, rel=1e-5)


def test_radial_equilibrium_equation_is_satisfied(solution):
    """d(Cx^2/2)/dr = dh0/dr - T ds/dr - (C_theta/r) d(r C_theta)/dr, checked
    away from the ends where one-sided differences are less accurate."""
    r, cx, ct = solution.r2, solution.cx2, solution.c_theta2
    t2 = solution.t02 - (cx ** 2 + ct ** 2) / (2.0 * CP)
    s = CP * np.log(solution.t02 / solution.t01) - R_AIR * np.log(solution.pr)
    lhs = np.gradient(0.5 * cx ** 2, r)
    rhs = (np.gradient(CP * solution.t02, r) - t2 * np.gradient(s, r)
           - ct / r * np.gradient(r * ct, r))
    inner = slice(5, -5)
    scale = np.max(np.abs(np.gradient(CP * solution.t02, r))) + 1.0
    assert np.max(np.abs(lhs[inner] - rhs[inner])) < 0.02 * scale


def test_streamlines_map_hub_to_hub_and_tip_to_tip(solution):
    assert solution.r1[0] == pytest.approx(INLET.r_hub)
    assert solution.r1[-1] == pytest.approx(INLET.r_tip)
    assert np.all(np.diff(solution.r1) > 0)


def test_swirl_growing_with_radius_starves_the_hub():
    """Constant swirl and forced vortex push the axial velocity towards the tip."""
    for n in (0, 1):
        sol = design_for_pressure_ratio(INLET, EXIT, RPM, MDOT, 1.63, n, 0.92)
        assert sol.cx2[0] < sol.cx2[-1]
        assert sol.pr[0] < sol.pr[-1]