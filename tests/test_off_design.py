"""Tests for the off-design analysis with the real Rotor 67 blade."""

import math

import numpy as np
import pytest

from fan.annulus import ROTOR67, Annulus
from fan.off_design import analyze_point, blade_on_grid
from fan.radial_equilibrium import solve_exit, solve_exit_swirl
from fan.velocity_triangles import CP

INLET = ROTOR67["annulus_inlet"]
EXIT = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
RPM = ROTOR67["rpm"]


def test_solve_exit_swirl_reproduces_solve_exit():
    a = solve_exit(INLET, EXIT, RPM, 33.25, 156.5, -1, 0.92)
    b = solve_exit_swirl(INLET, EXIT, RPM, 33.25, a.c_theta2, 0.92)
    assert b.cx2 == pytest.approx(a.cx2)
    assert b.pr == pytest.approx(a.pr)


@pytest.fixture(scope="module")
def grid():
    return blade_on_grid()


def test_exit_angle_is_less_turned_than_the_blade(grid):
    assert np.all(grid["deviation"] > 0)
    assert np.all(grid["beta2"] < grid["kappa2"])


@pytest.fixture(scope="module")
def points(grid):
    peak = analyze_point(34.573, grid)
    stall = analyze_point(32.305, grid, peak["incidence"])
    return peak, stall


def test_work_follows_euler_with_the_blade_exit_angle(points, grid):
    sol = points[0]["sol"]
    omega = RPM * 2.0 * math.pi / 60.0
    assert CP * (sol.t02 - sol.t01) == pytest.approx(omega * sol.r2 * sol.c_theta2, rel=1e-10)
    beta2 = np.degrees(np.arctan2(sol.c_theta2 - sol.u2, sol.cx2))
    assert beta2 == pytest.approx(grid["beta2"], abs=0.1)


def test_lower_flow_raises_incidence_and_pressure_ratio(points):
    peak, stall = points
    assert np.all(stall["incidence"] > peak["incidence"])
    assert stall["pressure_ratio_avg"] > peak["pressure_ratio_avg"]
    assert stall["temperature_ratio_avg"] > peak["temperature_ratio_avg"]