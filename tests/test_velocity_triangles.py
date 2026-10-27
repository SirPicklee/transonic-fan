"""Tests for the velocity triangles: kinematics, energy and limits."""

import math

import pytest

from fan.velocity_triangles import CP, GAMMA, Triangle, rotor_from_loading

T01 = 288.15


def test_relative_velocity_is_absolute_minus_blade_speed():
    t = Triangle(u=400.0, cx=180.0, c_theta=50.0)
    assert t.w_theta == pytest.approx(50.0 - 400.0)
    assert t.w == pytest.approx(math.hypot(180.0, 350.0))


def test_no_inlet_swirl_gives_axial_inflow_and_tan_beta_equal_minus_one_over_phi():
    rotor = rotor_from_loading(u=430.0, cx1=190.0, loading=0.25)
    assert rotor.inlet.alpha == pytest.approx(0.0)
    tan_beta1 = math.tan(math.radians(rotor.inlet.beta))
    assert tan_beta1 == pytest.approx(-1.0 / rotor.flow_coefficient)


def test_loading_round_trip():
    rotor = rotor_from_loading(u=350.0, cx1=170.0, loading=0.42)
    assert rotor.loading == pytest.approx(0.42)


def test_euler_work_matches_three_term_form():
    """dh0 = U dC_theta equals 1/2 [(C2^2 - C1^2) - (W2^2 - W1^2)] at constant radius:
    the work goes into kinetic energy and into diffusion of the relative flow."""
    rotor = rotor_from_loading(u=430.0, cx1=190.0, loading=0.25, cx2=175.0)
    i, o = rotor.inlet, rotor.outlet
    three_term = 0.5 * ((o.c ** 2 - i.c ** 2) - (o.w ** 2 - i.w ** 2))
    assert rotor.work == pytest.approx(three_term, rel=1e-12)


def test_zero_loading_does_no_work():
    rotor = rotor_from_loading(u=430.0, cx1=190.0, loading=0.0)
    assert rotor.work == pytest.approx(0.0)
    assert rotor.pressure_ratio(T01, efficiency=0.9) == pytest.approx(1.0)


def test_ideal_pressure_ratio_is_isentropic():
    rotor = rotor_from_loading(u=430.0, cx1=190.0, loading=0.25)
    t_ratio = 1.0 + rotor.work / (CP * T01)
    expected = t_ratio ** (GAMMA / (GAMMA - 1.0))
    assert rotor.pressure_ratio(T01, efficiency=1.0) == pytest.approx(expected)


def test_losses_reduce_pressure_ratio():
    rotor = rotor_from_loading(u=430.0, cx1=190.0, loading=0.25)
    assert rotor.pressure_ratio(T01, 0.85) < rotor.pressure_ratio(T01, 0.95)


def test_rotor_diffuses_the_relative_flow():
    # A compressor rotor turns the flow towards axial and slows it down: W2 < W1
    rotor = rotor_from_loading(u=430.0, cx1=190.0, loading=0.25)
    assert rotor.de_haller < 1.0
    assert abs(rotor.outlet.beta) < abs(rotor.inlet.beta)


def test_higher_loading_increases_diffusion_factor():
    low = rotor_from_loading(u=430.0, cx1=190.0, loading=0.20)
    high = rotor_from_loading(u=430.0, cx1=190.0, loading=0.35)
    assert high.diffusion_factor(1.3) > low.diffusion_factor(1.3)


def test_transonic_tip_has_supersonic_relative_and_subsonic_absolute_inflow():
    rotor = rotor_from_loading(u=430.0, cx1=190.0, loading=0.25)
    assert rotor.inlet.mach_rel(T01) > 1.0
    assert rotor.inlet.mach_abs(T01) < 1.0