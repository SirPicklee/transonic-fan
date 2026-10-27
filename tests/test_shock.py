"""Tests for the Prandtl-Meyer function and the MLH shock loss."""

import math

import numpy as np
import pytest

from fan.annulus import ROTOR67, Annulus
from fan.blade_sections import circular_arc_section
from fan.losses import analyze, rotor67_solidity, rotor67_thickness
from fan.radial_equilibrium import solve_exit
from fan.shock import (mach_from_prandtl_meyer, mlh_shock_loss, prandtl_meyer,
                       shock_loss, suction_surface_turning)

INLET = ROTOR67["annulus_inlet"]
EXIT = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
RPM, MDOT = ROTOR67["rpm"], ROTOR67["mass_flow"]


def test_prandtl_meyer_table_values():
    assert prandtl_meyer(1.0) == 0.0
    assert math.degrees(prandtl_meyer(2.0)) == pytest.approx(26.38, abs=0.01)


@pytest.mark.parametrize("mach", [1.1, 1.4, 2.0])
def test_prandtl_meyer_round_trip(mach):
    assert mach_from_prandtl_meyer(prandtl_meyer(mach)) == pytest.approx(mach, rel=1e-8)


def test_flat_plate_aligned_with_the_flow_does_not_expand_it():
    sec = circular_arc_section(0.0, -65.0, 0.09, 0.0)
    assert suction_surface_turning(sec, beta1_deg=-65.0, pitch=0.07) == pytest.approx(0.0)


def test_flat_plate_at_incidence_expands_the_flow_by_the_incidence():
    sec = circular_arc_section(0.0, -65.0, 0.09, 0.0)
    turning = suction_surface_turning(sec, beta1_deg=-68.0, pitch=0.07)
    assert math.degrees(turning) == pytest.approx(3.0, abs=1e-6)


def test_cambered_section_expands_the_flow_by_less_than_its_camber():
    # zero incidence on the camber line: beta1 = kappa1 = stagger - camber / 2
    sec = circular_arc_section(10.0, -60.0, 0.09, 0.0)
    turning = math.degrees(suction_surface_turning(sec, beta1_deg=-65.0, pitch=0.07))
    assert 0.0 < turning < 10.0


@pytest.fixture(scope="module")
def free_vortex():
    return solve_exit(INLET, EXIT, RPM, MDOT, c_theta_m=156.5, n=-1, efficiency=0.92)


def test_mlh_loss_is_at_least_the_normal_shock_loss(free_vortex):
    sol = free_vortex
    mlh = mlh_shock_loss(sol, rotor67_solidity(sol.span), rotor67_thickness(sol.span))
    normal = shock_loss(sol.mach_rel1())
    assert np.all(mlh >= normal - 1e-12)
    assert np.all(mlh[sol.mach_rel1() <= 1.0] == 0.0)


def test_mlh_lowers_the_tip_efficiency_and_shifts_flow_to_the_hub():
    """More loss in the outer span: radial equilibrium moves flow towards the
    hub, where the higher axial velocity lowers the diffusion. The subsonic hub
    gains efficiency although its own losses did not change model."""
    sol_n, eta_n, *_ = analyze(INLET, EXIT, RPM, MDOT, 156.5, -1, shock_model="normal")
    sol_m, eta_m, *_ = analyze(INLET, EXIT, RPM, MDOT, 156.5, -1, shock_model="mlh")
    assert eta_m[-1] < eta_n[-1]
    assert sol_m.cx2[0] > sol_n.cx2[0]
    assert eta_m[0] >= eta_n[0]


def test_unknown_shock_model_is_rejected():
    with pytest.raises(ValueError):
        analyze(INLET, EXIT, RPM, MDOT, 156.5, -1, shock_model="oblique")



from fan.shock import mlh_details  # noqa: E402


def test_low_front_camber_reduces_the_expansion_before_the_shock(free_vortex):
    sol = free_vortex
    sigma, thick = rotor67_solidity(sol.span), rotor67_thickness(sol.span)
    t_arc, _, w_arc = mlh_details(sol, sigma, thick, front_fraction=0.5)
    t_low, _, w_low = mlh_details(sol, sigma, thick, front_fraction=0.25)
    i = 60                                     # 60 % span, supersonic
    assert t_low[i] < t_arc[i]
    assert w_low[i] < w_arc[i]