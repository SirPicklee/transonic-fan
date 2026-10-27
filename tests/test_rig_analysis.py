"""Check that the processing chain reproduces the published NASA values."""

import pytest

from fan.rig_analysis import flow_state, process
from fan.rig_data import POINTS, load_overall


def test_flow_state_recovers_mach_number():
    # Isentropic: p0/ps = (1 + 0.2 M^2)^3.5 = 1.18621 at M = 0.5
    mach, _, _ = flow_state(1.18621, 288.15, 1.0)
    assert mach == pytest.approx(0.5, abs=1e-4)


@pytest.fixture(scope="module", params=POINTS)
def case(request):
    return process(request.param), load_overall(request.param)


def test_integrated_mass_flows_match_table_v(case):
    r, o = case
    assert r["flow_station1"] == pytest.approx(o["station1_flow_kg_s"], rel=0.01)
    assert r["flow_station2"] == pytest.approx(o["station2_flow_kg_s"], rel=0.01)


def test_averaged_ratios_match_table_iii(case):
    r, o = case
    assert r["pressure_ratio_avg"] == pytest.approx(o["pressure_ratio"], rel=0.005)
    assert r["temperature_ratio_avg"] == pytest.approx(o["temperature_ratio"], rel=0.002)


def test_efficiency_matches_table_iii(case):
    r, o = case
    assert r["efficiency_avg"] == pytest.approx(o["efficiency"], abs=0.01)


def test_tip_is_the_least_efficient_streamline(case):
    r, _ = case
    assert r["efficiency"][0] == min(r["efficiency"])     # first row is nearest the tip