"""Tests for the secondary air flow network solver."""

import math

import pytest

from fan.annulus import Q_CHOKE
from fan.secondary_air import Element, Network, example_network
from fan.velocity_triangles import R_AIR


def test_choked_orifice_matches_the_flow_function():
    """Same physics as the annulus sizing: at choke m sqrt(R T0) / (A p0) = 0.6847."""
    e = Element("o", "orifice", "a", "b", area=1e-3, cd=0.8)
    m = e.flow(5e5, 600.0, 1e5, 600.0)
    assert m == pytest.approx(0.8 * 1e-3 * 5e5 / math.sqrt(R_AIR * 600.0) * Q_CHOKE, rel=1e-6)
    assert e.flow(5e5, 600.0, 2e5, 600.0) == pytest.approx(m)     # x = 0.4, still choked


def test_reverse_flow_is_antisymmetric():
    e = Element("o", "orifice", "a", "b", area=1e-3)
    assert e.flow(2e5, 500.0, 1.5e5, 500.0) == pytest.approx(-e.flow(1.5e5, 500.0, 2e5, 500.0))


def test_more_teeth_leak_less():
    few = Element("l", "labyrinth", "a", "b", area=1e-4, teeth=2)
    many = Element("l", "labyrinth", "a", "b", area=1e-4, teeth=8)
    assert many.flow(3e5, 500.0, 1e5, 500.0) < few.flow(3e5, 500.0, 1e5, 500.0)


def test_orifices_in_series_conserve_mass():
    net = Network({"in": (2e5, 300.0), "out": (1.5e5, 300.0)}, ["mid"],
                  [Element("o1", "orifice", "in", "mid", 1e-3, 0.8),
                   Element("o2", "orifice", "mid", "out", 1e-3, 0.8)])
    res = net.solve()
    assert res["flows"]["o1"] == pytest.approx(res["flows"]["o2"], rel=1e-8)
    assert 1.5e5 < res["p"]["mid"] < 2e5


def test_example_network_conserves_mass():
    f = example_network().solve()["flows"]
    assert f["supply holes"] == pytest.approx(f["inner labyrinth"] + f["rim seal"], rel=1e-8)
    assert f["supply holes"] == pytest.approx(f["pre-swirl nozzles"], rel=1e-8)


def test_seal_wear_reduces_the_purge_flow():
    new = example_network(0.4e-3).solve()["flows"]
    worn = example_network(1.0e-3).solve()["flows"]
    assert worn["inner labyrinth"] > new["inner labyrinth"]
    assert worn["rim seal"] < new["rim seal"]