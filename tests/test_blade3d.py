"""Tests for blade stacking and export."""

import numpy as np
import pytest

from fan.blade3d import (polygon_centroid, rotor67_free_vortex_blade, section_loop,
                         stack_blade, write_stl)
from fan.blade_sections import circular_arc_section


def test_polygon_centroid_of_square():
    x = np.array([0.0, 2.0, 2.0, 0.0]) + 1.0
    y = np.array([0.0, 0.0, 2.0, 2.0]) - 3.0
    assert polygon_centroid(x, y) == pytest.approx((2.0, -2.0))


def test_section_loop_has_no_repeated_points():
    x, y = section_loop(circular_arc_section(30.0, -40.0, 0.1, 0.05, n_points=51))
    assert len(x) == 2 * 51 - 2
    assert np.hypot(np.diff(x), np.diff(y)).min() > 0


@pytest.fixture(scope="module")
def blade():
    return stack_blade(span=np.array([0.0, 0.5, 1.0]),
                       r1=np.array([0.100, 0.180, 0.257]),
                       r2=np.array([0.116, 0.180, 0.2425]),
                       camber=np.array([60.0, 25.0, 5.0]),
                       stagger=np.array([-10.0, -45.0, -65.0]),
                       chord=np.full(3, 0.094),
                       thickness_ratio=np.array([0.10, 0.06, 0.03]))


def test_sections_are_stacked_on_their_centroids(blade):
    for pts in blade.sections:
        r = np.hypot(pts[:, 1], pts[:, 2])
        theta = np.arctan2(pts[:, 1], pts[:, 2])
        cx, cy = polygon_centroid(pts[:, 0], r * theta)     # unrolled section
        assert cx == pytest.approx(0.0, abs=1e-9)
        assert cy == pytest.approx(0.0, abs=1e-9)


def test_edges_sit_at_inlet_and_exit_radius(blade):
    r = np.hypot(blade.sections[0][:, 1], blade.sections[0][:, 2])
    assert r[0] == pytest.approx(0.100)       # leading edge
    assert r[100] == pytest.approx(0.116)     # trailing edge (n_points = 101)


def test_stl_triangle_count(blade, tmp_path):
    path = tmp_path / "blade.stl"
    n = write_stl(blade, str(path))
    m = len(blade.sections[0])
    assert n == 2 * m * (len(blade.sections) - 1) + 2 * m
    assert path.read_text().count("facet normal") == n


def test_rotor67_blade_fills_the_annulus():
    b = rotor67_free_vortex_blade(n_sections=5, n_points=41)
    r_hub = np.hypot(b.sections[0][:, 1], b.sections[0][:, 2])
    r_tip = np.hypot(b.sections[-1][:, 1], b.sections[-1][:, 2])
    assert r_hub.min() == pytest.approx(0.193 / 2)    # inlet hub radius
    assert r_tip.max() == pytest.approx(0.514 / 2)    # inlet tip radius