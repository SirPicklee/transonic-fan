"""Consistency checks on the transcribed NASA TP-2879 data."""

import numpy as np
import pytest

from fan.rig_data import POINTS, load_overall, load_streamlines, load_survey


@pytest.mark.parametrize("point", POINTS)
def test_number_of_survey_radii(point):
    assert len(load_survey(1, point)["radius"]) == 19
    assert len(load_survey(2, point)["radius"]) == 9


@pytest.mark.parametrize("point", POINTS)
def test_station2_radii_are_the_design_streamlines(point):
    assert load_survey(2, point)["radius"] == pytest.approx(load_streamlines()["r2"])


@pytest.mark.parametrize("point", POINTS)
def test_station1_contains_the_design_streamlines(point):
    r1 = load_survey(1, point)["radius"]
    for r in load_streamlines()["r1"]:
        assert np.min(np.abs(r1 - r)) < 1e-9


@pytest.mark.parametrize("point", POINTS)
def test_published_efficiency_follows_from_pressure_and_temperature_ratio(point):
    """eta = (PR^((g-1)/g) - 1) / (TR - 1); the report rounds eta to 2-3 digits."""
    o = load_overall(point)
    eta = (o["pressure_ratio"] ** (0.4 / 1.4) - 1.0) / (o["temperature_ratio"] - 1.0)
    assert eta == pytest.approx(o["efficiency"], abs=0.005)


@pytest.mark.parametrize("point", POINTS)
def test_survey_radii_lie_inside_the_annulus(point):
    r1 = load_survey(1, point)["radius"]
    r2 = load_survey(2, point)["radius"]
    assert np.all((r1 > 0.091465) & (r1 < 0.25654))      # station 1 hub, shroud
    assert np.all((r2 > 0.120167) & (r2 < 0.244043))     # station 2 hub, shroud


def test_unknown_point_is_rejected():
    with pytest.raises(ValueError):
        load_survey(2, "design")