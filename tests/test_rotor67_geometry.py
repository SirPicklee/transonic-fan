"""Checks on the transcribed Rotor 67 blade geometry."""

import numpy as np
import pytest

from fan.rotor67_geometry import load_blade


@pytest.fixture(scope="module")
def blade():
    return load_blade()


def test_fourteen_sections(blade):
    assert len(blade["chord"]) == 14


def test_solidity_matches_the_published_values(blade):
    """Independent check of the transcription: chord from the edge coordinates
    against the solidities given in the text of the report."""
    assert blade["solidity"][0] == pytest.approx(3.11, rel=0.01)
    assert blade["solidity"][-1] == pytest.approx(1.29, rel=0.01)


def test_camber_falls_then_rises_towards_the_tip(blade):
    """Camber falls from hub to about 75 % span and rises again near the tip:
    the tip sections are loaded more, which the rig data show as a total
    temperature ratio rising towards the casing."""
    camber = blade["camber"]
    assert np.all(camber > 0)
    assert np.all(np.diff(camber[:11]) < 0)
    assert camber[-1] > camber[10]


def test_transonic_tip_has_little_front_camber(blade):
    assert blade["front_fraction"][-1] < 0.2
    assert blade["front_fraction"][0] > 0.3


def test_thickness_ratio(blade):
    assert blade["thickness_ratio"][-1] == pytest.approx(0.03, abs=0.005)
    assert np.all(np.diff(blade["thickness_ratio"][:12]) < 0)