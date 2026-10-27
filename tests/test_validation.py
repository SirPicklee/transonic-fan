"""Tests for the measurement uncertainty Monte Carlo."""

import numpy as np
import pytest

from fan.rig_analysis import process
from fan.validation import Instrumentation, monte_carlo

POINT = "peak_efficiency"
NO_ERRORS = Instrumentation(0.0, 0.0, 0.0, 0.0, 0.0)
TEMPERATURE_ONLY = Instrumentation(t_random=0.25, t_bias=0.25, p_random=0.0,
                                   p_bias=0.0, angle_random=0.0)


def test_no_instrument_error_gives_the_nominal_result():
    mc = monte_carlo(POINT, NO_ERRORS, n_samples=5)
    assert np.ptp(mc["efficiency_avg"]) == 0.0
    assert mc["efficiency_avg"][0] == pytest.approx(process(POINT)["efficiency_avg"])


def test_bias_widens_the_uncertainty():
    random_only = monte_carlo(POINT, Instrumentation(t_bias=0.0, p_bias=0.0), 400)
    with_bias = monte_carlo(POINT, Instrumentation(), 400)
    assert with_bias["efficiency_avg"].std() > random_only["efficiency_avg"].std()


def test_spread_scales_with_temperature_error():
    """Small errors: doubling the temperature error doubles the efficiency spread."""
    one = monte_carlo(POINT, TEMPERATURE_ONLY, 400, seed=1)
    two = monte_carlo(POINT, Instrumentation(0.5, 0.5, 0.0, 0.0, 0.0), 400, seed=1)
    ratio = two["efficiency_avg"].std() / one["efficiency_avg"].std()
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_monte_carlo_mean_is_unbiased():
    mc = monte_carlo(POINT, Instrumentation(), 1000)
    assert mc["efficiency_avg"].mean() == pytest.approx(
        process(POINT)["efficiency_avg"], abs=0.002)