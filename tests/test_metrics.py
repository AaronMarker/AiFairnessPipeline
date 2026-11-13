import numpy as np
import pytest

from src.metrics import (anderson_darling, bci, concentration_curve,
                         concentration_index, ks_statistic, order_errors)

N = 50


def all_error_in_last_unit(n=N):
    errors = np.zeros(n)
    errors[-1] = 1.0
    return errors


def numerical_bci(errors, points=200_001):
    """Independent check: integrate |curve - equality| on a fine grid."""
    curve = concentration_curve(errors)
    knots = np.linspace(0, 1, curve.size)
    x = np.linspace(0, 1, points)
    y = np.abs(np.interp(x, knots, curve) - x)
    return 2 * np.sum((y[1:] + y[:-1]) / 2 * np.diff(x))


# Order ----------------------------------------------------------------------

def test_order_errors_sorts_by_demographic():
    assert order_errors([1, 2, 3], [30, 10, 20]).tolist() == [2, 3, 1]


def test_order_errors_breaks_ties_by_error_size():
    # Matches how the paper's results were computed
    assert order_errors([5, 1, 3], [7, 7, 7]).tolist() == [1, 3, 5]


def test_order_errors_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        order_errors([1, 2], [1, 2, 3])


# Edge cases ------------------------------------------------------------------

def test_all_zero_errors_rejected():
    with pytest.raises(ValueError, match="zero"):
        bci(np.zeros(5))


def test_negative_errors_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        bci([1.0, -1.0, 2.0])


@pytest.mark.parametrize("metric", [bci, concentration_index, ks_statistic,
                                    anderson_darling])
def test_equal_errors_mean_no_disparity(metric):
    assert metric(np.full(N, 2.5)) == pytest.approx(0, abs=1e-12)


# Known answers -----------------------------------------------------------------

def test_bci_all_error_in_last_unit():
    # The gap is -x until (N-1)/N, then closes linearly: area = (N-1)/(2N)
    assert bci(all_error_in_last_unit()) == pytest.approx((N - 1) / N)


def test_ci_all_error_in_last_unit():
    assert concentration_index(all_error_in_last_unit()) == pytest.approx(
        -(N - 1) / N)


def test_ks_all_error_in_last_unit():
    assert ks_statistic(all_error_in_last_unit()) == pytest.approx((N - 1) / N)


def test_bci_matches_numerical_integration_when_curve_crosses_equality():
    # High error at both ends, low in the middle: the curve crosses the line
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 400)
    errors = (x**2 + 0.1) * rng.exponential(size=x.size)
    gap = concentration_curve(errors) - np.linspace(0, 1, errors.size + 1)
    assert np.any(gap[:-1] * gap[1:] < 0), "test data should cross"
    assert bci(errors) == pytest.approx(numerical_bci(errors), abs=1e-6)


# Relationships between metrics -----------------------------------------------

def test_bci_equals_abs_ci_without_crossing():
    increasing = np.linspace(1, 5, 100)  # curve stays below the line
    assert bci(increasing) == pytest.approx(abs(concentration_index(increasing)))


def test_bci_exceeds_abs_ci_when_deviations_cancel():
    # Error high in the middle: CI cancels, BCI does not (paper, Fig. 1)
    x = np.linspace(-1, 1, 101)
    errors = 2 - x**2
    assert abs(concentration_index(errors)) < 1e-9
    assert bci(errors) > 0.01


def test_reversing_order_flips_ci_but_not_bci():
    rng = np.random.default_rng(1)
    errors = rng.exponential(size=200) * np.linspace(1, 3, 200)
    assert bci(errors[::-1]) == pytest.approx(bci(errors))
    assert concentration_index(errors[::-1]) == pytest.approx(
        -concentration_index(errors))


def test_bci_scale_invariant():
    errors = np.random.default_rng(2).exponential(size=100)
    assert bci(errors * 1000) == pytest.approx(bci(errors))
