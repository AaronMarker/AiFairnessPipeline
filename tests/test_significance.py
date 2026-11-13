import numpy as np

from src.metrics import bci
from src.significance import bootstrap_p_value

N = 500


def demographic_and_errors(disparity, seed):
    rng = np.random.default_rng(seed)
    demographic = rng.normal(size=N)
    errors = rng.exponential(size=N) * (1 + disparity * (demographic > 0))
    return demographic, errors


def test_strong_disparity_is_significant():
    demographic, errors = demographic_and_errors(disparity=1.0, seed=0)
    p = bootstrap_p_value(bci, errors, demographic, n_resamples=300,
                          rng=np.random.default_rng(0))
    assert p < 0.01


def test_no_disparity_is_not_significant():
    demographic, errors = demographic_and_errors(disparity=0.0, seed=0)
    p = bootstrap_p_value(bci, errors, demographic, n_resamples=300,
                          rng=np.random.default_rng(0))
    assert p > 0.05


def test_baseline_comparison_detects_larger_disparity():
    demographic, baseline_errors = demographic_and_errors(disparity=0.0, seed=1)
    _, model_errors = demographic_and_errors(disparity=1.0, seed=1)
    p = bootstrap_p_value(bci, model_errors, demographic,
                          baseline_errors=baseline_errors, n_resamples=300,
                          rng=np.random.default_rng(0))
    assert p < 0.01


def test_baseline_comparison_is_one_sided():
    # A model with *less* disparity than its baseline is not significant
    demographic, baseline_errors = demographic_and_errors(disparity=1.0, seed=1)
    _, model_errors = demographic_and_errors(disparity=0.0, seed=1)
    p = bootstrap_p_value(bci, model_errors, demographic,
                          baseline_errors=baseline_errors, n_resamples=300,
                          rng=np.random.default_rng(0))
    assert p > 0.95


def test_same_seed_gives_same_p_value():
    demographic, errors = demographic_and_errors(disparity=0.3, seed=2)
    p1 = bootstrap_p_value(bci, errors, demographic, n_resamples=100,
                           rng=np.random.default_rng(5))
    p2 = bootstrap_p_value(bci, errors, demographic, n_resamples=100,
                           rng=np.random.default_rng(5))
    assert p1 == p2
