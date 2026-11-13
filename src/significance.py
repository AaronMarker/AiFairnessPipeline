"""Bootstrap significance tests for disparity metrics (the paper's method)."""

import numpy as np

from src.metrics import order_errors


def bootstrap_p_value(metric, errors, demographic, baseline_errors=None,
                      n_resamples=1000, rng=None):
    """One-sided p-value that the disparity in `errors` exceeds a comparison.

    Each round computes `metric` on a bootstrap resample of the
    (demographic, error) pairs and compares it with an independent resample
    of the comparison:

    - `baseline_errors` is None: a random ordering, made by pairing resampled
      errors with the original demographic values (paper, Table 3).
    - otherwise: the baseline model's errors on the same units (Table 4).

    The p-value is the share of rounds in which the comparison's disparity is
    larger (in absolute value) than the model's. This test is conservative:
    p-values tend to be too large rather than too small.
    """
    rng = np.random.default_rng() if rng is None else rng
    errors = np.asarray(errors, dtype=float)
    demographic = np.asarray(demographic, dtype=float)
    if baseline_errors is not None:
        baseline_errors = np.asarray(baseline_errors, dtype=float)
    n = errors.size

    comparison_larger = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        observed = metric(order_errors(errors[idx], demographic[idx]))

        idx = rng.integers(0, n, n)
        if baseline_errors is None:
            comparison = metric(order_errors(errors[idx], demographic))
        else:
            comparison = metric(
                order_errors(baseline_errors[idx], demographic[idx]))

        if abs(comparison) > abs(observed):
            comparison_larger += 1

    return comparison_larger / n_resamples
