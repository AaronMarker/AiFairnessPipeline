"""Disparity metrics for continuous sociodemographic factors.

Every metric here takes absolute prediction errors that are already ordered by
the sociodemographic factor (see `order_errors`). Each one compares the
concentration curve -- the cumulative share of total error at each cumulative
share of units -- with the line of equality, where every unit carries the same
error. Equation numbers refer to Marker et al., "Quantifying Error Disparities
in Population Health Models" (IJCAI-26).
"""

import numpy as np


def order_errors(errors, demographic):
    """Return `errors` ordered by `demographic`, ascending.

    Ties in the demographic factor are broken by error size (ascending), which
    is how the paper's results were computed.
    """
    errors = np.asarray(errors, dtype=float)
    demographic = np.asarray(demographic, dtype=float)
    if errors.shape != demographic.shape:
        raise ValueError("errors and demographic must have the same length")
    # lexsort uses the last key as the primary sort key
    return errors[np.lexsort((errors, demographic))]


def concentration_curve(ordered_errors):
    """Cumulative share of total error after 0, 1, ..., N units (N + 1 points)."""
    errors = np.asarray(ordered_errors, dtype=float)
    if errors.size == 0:
        raise ValueError("At least one error value is required")
    if np.any(errors < 0):
        raise ValueError("Errors must be absolute (non-negative)")
    total = errors.sum()
    if total == 0:
        raise ValueError("All errors are zero, so disparity is undefined")
    return np.concatenate([[0.0], np.cumsum(errors) / total])


def _gap_from_equality(ordered_errors):
    """Concentration curve minus the line of equality, at the N + 1 knots."""
    curve = concentration_curve(ordered_errors)
    return curve - np.linspace(0, 1, curve.size)


def bci(ordered_errors):
    """Bilateral Concentration Index (Eq. 8).

    Twice the absolute area between the concentration curve and the line of
    equality, so disparities in opposite directions add up instead of
    cancelling. Ranges from 0 (equal error everywhere) towards 1.

    Both curves are linear on each interval [i/N, (i+1)/N], so each piece of
    the integral has a closed form: a trapezoid when the gap keeps its sign,
    or two triangles when the curve crosses the line of equality.
    """
    gap = _gap_from_equality(ordered_errors)
    n = gap.size - 1
    left, right = np.abs(gap[:-1]), np.abs(gap[1:])
    crosses = gap[:-1] * gap[1:] < 0
    # Only divide on crossing pieces, where left + right > 0
    safe_sum = np.where(crosses, left + right, 1.0)
    piece_area = np.where(crosses, (left**2 + right**2) / (2 * safe_sum),
                          (left + right) / 2) / n
    return float(2 * piece_area.sum())


def concentration_index(ordered_errors):
    """Concentration Index (CI): twice the signed area between the curves.

    CI > 0 means error is concentrated among units low on the demographic
    factor (the curve lies above the line of equality); CI < 0 means it is
    concentrated among units high on it. Deviations in opposite directions
    cancel, which is the limitation BCI addresses.
    """
    gap = _gap_from_equality(ordered_errors)
    n = gap.size - 1
    # The gap is piecewise linear, so the trapezoid rule is exact
    return float(2 * ((gap[:-1] + gap[1:]) / 2).sum() / n)


def ks_statistic(ordered_errors):
    """Kolmogorov-Smirnov statistic: largest vertical gap between the curves."""
    return float(np.abs(_gap_from_equality(ordered_errors)).max())


def anderson_darling(ordered_errors):
    """Anderson-Darling-style statistic: squared gap weighted towards the tails.

    The weight 1 / (x(1 - x)) is undefined at the end points, where the gap
    is always zero, so they are left out. Unbounded, so hard to compare
    across settings.
    """
    gap = _gap_from_equality(ordered_errors)
    n = gap.size - 1
    x = np.linspace(0, 1, n + 1)[1:-1]
    return float(n * np.sum(gap[1:-1]**2 / (x * (1 - x))))


METRICS = {
    "bci": bci,
    "ci": concentration_index,
    "ks": ks_statistic,
    "ad": anderson_darling,
}
