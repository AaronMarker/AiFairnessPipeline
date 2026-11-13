"""Run the disparity analysis on a table of predictions."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.metrics import METRICS, order_errors
from src.plots import make_plots
from src.significance import bootstrap_p_value

NUM_BINS = 3  # terciles, used to color the scatterplots


def tercile_labels(values, num_bins=NUM_BINS):
    """Label each value 0..num_bins-1 by quantile group of the factor."""
    values = pd.Series(values, dtype=float)
    edges = ([values.min() - 1] +
             [values.quantile(i / num_bins) for i in range(1, num_bins)] +
             [values.max()])
    try:
        bins = pd.cut(values, bins=edges, labels=list(range(num_bins)))
    except ValueError as e:
        raise ValueError(
            f"Could not split the sociodemographic factor into {num_bins} "
            f"groups for plotting (it has only {values.nunique()} distinct "
            "values). This tool expects a continuous variable such as "
            "median income; binary or categorical attributes are not "
            "supported.") from e
    return bins.astype(int)


def prepare_data(df, demographic, pred=None, true=None, error=None,
                 baseline=None):
    """Collect the selected columns into one numeric table.

    Errors come from `error` if given, otherwise |pred - true|. `baseline`
    holds the baseline model's predictions, or its errors when `error` is
    used. Rows with missing or non-numeric values in any selected column are
    dropped. Returns columns demographic, error, bin, and when available
    pred, true and baseline_error.
    """
    if error is None and (pred is None or true is None):
        raise ValueError("Select either an error column or both predicted "
                         "and true value columns")

    def numeric(column):
        return pd.to_numeric(df[column], errors="coerce")

    data = pd.DataFrame({"demographic": numeric(demographic)})
    if error is not None:
        data["error"] = numeric(error).abs()
    else:
        data["pred"], data["true"] = numeric(pred), numeric(true)
        data["error"] = (data["pred"] - data["true"]).abs()
    if baseline is not None:
        base = numeric(baseline)
        data["baseline_error"] = (base.abs() if error is not None else
                                  (base - data["true"]).abs())
    data = data.dropna().reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("At least 2 complete rows are required")
    if data["demographic"].nunique() < 2:
        raise ValueError("The sociodemographic factor has fewer than 2 "
                         "distinct values")
    data["bin"] = tercile_labels(data["demographic"])
    return data


def accuracy_metrics(pred, true):
    """Standard accuracy metrics for predicted vs. true values."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    residual = true - pred
    return {
        "n": int(true.size),
        "mse": float(np.mean(residual**2)),
        "r_squared": float(1 - np.sum(residual**2) /
                           np.sum((true - true.mean())**2)),
        "pearson": float(pearsonr(true, pred)[0]),
        "spearman": float(spearmanr(true, pred)[0]),
    }


def run_analysis(df, demographic, pred=None, true=None, error=None,
                 baseline=None, metrics=tuple(METRICS), graphs=(),
                 n_resamples=1000, rng=None):
    """Compute disparity metrics, p-values and figures.

    `metrics` is a collection of names from METRICS ("bci", "ci", "ks",
    "ad"); `graphs` of names from {"bci", "ks", "scatter"}. Only the
    requested ones are computed.

    Returns (results, figures). `results["metrics"][name]` has "value",
    "p_vs_random" and "p_vs_baseline" (None without a baseline);
    `results["accuracy"]` is None when only an error column was given.
    `figures` is a list of (label, Figure).
    """
    data = prepare_data(df, demographic, pred, true, error, baseline)
    ordered = order_errors(data["error"], data["demographic"])
    has_baseline = "baseline_error" in data

    results = {"n": len(data), "metrics": {}, "accuracy": None}
    for name in metrics:
        metric = METRICS[name]
        results["metrics"][name] = {
            "value": metric(ordered),
            "p_vs_random": bootstrap_p_value(
                metric, data["error"], data["demographic"],
                n_resamples=n_resamples, rng=rng),
            "p_vs_baseline": bootstrap_p_value(
                metric, data["error"], data["demographic"],
                baseline_errors=data["baseline_error"],
                n_resamples=n_resamples, rng=rng) if has_baseline else None,
        }
    if "pred" in data:
        results["accuracy"] = accuracy_metrics(data["pred"], data["true"])

    outcome_name = true if true is not None else error
    figures = make_plots(data, graphs, demographic, outcome_name)
    return results, figures
