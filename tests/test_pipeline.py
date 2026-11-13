import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.metrics import bci, order_errors
from src.pipeline import prepare_data, run_analysis, tercile_labels


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    n = 300
    income = rng.normal(50, 10, n)
    true = rng.normal(20, 5, n)
    pred = true + rng.normal(0, 1, n) * (1 + (income < 45))
    baseline = true + rng.normal(0, 1, n)
    return pd.DataFrame({"income": income, "true": true, "pred": pred,
                         "baseline": baseline,
                         "error": np.abs(pred - true)})


def run(df, **kwargs):
    kwargs.setdefault("n_resamples", 50)
    kwargs.setdefault("rng", np.random.default_rng(0))
    results, figures = run_analysis(df, "income", **kwargs)
    labels = [label for label, _ in figures]
    for _, fig in figures:
        plt.close(fig)
    return results, labels


def test_metric_values_match_direct_computation(predictions):
    results, _ = run(predictions, pred="pred", true="true")
    expected = bci(order_errors(predictions["error"], predictions["income"]))
    assert results["metrics"]["bci"]["value"] == pytest.approx(expected)


def test_only_requested_metrics_are_computed(predictions):
    results, _ = run(predictions, pred="pred", true="true",
                     metrics=["bci", "ks"])
    assert set(results["metrics"]) == {"bci", "ks"}


def test_baseline_p_value_only_with_baseline(predictions):
    without, _ = run(predictions, pred="pred", true="true", metrics=["bci"])
    with_base, _ = run(predictions, pred="pred", true="true",
                       baseline="baseline", metrics=["bci"])
    assert without["metrics"]["bci"]["p_vs_baseline"] is None
    assert 0 <= with_base["metrics"]["bci"]["p_vs_baseline"] <= 1


def test_error_column_gives_same_metrics_as_pred_and_true(predictions):
    from_pred, _ = run(predictions, pred="pred", true="true", metrics=["bci"])
    from_error, _ = run(predictions, error="error", metrics=["bci"])
    assert (from_error["metrics"]["bci"]["value"] ==
            pytest.approx(from_pred["metrics"]["bci"]["value"]))
    assert from_error["accuracy"] is None
    assert from_pred["accuracy"]["n"] == len(predictions)


def test_figure_labels_follow_selected_graphs(predictions):
    _, labels = run(predictions, pred="pred", true="true", metrics=[],
                    graphs=["ks", "scatter"])
    assert labels == ["KS Test Curve",
                      "Prediction Error vs. Sociodemographic Factor",
                      "True vs. Predicted"]


def test_true_vs_predicted_skipped_for_error_column(predictions):
    _, labels = run(predictions, error="error", metrics=[],
                    graphs=["bci", "scatter"])
    assert labels == ["BCI Curve",
                      "Prediction Error vs. Sociodemographic Factor"]


def test_rows_with_missing_or_non_numeric_values_are_dropped(predictions):
    df = predictions.astype(object)
    df.loc[0, "pred"] = "n/a"
    df.loc[1, "income"] = None
    data = prepare_data(df, "income", pred="pred", true="true")
    assert len(data) == len(predictions) - 2


def test_requires_error_or_pred_and_true(predictions):
    with pytest.raises(ValueError, match="error column"):
        prepare_data(predictions, "income", pred="pred")


def test_binary_demographic_rejected_with_clear_message():
    with pytest.raises(ValueError, match="continuous variable"):
        tercile_labels([0] * 7 + [1] * 3)


def test_terciles_split_continuous_values_evenly():
    counts = tercile_labels(np.arange(300.0)).value_counts()
    assert counts.tolist() == [100, 100, 100]
