"""Figures for the disparity analysis (styled as in the paper)."""

import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.metrics import bci, concentration_curve, ks_statistic, order_errors  # noqa: E402

CHART_TITLE_WIDTH = 32
STYLE = {
    "font.size": 17,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
    "axes.linewidth": 2,
}
TERCILE_NAMES = ["Low", "Medium", "High"]


def round_to_2_sig_figs(x):
    """Tick label with two significant figures and no scientific notation."""
    if x == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(x))))
    factor = 10**(magnitude - 1)
    rounded = round(x / factor, 1) * factor
    return f"{rounded:.0f}" if magnitude >= 2 else f"{rounded:.1f}"


def _set_title(ax, title):
    ax.set_title("\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH)))


def _three_y_ticks(ax):
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))


def _tercile_legend(ax, palette, **kwargs):
    handles = [Patch(facecolor=palette[i], label=name)
               for i, name in enumerate(TERCILE_NAMES)]
    legend = ax.legend(handles=handles, title="Bin", loc="upper right",
                       frameon=True, edgecolor="gray", fancybox=True,
                       framealpha=0.9, **kwargs)
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_boxstyle("round,pad=0.3")


def _curve_points(data):
    errors = order_errors(data["error"], data["demographic"])
    curve = concentration_curve(errors)
    return errors, np.linspace(0, 1, curve.size), curve


def plot_concentration_curve(ax, data, demographic_name, outcome_name):
    """Concentration curve against the line of equality, labeled with BCI."""
    errors, population_share, curve = _curve_points(data)
    ax.plot(population_share, curve, color="blue",
            label=f"BCI: {bci(errors) * 100:.1f}%")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--",
            label="Line of Equality")
    ax.fill_between(population_share, curve, population_share, color="blue",
                    alpha=0.2)
    _set_title(ax, " BCI")
    _three_y_ticks(ax)
    ax.set_xlabel("Cumulative proportion of counties ordered by \n" +
                  demographic_name)
    ax.set_ylabel("Cumulative % of error predicting \n" + outcome_name,
                  labelpad=-5)
    ax.legend()


def plot_ks_curve(ax, data, demographic_name, outcome_name):
    """Concentration curve with the KS distance (largest gap) marked."""
    errors, population_share, curve = _curve_points(data)
    largest = np.argmax(np.abs(curve - population_share))
    ax.plot(population_share, curve, color="gray")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--",
            label="Line of Equality")
    ax.fill_between(population_share, curve, population_share, color="gray",
                    alpha=0.2)
    ax.vlines(population_share[largest], curve[largest],
              population_share[largest], color="blue", linestyle="dashed",
              linewidth=2, label=f"KS Distance: {ks_statistic(errors):.3f}")
    ax.set_title("KS Test Curve")
    ax.set_xlabel("Cumulative proportion of counties")
    ax.set_ylabel("Cumulative predicted error", labelpad=-5)
    ax.legend()


def plot_error_scatter(ax, data, demographic_name, palette):
    """Absolute error against the demographic factor, colored by tercile,
    with a LOESS curve. The outer 5% of each axis is trimmed for display."""
    x_min, x_max = data["demographic"].quantile([0.05, 0.95])
    y_min, y_max = data["error"].quantile([0.05, 0.95])
    shown = data[data["demographic"].between(x_min, x_max)
                 & data["error"].between(y_min, y_max)]
    sns.scatterplot(x="demographic", y="error", hue="bin", data=shown, ax=ax,
                    palette=dict(enumerate(palette)), hue_order=[0, 1, 2],
                    alpha=.9, legend=False, s=18, edgecolor="none")
    sns.regplot(x="demographic", y="error", data=shown, ax=ax, scatter=False,
                lowess=True, color="black", line_kws={"lw": 5})
    _set_title(ax, "Prediction Error")
    _three_y_ticks(ax)
    ax.set_xlabel(demographic_name, labelpad=-1)
    ax.set_ylabel("Absolute Error", labelpad=-3)
    _tercile_legend(ax, palette)


def plot_true_vs_predicted(ax, data, palette):
    """True against predicted values, colored by tercile, with one regression
    line per tercile. The outer 2.5% of each axis is trimmed for display."""
    x_min, x_max = data["pred"].quantile([0.025, 0.975])
    y_min, y_max = data["true"].quantile([0.025, 0.975])
    shown = data[data["pred"].between(x_min, x_max)
                 & data["true"].between(y_min, y_max)]
    sns.scatterplot(x="pred", y="true", hue="bin", data=shown, ax=ax,
                    palette=dict(enumerate(palette)), hue_order=[0, 1, 2],
                    alpha=.5, legend=False, s=18, edgecolor="none")
    for tercile, group in shown.groupby("bin", observed=True):
        sns.regplot(x="pred", y="true", data=group, ax=ax, scatter=False,
                    color=mcolors.to_rgba(palette[int(tercile)], alpha=0.4),
                    line_kws={"lw": 5, "alpha": 1}, ci=None)
    _set_title(ax, "True vs Predicted")
    _three_y_ticks(ax)
    ax.set_xlabel("Predicted", labelpad=-1)
    ax.set_ylabel("True", labelpad=-3)
    _tercile_legend(ax, palette, borderpad=0.5)


def make_plots(data, graphs, demographic_name, outcome_name):
    """Build the requested figures in memory.

    `data` has columns demographic, error and bin, plus pred and true when
    they are available. `graphs` is a collection of names from
    {"bci", "ks", "scatter"}. Returns (label, Figure) pairs in display order;
    the caller closes the figures.
    """
    palette = sns.color_palette("deep", n_colors=3)
    has_pred_true = {"pred", "true"} <= set(data.columns)
    figures = []

    def add(label, draw):
        fig, ax = plt.subplots(figsize=(7, 6))
        draw(ax)
        figures.append((label, fig))

    with plt.rc_context(STYLE):
        if "bci" in graphs:
            add("BCI Curve", lambda ax: plot_concentration_curve(
                ax, data, demographic_name, outcome_name))
        if "ks" in graphs:
            add("KS Test Curve", lambda ax: plot_ks_curve(
                ax, data, demographic_name, outcome_name))
        if "scatter" in graphs:
            add("Prediction Error vs. Sociodemographic Factor",
                lambda ax: plot_error_scatter(ax, data, demographic_name,
                                              palette))
            if has_pred_true:
                add("True vs. Predicted",
                    lambda ax: plot_true_vs_predicted(ax, data, palette))
    return figures
