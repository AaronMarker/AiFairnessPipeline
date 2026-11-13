"""Streamlit web app: upload predictions and measure sociodemographic error
disparities with the Bilateral Concentration Index (BCI) and related metrics."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pipeline import run_analysis

N_RESAMPLES = 1000
METRIC_LABELS = {
    "bci": "BCI",
    "ci": "Concentration Index (CI)",
    "ks": "KS Statistic",
    "ad": "Anderson-Darling",
}
SAMPLE_DATA = pd.DataFrame({
    "id": list(range(1, 21)),
    "percent_high_school": [
        74.98, 98.03, 89.28, 83.95, 66.24, 66.24, 62.32, 94.65, 84.04, 88.32,
        60.82, 98.8, 93.3, 68.49, 67.27, 67.34, 72.17, 80.99, 77.28, 71.65
    ],
    "true": [
        34.47, 15.58, 21.69, 24.65, 28.24, 41.41, 17.99, 30.57, 33.7, 11.86,
        34.3, 16.82, 12.6, 47.96, 48.63, 42.34, 22.18, 13.91, 37.37, 27.61
    ],
    "predicted": [
        34.4, 10.29, 25.8, 18.55, 29.28, 31.61, 11.35, 31.55, 37.39, 12.72,
        33.72, 15.31, 5.21, 44.36, 46.33, 47.63, 23.9, 5.09, 38.99, 25.68
    ],
    "error": [
        0.07, 5.29, 4.11, 6.1, 1.04, 9.8, 6.64, 0.98, 3.69, 0.86, 0.58, 1.51,
        7.39, 3.6, 2.3, 5.29, 1.72, 8.82, 1.62, 1.93
    ],
})


def format_p(p):
    if p is None:
        return "—"
    if p == 0:
        return f"< {1 / N_RESAMPLES:g}"  # below the bootstrap's resolution
    return f"{p:.3g}"


def results_table(results):
    """Metrics as columns; value and p-value rows."""
    rows = {"Value": {}, "P (vs. random ordering)": {}}
    with_baseline = any(m["p_vs_baseline"] is not None
                        for m in results["metrics"].values())
    if with_baseline:
        rows["P (vs. baseline)"] = {}
    for name, m in results["metrics"].items():
        label = METRIC_LABELS[name]
        rows["Value"][label] = f"{m['value']:.3f}"
        rows["P (vs. random ordering)"][label] = format_p(m["p_vs_random"])
        if with_baseline:
            rows["P (vs. baseline)"][label] = format_p(m["p_vs_baseline"])
    return pd.DataFrame(rows).T


st.set_page_config(page_title="AI Fairness Pipeline", layout="wide")
st.title("AI Fairness Pipeline")
st.markdown("Run bias detection analysis on machine learning models")

st.subheader("Bilateral Concentration Index")
ic1, ic2, ic3 = st.columns([4, 2, 2])
with ic1:
    st.write(
        "BCI is an adaptation of the concentration index based on the "
        "cumulative percent of total error for each county sorted by the "
        "sociodemographic variable. To calculate BCI we take the integration "
        "of the difference between the concentration curve and a cumulative "
        "uniform distribution (a 45 degree diagonal -- perfect equality): "
        "Curves with large area under the cumulative uniform distribution "
        "indicate prediction error increases with the sociodemographic "
        "variable; curves above the diagonal indicate the opposite")
with ic2:
    st.image("static/static_img_bilateral_concentration_curve.png", width=300)
with ic3:
    st.image("static/static_img_BCI_formula.png", width=300)
    st.image("static/static_img_fx_formula.png", width=300)

st.subheader("How to Use")
st.markdown(
    "Make sure your CSV file has headers. This tool allows you to select the "
    "columns that represent certain values such as true/predicted as well as "
    "error and a sociodemographic factor. Then various bias metrics will be "
    "calculated as well as graphs")
st.download_button(label="Download Sample Data",
                   data=SAMPLE_DATA.to_csv(index=False).encode("utf-8"),
                   file_name="sample_data.csv",
                   mime="text/csv")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is None:
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    st.error(f"⚠️ **Could not read the CSV file**: {e}")
    st.stop()
columns = list(df.columns)

with st.container(border=True):
    st.subheader("Column Select")
    demographic = st.selectbox(
        "Sociodemographic Factor", columns,
        help="Must be a continuous variable (e.g. median income, % with a "
        "high school diploma). Binary or categorical attributes are not "
        "supported.")
    calc_error = st.checkbox("Calculate error from predicted and true values",
                             value=True)
    use_baseline = st.checkbox(
        "Compare against a baseline model", value=False,
        help="Adds a p-value for whether this model's disparity is larger "
        "than the baseline's.")

    selectors = st.columns(3)
    pred = true = error = baseline = None
    if calc_error:
        pred = selectors[0].selectbox("Predicted Values", columns)
        true = selectors[1].selectbox("True Values", columns)
    else:
        error = selectors[0].selectbox("Error Column", columns)
    if use_baseline:
        baseline = selectors[2].selectbox(
            "Baseline Predictions" if calc_error else "Baseline Errors",
            columns)

with st.container(border=True):
    ca1, ca2 = st.columns(2)
    with ca1:
        st.header("Metrics")
        selected_metrics = ["bci"] if st.checkbox("BCI", value=True) else []
        st.header("Alternative Metrics")
        for name in ["ci", "ks", "ad"]:
            if st.checkbox(METRIC_LABELS[name], value=True):
                selected_metrics.append(name)
    with ca2:
        st.header("Graphs")
        graph_labels = {"bci": "BCI Curve", "ks": "KS Test Curve",
                        "scatter": "Scatterplots"}
        selected_graphs = [g for g, label in graph_labels.items()
                           if st.checkbox(label, value=True)]

if not st.button("Run Analysis"):
    st.stop()

if not selected_metrics and not selected_graphs:
    st.error("⚠️ Select at least one metric or graph.")
    st.stop()

with st.spinner("Computing metrics and bootstrap p-values..."):
    try:
        results, figures = run_analysis(
            df, demographic, pred=pred, true=true, error=error,
            baseline=baseline, metrics=selected_metrics,
            graphs=selected_graphs, n_resamples=N_RESAMPLES)
    except ValueError as e:
        st.error(f"⚠️ **Analysis Failed**: {e}")
        st.stop()
    except Exception as e:
        st.error("⚠️ **Analysis Failed**: There was an issue with your "
                 "column selections.")
        st.info("💡 **Please check**: \n- Use different columns for "
                "Predicted/True Values \n- Select a demographic factor with "
                "meaningful groups (not ID columns) \n- Ensure your data "
                "columns contain numeric values")
        st.info(f"**Technical details**: {e}")
        st.stop()

st.header("Bias Scores")
st.caption(f"{results['n']} rows used (rows with missing or non-numeric "
           "values in the selected columns are dropped).")
if results["metrics"]:
    st.table(results_table(results))

st.header("Graphs")
if figures:
    cols = st.columns(2)
    for i, (label, fig) in enumerate(figures):
        with cols[i % 2]:
            st.pyplot(fig)
            st.caption(label)
        plt.close(fig)
else:
    st.info("No graphs selected.")
