#variables for the columns:
#st.session_state.sdf
#st.session_state.pred
#st.session_state.true
#st.session_state.error
#st.session_state.base

import faulthandler, sys, signal

faulthandler.enable(file=sys.stderr, all_threads=True)

import os

os.environ["STREAMLIT_LOG_LEVEL"] = "debug"

import streamlit as st
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import tempfile
from auth import is_logged_in, show_login_page, show_user_info, login_required
import csv
from typing import Any, Dict, List
import pandas as _pd
from PIL import Image

options = {
    'customConcentrationIntegrateValue': "BCI",
    #'customConcentrationIntegrateValue': "BCI Integrate Value",
    'customConcentrationValueNullP': "BCI Null P",
    'customConcentrationValueOtherP': "BCI Other P",
    'giniCoefficient': "Gini Coefficient",
    'giniCoefficientP': "Gini Coefficient P",
    'ksStat': "KS Stat",
    'ksStatP': "KS Stat P",
    'andersonDarling': "Anderson Darling",
    'andersonDarlingNullP': "Anderson Darling Null P",
    'andersonDarlingOtherP': "Anderson Darling Other P"

    #'pearson': "Pearson Correlation"
}

selections = {}
graphs = {}

if st.session_state.get("trigger_rerun", False):
    st.session_state.trigger_rerun = False
    st.rerun()


def readDataFromCSV3(path: str):
    """
    Reads a simple CSV with a header row.
    Returns (headers, data_rows).
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    headers = rows[0]
    data = rows[1:]
    return headers, data


class ColumnConfig:

    def __init__(self, col_num: int, original: str, sample_value: Any):
        self.col_num = col_num
        self.original = original
        self.sample_value = sample_value
        self.p_val = None
        self.t_val = None
        self.app_sdf = None
        self.use_col = True

    @property
    def new_header(self):
        return [self.p_val, self.t_val]

    def to_dict(self) -> Dict:
        return {
            "ColNum": self.col_num,
            "Original": self.original,
            "Attributes": {
                "p_val": self.p_val,
                "t_val": self.t_val,
                "app_sdf": self.app_sdf
            },
            "useCol": self.use_col
        }



NO_BASELINE_VARIABLE= "NO_BASELINE"
ID_INSTRUCTION = "{dem_factor1[df1]}__{dem_factor2[df2]}__{demographic}__{custom}"

matches = {}

if 'dataIn' not in st.session_state:
    st.session_state.dataIn = False

st.set_page_config(page_title="AI Fairness Pipeline", layout="wide")

st.title("AI Fairness Pipeline")
st.markdown("Run bias detection analysis on machine learning models")

from PIL import Image
import requests
from io import BytesIO

st.subheader("Bilateral Concentration Index")

ic1, ic2, ic3 = st.columns([4, 2, 2])

with ic1:
    st.write(
        "BCI is adaptation of the concentration index based on the cumulative percent of total error for each county sorted by the sociodemographic variable. To calculate BCI we take the integration of the difference between the concentration curve and a cumulative uniform distribution (a 45 degree diagonal -- perfect equality): Curves with large area under the cumulative uniform distribution indicate prediction error increases with the sociodemographic variable; curves above the diagonal indicate the opposite"
    )

with ic2:
    img = Image.open("static/static_img_bilateral_concentration_curve.png")
    st.image(img, width=300)

with ic3:
    img = Image.open("static/static_img_BCI_formula.png")
    st.image(img, width=300)
    img = Image.open("static/static_img_fx_forumla.png")
    st.image(img, width=300)
#st.info('This is a purely informational message', icon="ℹ️")

st.subheader("How to Use")
st.markdown(
    "Make sure your CSV file has headers. This tool allows you to select the columns that represent certain values such as true/predicted as well as error and a sociodemographic factor. Then various bias metrics will be calculated as well as graphs"
)

data = {
    "id":
    list(range(1, 21)),
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
    ]
}

df = pd.DataFrame(data)

csv_bytes = df.to_csv(index=False).encode('utf-8')

st.download_button(label="Download Sample Data",
                   data=csv_bytes,
                   file_name="static/synthetic_dataset.csv",
                   mime="text/csv")

#x = st.file_uploader("Upload a CSV file", type="csv")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    df_cols = pd.read_csv(tmp_file_path)

    st.session_state.dataIn = True
    tests_run = {"correlation": True, "gini": True, "ks": True}

    if st.session_state.dataIn:
        #tag_button = st.button("Collect Data",
        #key="ver3_tag",
        #use_container_width=True)
        #if tag_button:
        headers, data = readDataFromCSV3(tmp_file_path)

        base_df = pd.read_csv(tmp_file_path)

        st.session_state.raw_headers = list(base_df.columns)
        st.session_state.raw_headers2 = headers
        st.session_state.raw_headers2.insert(0, "")
        st.session_state.raw_data = data
        configs = []
        sample_row = data[0] if data else []
        for i, h in enumerate(headers):
            cfg = ColumnConfig(i, h,
                               sample_row[i] if i < len(sample_row) else None)
            configs.append(cfg)
        st.session_state.configs = configs
        st.session_state.num_f = 1

        if st.session_state.get("configs"):
            #bc1, bc2 = st.columns(2)
            #with bc1:
            with st.container(border=True):
                st.subheader("Column Select")

                st.session_state.sdf = st.selectbox(
                    "Sociodemographic Factor",
                    st.session_state.raw_headers,
                    key="app_sdf")

                st.session_state.calc_error = st.checkbox("Calculate Error",
                                                          value=True)
                st.session_state.baseline = st.checkbox("Baseline", value=True)

                if st.session_state.calc_error and not st.session_state.baseline:
                    header_cols = st.columns([3, 3])
                    #header_cols[0].write("Predicted Values")
                    #header_cols[1].write("True Values")

                    st.session_state.pred = header_cols[0].selectbox(
                        "Predicted Values", st.session_state.raw_headers)
                    st.session_state.true = header_cols[1].selectbox(
                        "True Values", st.session_state.raw_headers)

                    #st.write(st.session_state.pred, st.session_state.true)

                if not st.session_state.calc_error and st.session_state.baseline:
                    header_cols = st.columns([3, 3])
                    #header_cols[0].write("Predicted Values")
                    #header_cols[1].write("True Values")

                    st.session_state.error = header_cols[0].selectbox(
                        "Error Column", st.session_state.raw_headers)
                    st.session_state.base = header_cols[1].selectbox(
                        "Baseline", st.session_state.raw_headers)

                if st.session_state.calc_error and st.session_state.baseline:
                    header_cols = st.columns([3, 3, 3])
                    #header_cols[0].write("Predicted Values")
                    #header_cols[1].write("True Values")

                    st.session_state.pred = header_cols[0].selectbox(
                        "Predicted Values", st.session_state.raw_headers)
                    st.session_state.true = header_cols[1].selectbox(
                        "True Values", st.session_state.raw_headers)
                    st.session_state.base = header_cols[2].selectbox(
                        "Baseline", st.session_state.raw_headers)

                #use selectbox

            #with bc2:
            with st.container(border=True):

                ca1, ca2 = st.columns(2)

                with ca1:
                    st.header("Metrics")
                    selections[
                        "customConcentrationIntegrateValue"] = st.checkbox(
                            "BCI", value=True)
                    st.header("Alternative Metrics")
                    #selections['pearson'] = st.checkbox("Correlation", value=True)
                    selections["giniCoefficient"] = st.checkbox(
                        "Gini Coefficient", value=True)
                    selections["ksStat"] = st.checkbox("KS Test", value=True)
                    selections["andersonDarling"] = st.checkbox(
                        "Anderson Darling", value=True)
                with ca2:
                    st.header("Graphs")

                    graphs["bci"] = st.checkbox("BCI Curve", value=True)
                    #graphs["bci"] = st.checkbox("BCI Curve", value=True)
                    graphs["ks"] = st.checkbox("KS Test Curve", value=True)
                    graphs["scatter"] = st.checkbox("Scatterplots", value=True)

                #st.header("Graphs")
                #ksG = st.checkbox("KS Test ", value=True)
                #bciG = st.checkbox("BCI ", value=True)
                #scatter = st.checkbox("Scatterplots ", value=True)

            save_and_run = st.button("Run Analysis", key="ver3_save")
            #st.write(graphs)
            if save_and_run:
                # CRITICAL: Validate checkbox combinations before proceeding
                if not st.session_state.calc_error and not st.session_state.baseline:
                    st.error("⚠️ **Invalid Configuration**: Both 'Calculate Error' and 'Baseline' are unchecked. Please select at least one option to proceed with the analysis.")
                    st.info("💡 **Tip**: Check 'Calculate Error' if you have prediction and true value columns, or check 'Baseline' if you have pre-calculated error data.")
                    st.stop()
                
                # Validate that required columns are accessible based on checkbox states
                missing_columns = []
                if st.session_state.calc_error:
                    if not hasattr(st.session_state, 'pred') or not st.session_state.pred:
                        missing_columns.append("Predicted Values")
                    if not hasattr(st.session_state, 'true') or not st.session_state.true:
                        missing_columns.append("True Values")
                
                if not st.session_state.calc_error and st.session_state.baseline:
                    if not hasattr(st.session_state, 'error') or not st.session_state.error:
                        missing_columns.append("Error Column")
                
                if st.session_state.baseline:
                    if not hasattr(st.session_state, 'base') or not st.session_state.base:
                        missing_columns.append("Baseline")
                
                if missing_columns:
                    st.error(f"⚠️ **Missing Required Columns**: Please select columns for: {', '.join(missing_columns)}")
                    st.stop()

                # CHANGE: Initialize session_id early for image generation
                if 'session_id' not in st.session_state:
                    import uuid
                    st.session_state.session_id = str(uuid.uuid4())[:8]
                    print(
                        f"DEBUG: Created new session_id early: {st.session_state.session_id}"
                    )

                matches = [[]]
                # Handle different checkbox scenarios for building matches

                if st.session_state.calc_error:
                    matches[0].append(st.session_state.pred)
                    matches[0].append(st.session_state.true)
                else:
                    # For error-only workflow, we use placeholders that will be handled later
                    matches[0].append("pred_synthetic")
                    matches[0].append("true_synthetic")
                #st.write(matches)

                #st.write(selections)

                print("DEBUG: Save Mapping & Run Analysis V3 button clicked")
                try:

                    print("DEBUG: raw_data length:",
                          len(st.session_state.raw_data))
                    raw_df = pd.DataFrame(st.session_state.raw_data,
                                          columns=st.session_state.raw_headers)
                    print("DEBUG: raw_df shape:", raw_df.shape)
                    
                    # Additional data quality validation
                    if raw_df.empty:
                        st.error("⚠️ **Empty Dataset**: The uploaded CSV file contains no data rows.")
                        st.stop()
                    
                    if len(raw_df) < 2:
                        st.error("⚠️ **Insufficient Data**: At least 2 data rows are required for analysis.")
                        st.stop()
                    
                    # Validate sociodemographic factor has multiple unique values
                    if st.session_state.sdf and st.session_state.sdf in raw_df.columns:
                        unique_demographics = raw_df[st.session_state.sdf].nunique()
                        if unique_demographics < 2:
                            st.error(f"⚠️ **Insufficient Demographic Variation**: The selected demographic factor '{st.session_state.sdf}' has only {unique_demographics} unique value(s). Bias analysis requires at least 2 different demographic groups.")
                            st.stop()


                    for m in range(len(matches)):
                        if matches[m][0] == "" or matches[m][
                                1] == "" or matches[m][1] is None or matches[
                                    m][0] is None:
                            matches.pop(m)

                    #for cfg in st.session_state.configs:
                    #st.write(cfg.t_val)
                    #st.write(cfg.use_col)

                    #print("DEBUG: matches =", matches)
                    #st.write("DEBUG: matches =", matches)
                    invalid = [m for m in matches if None in m or "" in m]
                    if invalid:
                        st.error(f"Incomplete column selection: {invalid}")
                        st.stop()

                    from aiFairnessPipeline.src.ParsePredictionsByDem import iterateOverData

                    
                    if not st.session_state.baseline:
                        #st.session_state.base = ["hi"]
                        st.session_state.base = [NO_BASELINE_VARIABLE]
                    #st.write(st.session_state.base)
                    try:
                        results_df = iterateOverData(raw_df, matches,
                                                      st.session_state.sdf,
                                                      st.session_state.base,
                                                      tests_run, graphs)
                    except Exception as e:
                        st.error("⚠️ **Analysis Failed**: There was an issue with your column selections.")
                        st.info("💡 **Please check**: \n- Use different columns for Predicted/True Values \n- Select a demographic factor with meaningful groups (not ID columns) \n- Ensure your data columns contain numeric values")
                        st.info(f"**Technical details**: {str(e)}")
                        st.stop()

                    #add comma graphs back

                    print("DEBUG: results_df shape:", results_df.shape)

                    results_df = results_df.reset_index(drop=True)

                    st.header("Bias Scores")
                    #st.table(results_df)

                    res_col_count = 0
                    for index, row in results_df.iterrows():
                        res_col_count += 1
                    res_cols = st.columns(res_col_count)

                    for i in range(len(res_cols)):
                        with res_cols[i]:
                            st.subheader(results_df["outcome"].iloc[i])

                    if st.session_state.baseline:
                        pvals = {}

                        if selections["ksStat"]:
                            pvals["ksStat"] = "ksStatP"
                        if selections["giniCoefficient"]:
                            pvals["giniCoefficient"] = "giniCoefficientP"
                        if selections["customConcentrationIntegrateValue"]:
                            pvals[
                                "customConcentrationIntegrateValue"] = "customConcentrationValueNullP"
                        if selections["andersonDarling"]:
                            pvals["andersonDarling"] = "andersonDarlingNullP"

                    #for key in options:
                    #if key in selections and selections[key]:
                    #for index, row in results_df.iterrows():
                    #with res_cols[index]:
                    #st.metric(label=options[key],
                    #value=round(row[key], 3))
                    #st.write("P Value:", row[pvals[key]])

                    selected_keys = [
                        key for key in options
                        if key in selections and selections[key]
                    ]

                    data = []
                    for index, row in results_df.iterrows():
                        # Row for metrics
                        row_data = {}
                        for key in selected_keys:
                            metric_label = options[key]
                            # Handle "N/A" values (e.g., Gini coefficient for synthetic data)
                            if row[key] == "N/A":
                                metric_val = "N/A"
                            else:
                                metric_val = round(row[key], 3)
                            row_data[metric_label] = metric_val
                        data.append(row_data)

                        # Row for p-values (if baseline is enabled)
                        if st.session_state.baseline:
                            p_row_data = {}
                            for key in selected_keys:
                                metric_label = options[key]
                                p_val = row[pvals[key]]
                                # Handle "N/A" p-values (e.g., Gini coefficient for synthetic data)
                                if p_val == "N/A":
                                    p_row_data[metric_label] = "P = N/A"
                                else:
                                    p_row_data[
                                        metric_label] = f"P = {p_val:.3g}"  # short scientific notation if needed
                            data.append(p_row_data)

                    df = pd.DataFrame(data).reset_index(drop=True)

                    st.table(df)

                    original_dir = os.getcwd()
                    os.chdir('aiFairnessPipeline')
                    sys.path.append('.')
                    os.chdir(original_dir)

                    print("DEBUG: Analysis complete, loading graphs")
                    st.header("Graphs")

                    # CHANGE: Session ID should already be initialized at start of analysis
                    print(
                        f"DEBUG: Display using session_id: {st.session_state.get('session_id', 'None')}"
                    )

                    # CHANGE: Display multiple images from session-based output folder
                    import glob
                    import os

                    session_output_dir = f"aiFairnessPipeline/output/{st.session_state.session_id}"
                    image_files = glob.glob(f"{session_output_dir}/img*.png")
                    image_files.sort()  # ensures img1, img2, ...

                    # --- Map graph checkboxes to image order ---
                    # Assume order: BCI, KS, Scatter1, Scatter2
                    graph_order = [("bci", "BCI Curve"),
                                   ("ks", "KS Test Curve"),
                                   ("scatter", "Scatterplot 1"),
                                   ("scatter", "Scatterplot 2 (Error-based)")]

                    selected_images = []
                    for idx, ((key, label), img_file) in enumerate(
                            zip(graph_order, image_files)):
                        if not os.path.exists(img_file):
                            continue

                        # Skip second scatterplot if calc_error is False
                        if "Scatterplot 2" in label and not st.session_state.calc_error:
                            continue

                        # Only include if checkbox for this graph type is checked
                        if graphs.get(key, False):
                            selected_images.append((img_file, label))

                    # --- Display selected graphs in 2-column layout ---
                    if selected_images:
                        cols = st.columns(2)
                        for i, (img_file, label) in enumerate(selected_images):
                            col = cols[i % 2]
                            with col:
                                image = Image.open(img_file)
                                st.image(image,
                                         caption=label,
                                         use_container_width=True)
                    else:
                        st.warning("No graphs selected or generated yet.")

                except Exception as e:
                    print("DEBUG: Exception in analysis or graph block:", e)
                    st.error(f"Error saving/running V3 analysis: {e}")
                    st.code(str(e), language='text')
