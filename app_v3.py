from re import T
import streamlit as st
import subprocess
import sys
import os
import time
from pathlib import Path
import pandas as pd
import tempfile
from auth import is_logged_in, show_login_page, show_user_info, login_required
import csv
from typing import Any, Dict, List
import pandas as _pd


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
        self.def1 = None
        self.def2 = None
        self.wlang = None
        self.tval = None
        self.use_col = True

    @property
    def new_header(self) -> str:
        parts = [p for p in (self.def1, self.def2, self.wlang, self.tval) if p]

        dem_factor1 = {
            "": "",
            "Income": "control_logincomeHC01_VC85ACS3yr$10",
            "Heart Disease": "heart_disease",
            "Life Satisfaction": "life_satisfaction",
            "Percent Fair/Poor Health": "perc_fair_poor_health",
            "Suicide": "suicide"
        }

        dem_factor2 = {
            "": "",
            "Income": "logincomeHC01_VC85ACS3yr$10",
            "Heart Disease": "heart_disease",
            "Life Satisfaction": "life_satisfaction",
            "Percent Fair/Poor Health": "perc_fair_poor_health",
            "Suicide": "suicide"
        }

        x = ""
        if self.def1 is not None:
            x = f"{dem_factor1[self.def1]}"
            if self.def2 is not None:
                if dem_factor2[self.def2] != "":
                    x += f"__{dem_factor2[self.def2]}"
            if self.wlang:
                x += "__withLanguage"
            if self.tval:
                x += "__trues"

        return x if parts else self.original

    def to_dict(self) -> Dict:
        return {
            "ColNum": self.col_num,
            "Original": self.original,
            "Attributes": {
                "df1": self.def1,
                "df2": self.def2,
                "wlanguage": self.wlang,
                "trueval": self.tval
            },
            "useCol": self.use_col
        }


class UserDataSet:

    def __init__(self,
                 new_col_headers: List[str] = None,
                 data: List[List[Any]] = None):
        self.new_col_headers = new_col_headers or []
        self.data = data or []

    def to_dataframe(self):
        return _pd.DataFrame(self.data, columns=self.new_col_headers)


ID_INSTRUCTION = "{dem_factor1[df1]}__{dem_factor2[df2]}__{demographic}__{custom}"

matches = {}

if 'dataIn' not in st.session_state:
    st.session_state.dataIn = False

st.set_page_config(page_title="AI Fairness Pipeline", layout="wide")

st.title("AI Fairness Pipeline")
st.markdown("Run bias detection analysis on machine learning models")

x = st.file_uploader("Upload a CSV file", type="csv")

if x is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(x.read())
        tmp_file_path = tmp_file.name
    df_cols = pd.read_csv(tmp_file_path)

    step0_button = st.button("Collect Data", use_container_width=True)
    if step0_button:
        st.session_state.dataIn = True

    if st.session_state.dataIn:
        tag_button = st.button("Tag Data", key="ver3_tag")
        if tag_button:
            headers, data = readDataFromCSV3(tmp_file_path)
            st.session_state.raw_headers = headers
            st.session_state.raw_data = data
            configs = []
            sample_row = data[0] if data else []
            for i, h in enumerate(headers):
                cfg = ColumnConfig(
                    i, h, sample_row[i] if i < len(sample_row) else None)
                configs.append(cfg)
            st.session_state.configs = configs

        if st.session_state.get("configs"):
            st.subheader("Tag Columns V3")

            st.markdown("**ID Column will be formatted as:** "
                        f"`{ID_INSTRUCTION}`")

            header_cols = st.columns([1, 1, 2, 3, 1])
            header_cols[0].write("Col #")
            header_cols[1].write("Original Name")
            header_cols[2].write("Sample Value")
            header_cols[3].write("Attributes")
            header_cols[4].write("New Header / Use")

            for cfg in st.session_state.configs:
                c1, c2, c3, c4, c5 = st.columns([1, 1, 2, 3, 1])
                c1.write(f"#{cfg.col_num}")
                c2.write(cfg.original)
                c3.write(cfg.sample_value)

                if cfg.col_num == 0:
                    # ID column: show the instruction pattern
                    c4.markdown(f"`{ID_INSTRUCTION}`")
                    cfg.use_col = c5.checkbox("Use",
                                              value=True,
                                              key=f"use_{cfg.col_num}")

                    c5.write(f"**New:** {ID_INSTRUCTION}")
                else:
                    cfg.def1 = c4.selectbox("Demographic Factor 1", [
                        "", "Income", "Heart Disease", "Life Satisfaction",
                        "Percent Fair/Poor Health", "Suicide"
                    ],
                                            key=f"app_{cfg.col_num}")
                    cfg.def2 = c4.selectbox("Demographic Factor 2", [
                        "", "Income", "Heart Disease", "Life Satisfaction",
                        "Percent Fair/Poor Health", "Suicide"
                    ],
                                            key=f"out_{cfg.col_num}")
                    cfg.wlang = c4.checkbox("With Language",
                                            value=False,
                                            key=f"dem_{cfg.col_num}")
                    cfg.tval = c4.checkbox("True Values",
                                           value=False,
                                           key=f"dema_{cfg.col_num}")
                    cfg.custom = c4.text_input("Custom",
                                               key=f"cus_{cfg.col_num}")
                    cfg.use_col = c5.checkbox("Use",
                                              value=True,
                                              key=f"use_{cfg.col_num}")
                    c5.write(f"**New:** {cfg.new_header}")

            save_and_run = st.button("Save Mapping & Run Analysis V3",
                                     key="ver3_save")
            if save_and_run:
                try:
                    #Build mapped DataFrame
                    raw_df = pd.DataFrame(st.session_state.raw_data,
                                          columns=st.session_state.raw_headers)
                    keep_cfgs = [
                        cfg for cfg in st.session_state.configs if cfg.use_col
                    ]
                    origs = [cfg.original for cfg in keep_cfgs]
                    new_names = [cfg.new_header for cfg in keep_cfgs]
                    mapped_df = raw_df[origs].rename(
                        columns=dict(zip(origs, new_names)))

                    st.write("testing")

                    original_dir = os.getcwd()
                    os.chdir('aiFairnessPipeline')
                    sys.path.append('.')
                    from src.ParsePredictionsByDem import iterateOverData
                    os.chdir(original_dir)

                    st.dataframe(mapped_df)

                except Exception as e:
                    st.error(f"Error saving/running V3 analysis: {e}")
                    st.code(str(e), language='text')

    step1_button = st.button("Run Bias Analysis", use_container_width=True)
    st.write(matches)
    if step1_button:
        st.subheader("Results")
    output_container = st.empty()

    if step1_button:
        original_dir = os.getcwd()
        try:
            import sys
            import os

            os.chdir('aiFairnessPipeline')
            sys.path.append('.')

            from src.ParsePredictionsByDem import readDataFromCSV2, iterateOverData, readDataFromCSV
            import io
            from contextlib import redirect_stdout

            csv_file = 'features/Regression_CTLB_1grams_ControlsTested1AtATime_Oct15th_24_PaperVersion.csv'
            output_file = 'results/resultsFrom_' + csv_file.split('/')[-1]

            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                print(f"Loading data from: {csv_file}")
                dfAllRuns = readDataFromCSV(csv_file)
                print("Processing bias analysis")
                results_df = iterateOverData(dfAllRuns)

                os.makedirs('results', exist_ok=True)

                results_df.to_csv(output_file)
                print(f"Results saved to: {output_file}")

                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    print(f"Output file size: {file_size} bytes")
                    print(f"Final results shape: {results_df.shape}")
                    print("Results saved")
                else:
                    print("Warning: Output file not found")

            os.chdir(original_dir)

            output = output_buffer.getvalue()
            st.success("Results saved to CSV")

        except Exception as e:
            os.chdir(original_dir)
            st.error(f"Error in Step 3: {str(e)}")
            st.code(str(e), language='text')

        st.subheader("Bias Scores")
        res = pd.read_csv(
            "aiFairnessPipeline/results/resultsFrom_Regression_CTLB_1grams_ControlsTested1AtATime_Oct15th_24_PaperVersion.csv"
        )
        st.dataframe(res)

        st.subheader("Graphs")
        from PIL import Image
        pic_file = "aiFairnessPipeline/ConcentrationCurve.png"
        image = Image.open(pic_file)
        st.image(image, caption="Uploaded PNG", use_column_width=True)
