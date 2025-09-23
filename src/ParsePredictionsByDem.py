#Import Libraries
import sys
import os
import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest, uniform
from scipy.stats import linregress
from scipy.stats import zscore
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
from scipy.stats import norm
import math
import streamlit as st

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.stats import chi2_contingency
from collections import defaultdict
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, f1_score
import pandas as pd
import re
import ast
import textwrap

NUM_OF_CONTROLS = 1
NUM_OF_OUTCOMES = 4
CHART_TITLE_WIDTH = 32

#Set the dimensions of the graphs that are created with matplotlib
plt.rcParams.update({
    'font.size': 17,  # Adjust overall font size
    'axes.titlesize': 19,  # Title size
    'axes.titleweight': 'bold',  # Bold title
    'axes.labelsize': 14,  # Axis labels size
    'axes.labelweight': 'bold',  # Bold axis labels
    'xtick.labelsize': 14,  # X-axis tick labels
    'ytick.labelsize': 14,  # Y-axis tick labels
    'legend.fontsize': 14,  # Legend font size
    'figure.titlesize': 20,  # Figure title size
    'axes.linewidth': 2
})

fig, axes = plt.subplots(NUM_OF_CONTROLS * NUM_OF_OUTCOMES,
                         6,
                         figsize=(40, 60))

#Global variables for testing different approaches (you shouldnt need to edit these)
task = "Regression"
IS_REGRESSION = True
CSV_FILE = "Regression_CTLB_1grams_ControlsTested1AtATime_Oct15th_24_PaperVersion.csv"
num_bins = 3
ZSCORE = False
BOOTSTRAP_COUNT = 1000
NUM_OF_CONTROLS = 1
NUM_OF_OUTCOMES = 4  #5

#cleaned, shortened names for printing more legible names in results
cleanNames = {
    ("Regression Test", "language"):
    "Language Only",
    ("Regression Test", "demographic"):
    "Control Only",
    ("Regression Test", "demographic_and_language"):
    "Language and Control",
    ("Residualized Controls Regression Test", "demographic_and_language"):
    "Residualized Controls",
    ("Factor Adaptation Regression Test", "demographic_and_language"):
    "Factor Adaptation",
    ("Residualized Factor Adaptation Regression Test", "demographic_and_language"):
    "Residualized Factor Adaptation",
    "logincomeHC01_VC85ACS3yr$10":
    "Income",
    "hsgradHC03_VC93ACS3yr$10":
    "HS Graduation",
    "forgnbornHC03_VC134ACS3yr$10":
    "Foreign Born",
    "heart_disease":
    "Heart Disease",
    "life_satisfaction":
    "Life Satisfaction",
    "perc_fair_poor_health":
    "Fair Health",
    "suicide":
    "Suicide"
}

#All the combinations of approaches to run
approachesToRun = [
    ("Regression Test", "language"),
    #("Regression Test", "demographic"),
    #("Regression Test", "demographic_and_language"),
    #("Residualized Controls Regression Test", "demographic_and_language"),
    ("Factor Adaptation Regression Test", "demographic_and_language"),
    #("Residualized Factor Adaptation Regression Test", "demographic_and_language")
]

#The baseline approach to compare to
baseline = ("Regression Test", "language")


# main not called by streamlit
def main():
    dfAllRuns = readDataFromCSV3('../features/' + CSV_FILE)
    results_df = iterateOverData(dfAllRuns)
    results_df.to_csv('../results/resultsFrom_' + CSV_FILE)


def iterateOverData(dfAllRuns, matches, control, base, tests_run, graphs):
    #if "hi" in base:
    #st.write("work")
    #st.write("base", base)
    NO_BASELINE_VARIABLE= "NO_BASELINE"
    plot_data = []
    results = []

    # If user data doesn't have 'Id' column, create one based on index
    if 'Id' not in dfAllRuns.columns:
        dfAllRuns = dfAllRuns.reset_index(drop=True)
        dfAllRuns['Id'] = dfAllRuns.index
        print("DEBUG: Created 'Id' column for user data using row index")

    # NEW: Check if user provided error data and create synthetic pred/true values
    # This happens when user unchecks "Calculate Error" and provides error column
    error_column_name = getattr(st.session_state, 'error', None)
    if error_column_name and error_column_name in dfAllRuns.columns:
        print(
            f"DEBUG: Creating synthetic pred/true from provided error column: {error_column_name}"
        )

        # Check for and handle any NaN values in the error column
        error_series = dfAllRuns[error_column_name]
        print(f"DEBUG: Error column data type: {error_series.dtype}")
        print(f"DEBUG: Error column has NaN: {error_series.isna().any()}")
        print(f"DEBUG: Error column shape: {error_series.shape}")
        print(f"DEBUG: First few error values: {error_series.head().tolist()}")

        # Convert to numeric and handle any non-numeric values
        try:
            error_series = pd.to_numeric(error_series, errors='coerce')
            print(
                f"DEBUG: After pd.to_numeric conversion, NaN count: {error_series.isna().sum()}"
            )
        except Exception as e:
            print(f"DEBUG: Error in numeric conversion: {e}")

        # Create synthetic pred and true columns that reproduce the exact error values
        baseline = 0.0  # Simple baseline approach
        dfAllRuns['true_synthetic'] = baseline
        dfAllRuns['pred_synthetic'] = baseline + error_series

        # Update matches to use synthetic columns instead of original selections
        print(f"DEBUG: Original matches: {matches}")
        for match in matches:
            if len(match) >= 2:
                match[
                    0] = 'pred_synthetic'  # Replace pred column with synthetic
                match[
                    1] = 'true_synthetic'  # Replace true column with synthetic
        print(f"DEBUG: Updated matches to use synthetic columns: {matches}")

        # Verification
        verification_errors = [
            abs(p - t) for p, t in zip(dfAllRuns['pred_synthetic'][:3],
                                       dfAllRuns['true_synthetic'][:3])
        ]
        original_errors = error_series[:3].tolist()
        print(f"DEBUG: Verification - synthetic errors: {verification_errors}")
        print(f"DEBUG: Verification - original errors: {original_errors}")

    # Loop over all control/outcome/approach combinations
    for outcome_list in matches:
        print("\n# DEBUG: Total outcome pairs:", len(matches))
        print("# DEBUG: Outcome pairs:", matches)
        #Get a dataframe with columns (pred, true, demographic_val, base) to perform calculations with

        pred_df = _loadApproachColumn(dfAllRuns, outcome_list[0], 'pred')
        print("\n# DEBUG: Processing outcome pair:", outcome_list)
        true_df = _loadApproachColumn(dfAllRuns, outcome_list[1], 'true')
        cont_df = _loadApproachColumn(dfAllRuns, control, 'demographic_val')
        if "hi" not in base:
            base_df = _loadApproachColumn(dfAllRuns, base, 'base')

        cont_bins_df = labelBins(cont_df, 'demographic_val')
        if NO_BASELINE_VARIABLE not in base:
            #st.write("making df with all")
            combined_df = pd.merge(pd.merge(
                pd.merge(pred_df, base_df, on='Id', how='inner'),
                cont_bins_df[['Id', 'bin', 'demographic_val']],
                on='Id',
                how='inner'),
                                   true_df,
                                   on='Id',
                                   how='inner').dropna()
        else:
            combined_df = pd.merge(pd.merge(
                pred_df,
                cont_bins_df[['Id', 'bin', 'demographic_val']],
                on='Id',
                how='inner'),
                                   true_df,
                                   on='Id',
                                   how='inner').dropna()

        #st.write("checkpoint: made combined df")

        print("# DEBUG: Combined DF shape:", combined_df.shape)
        print("# DEBUG: Columns in combined DF:", combined_df.columns.tolist())


        #Zscore the data if desired
        if ZSCORE:
            #st.write("testing Z")
            combined_df['pred'] = zscore(combined_df['pred'])
            combined_df['true'] = zscore(combined_df['true'])

        import numpy as np

        def gini_coefficient1(x):
            """
            Compute the Gini coefficient of a numpy array or list.
            """
            x = np.array(x)
            if x.size == 0:
                return np.nan
            sorted_x = np.sort(x)
            n = x.size
            cumulative_x = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cumulative_x) / cumulative_x[-1]) / n

        # Check if we're using synthetic data (all true values identical)
        # When calc_error=False, true_synthetic column has all identical baseline values
        # Gini coefficient is meaningless for identical values, so skip calculation
        if len(combined_df["true"].dropna().unique()) <= 1:
            # All true values are identical (synthetic data scenario)
            giniCoefficient = "N/A"
            print("DEBUG: Skipping Gini coefficient - synthetic data with identical true values")
        else:
            # Normal calculation for real prediction/true data
            giniCoefficient = gini_coefficient1(combined_df["true"].dropna())
        #st.write("checkpoint: made gini")

        #st.write("gini: ", giniCoefficient)
        if NO_BASELINE_VARIABLE not in base and giniCoefficient != "N/A":
            giniCoefficient_p = bootstrapResampleBoth(
                combined_df,
                disp_metric=lambda x: discreteGiniCoefficient(x),
                bins=False,
                compareWithNull=True)
        else:
            giniCoefficient_p = "N/A"
        #st.write("gini p: ", giniCoefficient_p)


        absolute_diff = [
            abs(a - b) for a, b in zip(list(combined_df["pred"]),
                                       list(combined_df["true"]))
        ]

        #st.write("abs diff", absolute_diff)
        sorted_values = [
            v for _, v in sorted(
                zip(list(combined_df["demographic_val"]), absolute_diff))
        ]
        #st.write("sorted vals", sorted_values)

        #cumulative_share_of_population = np.linspace(0, 1, len(combined_df)+1)
        ksStat, ksStat_p = kstest(sorted_values, uniform.cdf, args=(0, 1))

        ksStat1 = calcMetricOnFullData(list(combined_df["pred"]),
                                       list(combined_df["true"]),
                                       list(combined_df["demographic_val"]),
                                       disp_metric=lambda x: KsTest(x))
        if NO_BASELINE_VARIABLE not in base:
            ksStat_p = bootstrapResampleBoth(combined_df,
                                             disp_metric=lambda x: KsTest(x),
                                             bins=False,
                                             compareWithNull=True)

        #likelihood = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcLikelihood(x))
        #likelihood_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: calcLikelihood(x), bins = False, compareWithNull=False)

        #JensenShannon = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcJensenShannon(x))
        #JensenShannon_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: calcJensenShannon(x), bins = False, compareWithNull=False)

        #chiSquared = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcChiSquared(x))
        #chiSquared_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: calcChiSquared(x), bins = False, compareWithNull=False)
        #chiSquaredBin, chiSquaredBin_p = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcChiSquared(x), bin_ids=list(combined_df["bin"]))#, internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))))
        #chiSquaredBin_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: 1-minMaxRatio(x), bins = True, compareWithNull=True)

        #ksTest = calcMetricOnFullData(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: KsTest(x))
        # ksTestNull_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: KsTest(x), bins = False, compareWithNull=True)
        # ksTestOther_p = bootstrapResampleBoth(combined_df, disp_metric=lambda x: KsTest(x), bins = False, compareWithNull=False)

        customConcentrationValue = calcMetricOnFullData(
            list(combined_df["pred"]),
            list(combined_df["true"]),
            list(combined_df["demographic_val"]),
            disp_metric=lambda x: npConcentrationCoefficient(x))
        if NO_BASELINE_VARIABLE not in base:
            customConcentrationValueNull_p = bootstrapResampleBoth(
                combined_df,
                disp_metric=lambda x: npConcentrationCoefficient(x),
                bins=False,
                compareWithNull=True)
            customConcentrationValueOther_p = bootstrapResampleBoth(
                combined_df,
                disp_metric=lambda x: npConcentrationCoefficient(x),
                bins=False,
                compareWithNull=False)
        customConcentrationIntegrateValue = calcMetricOnFullData(
            list(combined_df["pred"]),
            list(combined_df["true"]),
            list(combined_df["demographic_val"]),
            disp_metric=lambda x: npConcentrationCoefficientIntegrate(x))

        andersonDarling = calcMetricOnFullData(
            list(combined_df["pred"]),
            list(combined_df["true"]),
            list(combined_df["demographic_val"]),
            disp_metric=lambda x: calcAndersonDarling(x))
        if NO_BASELINE_VARIABLE not in base:
            andersonDarlingNull_p = bootstrapResampleBoth(
                combined_df,
                disp_metric=lambda x: calcAndersonDarling(x),
                bins=False,
                compareWithNull=True)
            andersonDarlingOther_p = bootstrapResampleBoth(
                combined_df,
                disp_metric=lambda x: calcAndersonDarling(x),
                bins=False,
                compareWithNull=False)

        andysMetricBin = calcMetricOnBins(
            list(combined_df["pred"]),
            list(combined_df["true"]),
            list(combined_df["demographic_val"]),
            disp_metric=lambda x: calcAndyDeviation(x),
            bin_ids=list(combined_df["bin"]),
            internal_metric=lambda x, y: np.mean(
                np.abs(np.array(x) - np.array(y))))
        if NO_BASELINE_VARIABLE not in base:
            andysMetricBinNull_p = bootstrapResampleBoth(
                combined_df,
                internal_metric=lambda x, y: np.mean(
                    np.abs(np.array(x) - np.array(y))),
                disp_metric=lambda x: calcAndyDeviation(x),
                bins=True,
                compareWithNull=True)
            andysMetricBinOther_p = bootstrapResampleBoth(
                combined_df,
                internal_metric=lambda x, y: np.mean(
                    np.abs(np.array(x) - np.array(y))),
                disp_metric=lambda x: calcAndyDeviation(x),
                bins=True,
                compareWithNull=False)

        # crossEntropyBin = calcMetricOnBins(list(combined_df["pred"]), list(combined_df["true"]), list(combined_df["demographic_val"]), disp_metric=lambda x: calcCrossEntropy(x), bin_ids=list(combined_df["bin"]), internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))))
        # crossEntropyBinNull_p = bootstrapResampleBoth(combined_df, internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))), disp_metric=lambda x: calcCrossEntropy(x), bins = True, compareWithNull=True)
        # crossEntropyBinOther_p = bootstrapResampleBoth(combined_df, internal_metric=lambda x, y: np.mean(np.abs(np.array(x) - np.array(y))), disp_metric=lambda x: calcCrossEntropy(x), bins = True, compareWithNull=False)

        correlation = calculateCorrelation(combined_df[["true", "pred"]])
        #st.write("corr", correlation)

        #binCorrelations = calcMetricOnBins(list(combined_df["pred"]),
        #list(combined_df["true"]),
        #list(
        #combined_df["demographic_val"]),
        #disp_metric=lambda x: x,
        #bin_ids=list(combined_df["bin"]))

        #Save results to a dictionary
        if NO_BASELINE_VARIABLE not in base:
            result = {
                'control': control,
                'outcome': outcome_list[0],
                #'approach': approach
                #'binCorrelations' : binCorrelations,
                #'inverseParity':inverseParityRatio,
                #'inverseParityP': inverseParityRatio_p,
                'giniCoefficient': giniCoefficient,
                'giniCoefficientP': giniCoefficient_p,
                #'concentrationCurveAbs': concentrationCurveAbs,
                #'concentrationCurveSum': concentrationCurveSum,
                #'concentrationCurveP': concentrationCurve_p
                'ksStat': ksStat,
                'ksStatP': ksStat_p,
                #'likelihood': likelihood,
                #'likelihoodP': likelihood_p
                #'JensenShannon': JensenShannon,
                #'JensenShannonP': JensenShannon_p,
                #'chiSquared': chiSquaredBin,
                #'chiSquaredBinP': chiSquaredBin_p,
                'andersonDarling': andersonDarling,
                'andersonDarlingNullP': andersonDarlingNull_p,
                'andersonDarlingOtherP': andersonDarlingOther_p,
                'andysMetricBin': andysMetricBin,
                'andysMetricBinOtherP': andysMetricBinOther_p,
                'andysMetricBinNullP': andysMetricBinNull_p,
                # 'ksTest': ksTest,
                # 'ksTestNullP': ksTestNull_p,
                # 'ksTestOtherP': ksTestOther_p,
                'customConcentrationValue': customConcentrationValue,
                'customConcentrationIntegrateValue':
                customConcentrationIntegrateValue,
                'customConcentrationValueNullP':
                customConcentrationValueNull_p,
                'customConcentrationValueOtherP':
                customConcentrationValueOther_p
                #'crossEntropyValue': crossEntropyBin,
                #'crossEntropyValueNullP': crossEntropyBinNull_p,
                #'crossEntropyValueOtherP': crossEntropyBinOther_p
            }
        else:
            result = {
                'control':
                control,
                'outcome':
                outcome_list[0],
                #'approach': approach
                #'binCorrelations' : binCorrelations,
                #'inverseParity':inverseParityRatio,
                #'inverseParityP': inverseParityRatio_p,
                'giniCoefficient':
                giniCoefficient,
                #'giniCoefficientP': giniCoefficient_p,
                #'concentrationCurveAbs': concentrationCurveAbs,
                #'concentrationCurveSum': concentrationCurveSum,
                #'concentrationCurveP': concentrationCurve_p
                'ksStat':
                ksStat,
                #'ksStatP': ksStat_p,
                #'likelihood': likelihood,
                #'likelihoodP': likelihood_p
                #'JensenShannon': JensenShannon,
                #'JensenShannonP': JensenShannon_p,
                #'chiSquared': chiSquaredBin,
                #'chiSquaredBinP': chiSquaredBin_p,
                'andersonDarling':
                andersonDarling,
                #'andersonDarlingNullP': andersonDarlingNull_p,
                #'andersonDarlingOtherP': andersonDarlingOther_p,
                #'andysMetricBin': andysMetricBin,
                #'andysMetricBinOtherP': andysMetricBinOther_p,
                #'andysMetricBinNullP': andysMetricBinNull_p,
                # 'ksTest': ksTest,
                # 'ksTestNullP': ksTestNull_p,
                # 'ksTestOtherP': ksTestOther_p,
                'customConcentrationValue':
                customConcentrationValue,
                'customConcentrationIntegrateValue':
                customConcentrationIntegrateValue,
                #'customConcentrationValueNullP': customConcentrationValueNull_p,
                #'customConcentrationValueOtherP': customConcentrationValueOther_p
                #'crossEntropyValue': crossEntropyBin,
                #'crossEntropyValueNullP': crossEntropyBinNull_p,
                #'crossEntropyValueOtherP': crossEntropyBinOther_p
            }
        results.append({**result, **correlation})
        plot_data.append((combined_df, control, outcome_list[0]))
        print("# DEBUG: Number of charts queued so far:", len(plot_data))

        #print(approach)
        #print(control, outcome)

    #st.write("plot data", plot_data)

    #Run code to generate plots
    #makePlotGrid2(plot_data)
    # CHANGE: Pass session_id from streamlit session state for separate user folders
    session_id = st.session_state.get('session_id', None)
    makePlotGrid3_separate(plot_data, graphs, session_id)
    #makePlotGrid(plot_data)
    print("# DEBUG: Final number of charts:", len(plot_data))

    #Save all those results dictionaries we created into a pandas dataframe
    results_df = pd.DataFrame(results)
    #results_df = results_df.pivot_table(index=['control', 'outcome'],
    #columns=['approach'],
    #values=None,
    #aggfunc='first')
    #level1, level2 = results_df.columns.levels
    #level2_sorted = sorted(level2,
    #key=lambda x: approachesToRun.index(x)
    #if x in approachesToRun else float('inf'))
    #sorted_columns = pd.MultiIndex.from_product([level1, level2_sorted],
    #names=results_df.columns.names)
    #results_df = results_df.reindex(columns=sorted_columns)
    return results_df
    print("# DEBUG: Final results_df shape:", results_df.shape)


def _loadApproachColumn(df, approach_column, new_col_name):
    subset = df[['Id', approach_column]].copy()
    subset[approach_column] = pd.to_numeric(subset[approach_column],
                                            errors='coerce')
    subset.columns = ['Id', new_col_name]
    return subset


#load the column for a specific approach from the total dataframe
#Rounds a number to 2 significant figures without scientific notation.
def round_to_2_sig_figs(x):
    if x == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(x))))  # Order of magnitude
    factor = 10**(magnitude - 1)  # Scale factor for two significant digits
    rounded = round(x / factor, 1) * factor  # Round and rescale
    return "{:.0f}".format(rounded) if magnitude >= 2 else "{:.1f}".format(
        rounded)


#Make grid of plots based on prediction data
def plotScatterplot(ax, combined_df, control, outcome, palette, unique_bins):

    combined_df["error"] = abs(combined_df["pred"] - combined_df["true"])

    bin_colors = {bin: palette[i] for i, bin in enumerate(unique_bins)}

    # Compute percentiles for trimming
    x_min, x_max = combined_df["demographic_val"].quantile([0.05, 0.95])
    y_min, y_max = combined_df["error"].quantile([0.05, 0.95])

    # Filter the data
    filtered_df = combined_df[(combined_df["demographic_val"] >= x_min)
                              & (combined_df["demographic_val"] <= x_max) &
                              (combined_df["error"] >= y_min) &
                              (combined_df["error"] <= y_max)]

    sns.scatterplot(x="demographic_val",
                    y="error",
                    hue="bin",
                    data=filtered_df,
                    ax=ax,
                    palette=bin_colors,
                    alpha=.9,
                    legend=False,
                    s=18,
                    edgecolor='none',
                    hue_order=unique_bins)

    sns.regplot(x="demographic_val",
                y="error",
                data=filtered_df,
                ax=ax,
                scatter=False,
                lowess=True,
                color='black',
                line_kws={'lw': 5})
    #title = "Predicting " + cleanNames[outcome] + " Using " + cleanNames[approach] + " With Respect To " + cleanNames[control]
    title = "Prediction Error"
    wrapped_title = "\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH))
    # Reduce the number of ticks to 3
    ax.set_title(wrapped_title)
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))

    # Format tick labels to two decimal places
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))
    ax.set_xlabel(control, labelpad=-1)
    ax.set_ylabel('Absolute Error', labelpad=-3)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=palette[0], label='Low'),
        Patch(facecolor=palette[1], label='Medium'),
        Patch(facecolor=palette[2], label='High')
    ]

    legend = ax.legend(handles=legend_elements,
                       title='Bin',
                       loc='upper right',
                       frameon=True,
                       edgecolor='gray',
                       fancybox=True,
                       framealpha=0.9)

    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_boxstyle('round,pad=0.3')


#Plot scatterplot with axes true vs model prediction for each datapoint
def plotScatterplotPredVsTrues(ax, combined_df, control, outcome, palette,
                               unique_bins):

    # Compute percentiles for trimming
    x_min, x_max = combined_df["pred"].quantile([0.025, 0.975])
    y_min, y_max = combined_df["true"].quantile([0.025, 0.975])

    # Filter the data
    filtered_df = combined_df[(combined_df["pred"] >= x_min)
                              & (combined_df["pred"] <= x_max) &
                              (combined_df["true"] >= y_min) &
                              (combined_df["true"] <= y_max)]

    #draw scatter
    bin_colors = {bin: palette[i] for i, bin in enumerate(unique_bins)}
    sns.scatterplot(x="pred",
                    y="true",
                    hue="bin",
                    data=filtered_df,
                    ax=ax,
                    palette=bin_colors,
                    alpha=.5,
                    legend=False,
                    s=18,
                    edgecolor='none',
                    hue_order=unique_bins)

    #Draw linear regressions for each split
    for bin, sub_df in filtered_df.groupby("bin"):

        slope, intercept, r_value, p_value, std_err = linregress(
            sub_df["pred"], sub_df["true"])

        transparent_color = mcolors.to_rgba(bin_colors[bin], alpha=0.4)
        sns.regplot(
            x="pred",
            y="true",
            data=sub_df,
            ax=ax,
            scatter=False,  # Hide points
            color=transparent_color,  # Use bin-specific color
            line_kws={
                'lw': 5,
                'alpha': 1
            },
            ci=None)

    #Draw the axes and title for the graph
    title = "True vs Predicted"
    wrapped_title = "\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH))
    ax.set_title(wrapped_title)
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))
    ax.set_xlabel('Predicted', labelpad=-1)
    ax.set_ylabel('True', labelpad=-3)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=palette[0], label='Low'),
        Patch(facecolor=palette[1], label='Medium'),
        Patch(facecolor=palette[2], label='High')
    ]

    legend = ax.legend(handles=legend_elements,
                       title='Bin',
                       loc='upper right',
                       frameon=True,
                       edgecolor='gray',
                       fancybox=True,
                       borderpad=0.5,
                       framealpha=0.9)

    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_boxstyle('round,pad=0.3')


#Plot the concenctration curve/lorenz curve
def plotConcentrationCurve(ax, combined_df, control, outcome, absolute=False):

    ypreds = combined_df["pred"]
    ytrues = combined_df["true"]
    demographics = combined_df["demographic_val"]
    absolute_diff = [abs(a - b) for a, b in zip(ypreds, ytrues)]

    rs = [v for _, v in sorted(zip(demographics, absolute_diff))]
    n = len(rs)

    cumulative_share_of_population = np.linspace(0, 1, n + 1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    sumDeviationArea = npConcentrationCoefficientIntegrate(rs)

    # Plot the Lorenz curve
    lorenz_label = 'BCI: {:.1f}%'.format(sumDeviationArea * 100)
    ax.plot(cumulative_share_of_population,
            cumulative_share_of_income,
            label=lorenz_label,
            color='blue')
    ax.plot([0, 1], [0, 1],
            label='Line of Equality',
            color='red',
            linestyle='--')
    ax.fill_between(cumulative_share_of_population,
                    cumulative_share_of_income,
                    cumulative_share_of_population,
                    color='blue',
                    alpha=0.2)

    # title = "Concentration Curve of Error of {} vs {} (Summed Deviation Area: {:.3f} Absolute Deviation Area: {:.3f})".format(
    #     cleanNames[approach], cleanNames[control], sumDeviationArea, absoluteDeviationArea
    # )
    title = " BCI"
    wrapped_title = "\n".join(textwrap.wrap(title, width=CHART_TITLE_WIDTH))
    ax.set_title(wrapped_title)
    # Reduce the number of ticks to 3
    ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 3))

    # Format tick labels to two decimal places
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: round_to_2_sig_figs(y)))
    ax.set_xlabel('Cumulative proportion of counties ordered by \n' + control)
    ax.set_ylabel('Cumulative % of error predicting \n' + outcome, labelpad=-5)
    ax.legend()





import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st


def makePlotGrid3_separate(plot_data, graphs, session_id=None):
    enabled_graphs = []
    if graphs["bci"]:
        enabled_graphs.append("bci")
    if graphs["ks"]:
        enabled_graphs.append("ks")
    if graphs["scatter"]:
        enabled_graphs.append("scatter")

    if not enabled_graphs:
        st.warning("No graphs selected for display")
        return

    # Unique (df, control, outcome) pairs
    unique_pairs = []
    seen_pairs = set()
    for df, control, outcome in plot_data:
        pair_key = (control, outcome)
        if pair_key not in seen_pairs:
            unique_pairs.append((df, control, outcome))
            seen_pairs.add(pair_key)

    if not unique_pairs:
        st.error("No unique pairs found in plot_data")
        return

    # CHANGE: Create session-based output folder
    import os
    print(f"DEBUG: session_id={session_id}")
    if session_id:
        output_dir = f"aiFairnessPipeline/output/{session_id}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"DEBUG: Created session folder: {output_dir}")
    else:
        output_dir = "."
        print("DEBUG: Using root directory for images")

    palette = sns.color_palette("deep", n_colors=3)
    img_counter = 1  # start naming from img1

    for set_idx, (combined_df, control, outcome) in enumerate(unique_pairs):
        unique_bins = sorted(combined_df["bin"].unique())

        if graphs["bci"]:
            fig, ax = plt.subplots(figsize=(7, 6))
            plotConcentrationCurve(ax, combined_df, control, outcome)
            ax.set_title(f"BCI")
            plt.savefig(f"{output_dir}/img{img_counter}.png",
                        dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            img_counter += 1

        if graphs["ks"]:
            fig, ax = plt.subplots(figsize=(7, 6))
            plotKSCurve(ax, combined_df, control, outcome)
            ax.set_title(f"KS")
            plt.savefig(f"{output_dir}/img{img_counter}.png",
                        dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            img_counter += 1

        if graphs["scatter"]:
            fig, ax = plt.subplots(figsize=(7, 6))
            plotScatterplot(ax, combined_df, control, outcome, palette,
                            unique_bins)
            ax.set_title(f"Scatter (Error)")
            plt.savefig(f"{output_dir}/img{img_counter}.png",
                        dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            img_counter += 1

            fig, ax = plt.subplots(figsize=(7, 6))
            plotScatterplotPredVsTrues(ax, combined_df, control, outcome,
                                       palette, unique_bins)
            ax.set_title(f"Scatter (True vs Pred)")
            plt.savefig(f"{output_dir}/img{img_counter}.png",
                        dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            img_counter += 1




#Plot the lorenz curve with ks dotted line
def plotKSCurve(ax, combined_df, control, outcome, absolute=False):

    ypreds = combined_df["pred"]
    ytrues = combined_df["true"]
    demographics = combined_df["demographic_val"]
    absolute_diff = [abs(a - b) for a, b in zip(ypreds, ytrues)]

    rs = [v for _, v in sorted(zip(demographics, absolute_diff))]
    n = len(rs)

    cumulative_share_of_population = np.linspace(0, 1, n + 1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    ks_distances = np.abs(cumulative_share_of_income -
                          cumulative_share_of_population)
    max_ks_index = np.argmax(ks_distances)
    max_ks_x = cumulative_share_of_population[max_ks_index]
    max_ks_y_lorenz = cumulative_share_of_income[max_ks_index]
    max_ks_y_equality = max_ks_x

    ax.plot(cumulative_share_of_population,
            cumulative_share_of_income,
            color='gray')
    ax.plot([0, 1], [0, 1],
            label='Line of Equality',
            color='red',
            linestyle='--')
    ax.fill_between(cumulative_share_of_population,
                    cumulative_share_of_income,
                    cumulative_share_of_population,
                    color='gray',
                    alpha=0.2)
    '''# Plot KS Test against Uniform(0,1)
    #sorted_values = np.sort(absolute_diff)  # Sort values
    ecdf = np.arange(1, len(rs) + 1) / len(rs)  # Empirical CDF
    uniform_cdf = uniform.cdf(rs, loc=np.average(rs), scale=0.01)  # Theoretical CDF (Uniform)

    ksStat, _ = kstest(rs, uniform.cdf, args=(0, 1))  # KS test

    # Plot empirical CDF
    ax.plot(cumulative_share_of_population, cumulative_share_of_income, label="Empirical CDF", color="green")
    ax.plot([0, 1], [0, 1], label='Line of Equality', color='red', linestyle='--')
    #ax.plot(cumulative_share_of_population, uniform_cdf, label="Uniform(0,1) CDF", color="orange", linestyle="--")'''

    # Highlight KS Statistic (max difference)
    ax.vlines(max_ks_x,
              max_ks_y_lorenz,
              max_ks_y_equality,
              color='blue',
              linestyle='dashed',
              linewidth=2,
              label='KS Distance: {:.3f}'.format(KsTest(rs)))

    #max_diff_idx = np.argmax(np.abs(ecdf - uniform_cdf))
    #ax.vlines(rs[max_diff_idx], uniform_cdf[max_diff_idx], ecdf[max_diff_idx], colors='black', linestyle='dotted', label='KS Stat: {:.3f}'.format(ksStat))

    ax.set_title("KS Test Curve")
    ax.set_xlabel('Cumululative proportion of counties')
    ax.set_ylabel('Cumululative predicted error', labelpad=-5)
    ax.legend()


# Perform KsTest
def KsTest(rs):
    n = len(rs)

    cumulative_share_of_population = np.linspace(0, 1, n + 1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    ks_distances = np.abs(cumulative_share_of_income -
                          cumulative_share_of_population)
    max_ks_index = np.argmax(ks_distances)

    return ks_distances[max_ks_index]







# correct residual data from csv loading
def _correctResiduals(df):
    for index, row in df[df['test'].str.contains('Residual',
                                                 na=False)].iterrows():
        #print(float(df.at[index, 'demographic']))
        if (df.at[index, 'demographic']):
            df.at[index, 'demographic_and_language'] = float(
                df.at[index, 'demographic_and_language']) + float(
                    df.at[index, 'demographic'])
            df.at[index, 'true'] = float(df.at[index, 'true']) + float(
                df.at[index, 'demographic'])
    return df


# calculate correlations for results
def calculateCorrelation(df):
    yPred = df['pred']
    yTrue = df['true']

    mse = mean_squared_error(yTrue, yPred)
    r_squared = r2_score(yTrue, yPred)
    pearson_corr, pearson_p = pearsonr(yTrue, yPred)
    spearman_corr, spearman_p = spearmanr(yTrue, yPred)

    results = {
        'length': len(yTrue),
        'mse': mse,
        'rSquared': r_squared,
        'pearson': pearson_corr,
        'pearsonP': pearson_p,
        'spearman': spearman_corr,
        'spearmanP': spearman_p
    }
    return results


def gini_coefficient(rs):
    # Ensure incomes are sorted
    rs = np.sort(rs)
    # Number of incomes
    n = rs.size
    # Cumulative sum of incomes
    cumulative_rs = np.cumsum(rs, dtype=float)
    # Gini coefficient formula
    gini = (2 / n) * (np.sum(
        (np.arange(1, n + 1) * rs)) / cumulative_rs[-1]) - (n + 1) / n
    return gini


def calcLikelihood(rs):

    # Compute mean and standard deviation of the data
    mu = np.mean(rs)
    sigma = np.std(rs)  #mu / 3

    # Compute the likelihood
    log_likelihood = 0
    for x in rs:
        log_likelihood += np.log(1 / (sigma * np.sqrt(2 * np.pi))) - 0.5 * (
            (x - mu) / sigma)**2
        #print("TEST: ", np.log(1 / (sigma * np.sqrt(2 * np.pi))) - 0.5 * ((x - mu) / sigma) ** 2)
        #print("X: ", x)
        #print("TEST: ", (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))

    return log_likelihood


def calcChiSquared(rs):
    expected = np.full_like(
        rs, 1 / len(rs),
        dtype=float)  #np.full_like(rs, np.mean(rs), dtype=float)
    rs = rs / np.sum(rs)
    #print("MEAN: ", rs)

    # Run the Chi-Squared test
    chi2, p, dof = chi2_contingency([rs, expected])[:3]

    # Output results
    print("Chi-Squared Statistic:", chi2)
    print("P-value:", p)
    print("Degrees of Freedom:", dof)

    # Interpret the result
    if p < 0.05:
        print("There is a significant difference between the counties.")
    else:
        print("There is no significant difference between the counties.")

    return chi2, p


def calcAndersonDarling(rs):
    n = len(rs)

    theoretical_cdf = np.linspace(0, 1, n + 1)
    empirical_cdf = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    weighted_diff = (empirical_cdf -
                     theoretical_cdf)**2 / (theoretical_cdf *
                                            (1 - theoretical_cdf))

    # Handle potential division by zero at the edges
    weighted_diff = np.nan_to_num(weighted_diff,
                                  nan=0.0,
                                  posinf=0.0,
                                  neginf=0.0)

    a_squared = n * np.sum(weighted_diff)

    return a_squared


def calcAndyDeviation(rs):
    rs = rs / np.sum(rs)  #percent of total county that the bin is
    mu = 1 / len(rs)  # expected
    product = 1
    for x in rs:
        product *= x * mu

    return product


def calcCrossEntropy(rs):
    rs = rs / np.sum(rs)  #percent of total county that the bin is
    mu = 1 / len(rs)  # expected
    product = 1
    for x in rs:
        product *= math.log(x) * mu

    return product


def calcJensenShannon(rs):
    # Create uniform distribution with same length as unique values in data
    rs = rs  # / np.sum(rs)
    n = len(rs)
    uniform_dist = np.ones(n) * np.mean(rs)  # / n

    # Convert to arrays for entropy calculation
    p = rs
    q = uniform_dist

    # Calculate the average distribution
    m = 0.5 * (p + q)

    # Calculate KL divergences
    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)

    # Jensen-Shannon divergence is average of KL divergences
    js_div = 0.5 * (kl_pm + kl_qm)

    return js_div


def npConcentrationCoefficient(rs):
    # Sort the data
    #rs = np.sort(rs)
    n = len(rs)

    # Calculate cumulative proportions
    cumulative_share_of_population = np.linspace(0, 1, n + 1)
    cumulative_share_of_income = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    # Compute the absolute deviation from the equality line
    absoluteDeviationCurve = np.abs(cumulative_share_of_income -
                                    cumulative_share_of_population)
    sumDeviationArea = 2 * ((np.trapz(cumulative_share_of_income,
                                      cumulative_share_of_population)) - .5)
    absoluteDeviationArea = 2 * np.trapz(absoluteDeviationCurve,
                                         cumulative_share_of_population)
    return absoluteDeviationArea  #, sumDeviationArea


def npConcentrationCoefficientIntegrate(rs):

    def f(x, n, e1, e2, u1, u2):
        return abs((n * (e2 - e1 - u2 + u1)) * x + e1 - u1)

    # Sort the data
    #rs = np.sort(rs)
    n = len(rs)

    # Calculate cumulative proportions
    cumulative_share_of_population = np.linspace(0, 1, n + 1)
    cumulative_share_of_error = np.insert(np.cumsum(rs) / np.sum(rs), 0, 0)

    absoluteDeviationArea = 0
    for i in range(len(cumulative_share_of_population) - 1):
        result, error = quad(
            lambda x: f(x, n, cumulative_share_of_error[
                i], cumulative_share_of_error[
                    i + 1], cumulative_share_of_population[i],
                        cumulative_share_of_population[i + 1]), 0, 1 / n)
        absoluteDeviationArea += result
        #print("Test: ", n, cumulative_share_of_error[i]- cumulative_share_of_error[i+1], cumulative_share_of_population[i]- cumulative_share_of_population[i+1], result)

    # Compute the absolute deviation from the equality line
    #absoluteDeviationCurve = np.abs(cumulative_share_of_error - cumulative_share_of_population)
    #sumDeviationArea = 2 * ((np.trapz(cumulative_share_of_error, cumulative_share_of_population)) - .5)
    #absoluteDeviationArea = 2 * np.trapz(absoluteDeviationCurve, cumulative_share_of_population)
    return absoluteDeviationArea  #, sumDeviationArea


def discreteGiniCoefficient(rs):

    # Gini coefficient calculation
    rs = np.sort(rs)
    n = len(rs)
    cumulative_rs = np.cumsum(rs)
    #cumulative_rs - cumulative_rs[0]
    mean_value = np.average(cumulative_rs)
    #print(cumulative_rs)
    gini_sum = 0
    for i in range(n - 1):
        gini_sum += (2 * i - n - 1) * (cumulative_rs[i])
        #print("I: ", gini_sum)

    #print("cumunfes: ", (.5 * cumulative_rs[-1]), " SUM: ", gini_sum)
    gini = 1 - (gini_sum / (n * n * mean_value))

    return gini


def calculateGini(df):
    # Grouping and calculating Gini coefficient
    result = []
    group_size = 3

    # Iterate over DataFrame in groups of 3 rows
    for start in range(0, len(df), group_size):

        end = start + group_size
        group = df.iloc[start:end]

        # Calculate the Gini coefficient for each subcolumn
        for col in group.columns.levels[
                1]:  # Loop over 'pearson_r' and 'length'

            if (IS_REGRESSION):
                pearson_values = group[('Pearson_r', col)].values
            else:
                pearson_values = group[('AUC', col)].values

            length_values = group[('Length', col)].values

            gini = discrete_gini_coefficient(pearson_values, length_values)
            new_col = ('Gini Coefficient', col)

            # Ensure that the new column is float and not a categorical index
            if new_col not in df.columns:
                df[new_col] = pd.Series(dtype='float')
            #print("INDEX: ", group)

            # Assign the Gini coefficient to the specified row (index 2)
            # Use loc since new_col is a MultiIndex (a tuple)
            #df[new_col] = pd.Series(dtype='float')
            first_row = group.iloc[0]
            first_index = first_row.name
            df.loc[(first_index[0], first_index[1], first_index[2]),
                   new_col] = gini

    return df


#Bin functions to cut data into terciles
def create_bin_function(num_bins):

    def binning_function(data):
        # Ensure data is a pandas Series
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        # Remove any remaining NaN values
        data = data.dropna()

        # Check if we have enough data points
        if len(data) == 0:
            raise ValueError("No valid data points for binning")

        # If all values are the same, create a single bin
        if data.nunique() <= 1:
            print(
                f"DEBUG: All demographic values are the same ({data.iloc[0]}), using single bin"
            )
            return pd.Series([0] * len(data), index=data.index)

        try:
            bins = [data.min() - 1] + [
                data.quantile(i / num_bins) for i in range(1, num_bins)
            ] + [data.max()]
            labels = list(range(0, num_bins))
            return pd.cut(data, bins=bins, labels=labels)
        except Exception as e:
            print(f"DEBUG: Error in binning: {e}")
            print(
                f"DEBUG: Data range: {data.min()} to {data.max()}, unique values: {data.nunique()}"
            )
            raise

    return binning_function


binning_functions = {
    'logincomeHC01_VC85ACS3yr': create_bin_function(num_bins),
    'hsgrad': create_bin_function(num_bins),
    'forgnborn': create_bin_function(num_bins),
    'age': create_bin_function(num_bins),
    'is_female': lambda df: df.map({
        1: 2,
        0: 1,
        None: 3
    }),
    'is_black': lambda df: df.map({
        1: 2,
        0: 1,
        None: 3
    }),
    'individual_income': create_bin_function(num_bins),
}


def labelBins(df, control):
    # Convert to numeric and handle NaN values
    demographic_series = pd.to_numeric(df["demographic_val"], errors='coerce')

    # Remove rows with NaN values before binning
    if demographic_series.isna().any():
        print(
            f"DEBUG: Found {demographic_series.isna().sum()} NaN values in demographic data, removing them"
        )
        valid_mask = demographic_series.notna()
        df_clean = df[valid_mask].copy()
        demographic_series = demographic_series[valid_mask]
    else:
        df_clean = df.copy()

    # Ensure we have valid data for binning
    if len(demographic_series) == 0:
        raise ValueError("No valid demographic data found after cleaning")

    # Apply binning function
    masks = create_bin_function(num_bins)(demographic_series)
    df_clean['bin'] = masks
    return df_clean


def bootstrapResampleBase(df,
                          internal_metric=pearsonr,
                          disp_metric=lambda x: 1 - minMaxRatio(x),
                          num_resamples=BOOTSTRAP_COUNT,
                          bins=True,
                          compareWithNull=False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    bin_ids = df["bin"]
    count_null_trials = 0

    if (bins):
        calculateMetric = lambda *args, **kwargs: calcMetricOnBins(
            *args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(
            *args, **kwargs)

    disp_new = calculateMetric(ypreds, ytrues, disp_metric, bin_ids,
                               internal_metric)
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_base_ypreds = base_ypreds.reset_index(
            drop=True).loc[indices].tolist()
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist()
        bs_bin_ids = bin_ids.reset_index(drop=True).loc[indices].tolist()

        disp_old = calculateMetric(bs_base_ypreds, bs_ytrues, disp_metric,
                                   bs_bin_ids, internal_metric)
        if abs(disp_base) <= abs(disp_new):
            count_null_trials += 1

    #print("AARON_DEBUG_RATIO: ", count_null_trials / num_resamples)
    return count_null_trials / num_resamples


def bootstrapResampleAlternative(df,
                                 internal_metric=pearsonr,
                                 disp_metric=lambda x: 1 - minMaxRatio(x),
                                 num_resamples=BOOTSTRAP_COUNT,
                                 bins=True,
                                 compareWithNull=False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    bin_ids = df["bin"]
    count_null_trials = 0

    if (bins):
        calculateMetric = lambda *args, **kwargs: calcMetricOnBins(
            *args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(
            *args, **kwargs)

    if compareWithNull:
        bin_ids = pd.Series(np.random.permutation(bin_ids.values),
                            index=bin_ids.index)
        disp_base = calculateMetric(ypreds, ytrues, disp_metric, bin_ids,
                                    internal_metric)
    else:
        disp_base = calculateMetric(base_ypreds, ytrues, disp_metric, bin_ids,
                                    internal_metric)
    for k in range(num_resamples):
        indices = np.random.choice(len(df), size=len(df), replace=True)

        bs_ypreds = ypreds.reset_index(drop=True).loc[indices].tolist()
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist()
        bs_bin_ids = bin_ids.reset_index(drop=True).loc[indices].tolist()

        disp_new = calculateMetric(bs_ypreds, bs_ytrues, disp_metric,
                                   bs_bin_ids, internal_metric)
        if disp_base <= disp_new:
            count_null_trials += 1

    return count_null_trials / num_resamples


def bootstrapResampleBoth(df,
                          internal_metric=pearsonr,
                          disp_metric=lambda x: 1 - minMaxRatio(x),
                          num_resamples=BOOTSTRAP_COUNT,
                          bins=True,
                          compareWithNull=False):
    ypreds = df["pred"]
    ytrues = df["true"]
    base_ypreds = df["base"]
    bin_ids = df["bin"]
    demographics = df["demographic_val"]
    count_null_trials = 0

    if (bins):
        calculateMetric = lambda *args, **kwargs: calcMetricOnBins(
            *args, **kwargs)
    else:
        calculateMetric = lambda *args, **kwargs: calcMetricOnFullData(
            *args, **kwargs)

    for k in range(num_resamples):

        indices_new = np.random.choice(len(df), size=len(df), replace=True)
        bs_new_ypreds = ypreds.reset_index(drop=True).loc[indices_new].tolist()
        bs_new_ytrues = ytrues.reset_index(drop=True).loc[indices_new].tolist()
        bs_new_bin_ids = bin_ids.reset_index(
            drop=True).loc[indices_new].tolist()
        bs_new_demographics = demographics.reset_index(
            drop=True).loc[indices_new].tolist()

        disp_new = calculateMetric(bs_new_ypreds,
                                   bs_new_ytrues,
                                   demographics=bs_new_demographics,
                                   disp_metric=disp_metric,
                                   bin_ids=bs_new_bin_ids,
                                   internal_metric=internal_metric)
        if compareWithNull:
            if (bins):
                disp_original = calculateMetric(
                    bs_new_ypreds,
                    bs_new_ytrues,
                    demographics=demographics,
                    disp_metric=disp_metric,
                    bin_ids=bs_new_bin_ids,
                    internal_metric=internal_metric)
                #print("DISP Original: ", disp_original)
                bs_new_bin_ids = np.random.permutation(bs_new_bin_ids)
                disp_base = calculateMetric(bs_new_ypreds,
                                            bs_new_ytrues,
                                            demographics=demographics,
                                            disp_metric=disp_metric,
                                            bin_ids=bs_new_bin_ids,
                                            internal_metric=internal_metric)
                #print("DISP BASE: ", disp_base)

            else:
                indices_new = np.random.choice(len(df),
                                               size=len(df),
                                               replace=True)
                bs_new_ypreds = ypreds.reset_index(
                    drop=True).loc[indices_new].tolist()
                bs_new_ytrues = ytrues.reset_index(
                    drop=True).loc[indices_new].tolist()
                bs_new_bin_ids = bin_ids.reset_index(
                    drop=True).loc[indices_new].tolist()
                disp_base = calculateMetric(bs_new_ypreds,
                                            bs_new_ytrues,
                                            demographics=demographics,
                                            disp_metric=disp_metric,
                                            bin_ids=bs_new_bin_ids,
                                            internal_metric=internal_metric)
        else:
            indices_base = np.random.choice(len(df),
                                            size=len(df),
                                            replace=True)
            bs_base_ypreds = base_ypreds.reset_index(
                drop=True).loc[indices_base].tolist()
            bs_base_ytrues = ytrues.reset_index(
                drop=True).loc[indices_base].tolist()
            bs_base_bin_ids = bin_ids.reset_index(
                drop=True).loc[indices_base].tolist()
            bs_base_demographics = demographics.reset_index(
                drop=True).loc[indices_base].tolist()
            disp_base = calculateMetric(bs_base_ypreds,
                                        bs_base_ytrues,
                                        demographics=bs_base_demographics,
                                        disp_metric=disp_metric,
                                        bin_ids=bs_base_bin_ids,
                                        internal_metric=internal_metric)
        #print("DISPARITIES: ", disp_new, disp_base)
        if abs(disp_base) <= abs(disp_new):
            count_null_trials += 1

    return count_null_trials / num_resamples


def calcMetricOnBins(ypreds,
                     ytrues,
                     demographics,
                     disp_metric=lambda x: 1 - minMaxRatio(x),
                     bin_ids=None,
                     internal_metric=pearsonr):
    numpyArrays = [
        np.array(var) for var in (ypreds, ytrues, demographics, bin_ids)
    ]
    stacked_array = np.column_stack(numpyArrays)
    sorted_array = stacked_array[stacked_array[:, -2].argsort()]
    #print(sorted_array.shape)
    ypreds, ytrues, demographs, bin_ids = sorted_array.T

    #Split values by bin
    terc_ypreds = [[] for _ in range(num_bins)]
    terc_ytrues = [[] for _ in range(num_bins)]

    #sorted_array = stacked_array[stacked_array[:, -1].argsort()]
    temp = ""

    for i in range(len(ypreds)):
        this_bin = bin_ids[i].astype(int)
        #temp += str(sorted_array[i])
        terc_ypreds[this_bin].append(ypreds[i])
        terc_ytrues[this_bin].append(ytrues[i])

    last_column = sorted_array[:, -1]

    # Convert to a string of numbers
    last_column_str = "".join(map(str, last_column.astype(int)))
    #print("SORTED BINS: ", last_column_str)
    #print("Test: ", terc_ypreds[0][:5])

    #calculate the metric score per bin
    results = []
    for t in range(num_bins):
        cor = internal_metric(terc_ypreds[t], terc_ytrues[t])
        results.append(cor)

    return disp_metric(results)


def calcMetricOnFullData(ypreds,
                         ytrues,
                         demographics=None,
                         disp_metric=lambda x: calculateGini(x),
                         bin_ids=None,
                         internal_metric=None):

    absolute_diff = [abs(a - b) for a, b in zip(ypreds, ytrues)]

    sorted_values = [v for _, v in sorted(zip(demographics, absolute_diff))]

    result = disp_metric(sorted_values)

    return result


def calcMetricOnFullData2(absolute_diff,
                          demographics=None,
                          disp_metric=lambda x: calculateGini(x),
                          bin_ids=None,
                          internal_metric=None):

    #absolute_diff = [abs(a - b) for a, b in zip(ypreds, ytrues)]

    sorted_values = [v for _, v in sorted(zip(demographics, absolute_diff))]

    result = disp_metric(sorted_values)

    return result


def minMaxRatio(rs):
    return np.min(rs) / np.max(rs)




if __name__ == "__main__":

    main()

#################
