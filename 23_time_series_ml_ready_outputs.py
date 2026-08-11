"""
Module W: ML-Ready Outputs
------------------------------
Section 19 of the spec. Produces the two required deliverables:
  19.1 Original-frequency cleaned dataset
  19.2 User-selected-frequency dataset (same frequency, or aggregated
       via Module T if the client chose a coarser one)

IMPUTATION-AWARE BUT NOT IMPUTATION-DEPENDENT: this dataset (and every
dataset until Phase B is built) never actually imputes anything, so
every "_was_imputed" flag is False and every "Imputation_Method" is
NA by construction. The function signature already accepts optional
imputation info (imputed_mask / imputation_method) so that when Phase
B lands, THIS module doesn't need to change -- only the caller wiring
imputation results into it does. Building it imputation-shaped now,
rather than retrofitting it later, is the cheaper order to do this in.
"""

import pandas as pd


def build_minimal_ml_ready_output(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """
    The actual minimal feature set a model would consume: just the
    sequential Time_Step and the cleaned value column(s) -- no raw
    time text, no component breakdown, no diagnostic flags. This is
    what belongs as the FIRST sheet in the delivered workbook; the
    richer build_original_frequency_output() below is still valuable
    for audit/traceability, but shouldn't be the first thing a client
    opens if their goal is "give me the ML-ready columns."
    """
    output = pd.DataFrame(index=df.index)
    output["Time_Step"] = df["Time_Step"]
    for value_col in value_columns:
        output[value_col] = df[value_col]
    return output


def build_original_frequency_output(
    df: pd.DataFrame,
    time_columns,
    component_columns: list,
    value_columns: list,
    imputed_mask: dict = None,
    imputation_method: dict = None,
) -> pd.DataFrame:
    """
    component_columns: (meaning, output_name) pairs from Module P.
    time_columns: a single column name (str) or a list of column names
    -- with several time columns (e.g. separate Date + Time), each
    gets its own "Original_Time_<col>" output column; with just one,
    the output column is named plain "Original_Time" (unchanged from
    before, for backward compatibility).

    Imputation info is read DIRECTLY from df's own
    "<col>_was_imputed" / "<col>_imputation_method" columns when
    present -- Modules Z/AB/AC (26/28/29) already add these onto the
    dataframe as part of actually running imputation, so this reads
    real state instead of requiring a separately-populated parameter
    that nothing in the real pipeline ever fills in. imputed_mask /
    imputation_method remain as an explicit override path for a caller
    that wants to supply this separately, but the columns on df take
    priority since that's how the actual pipeline works today.

    "_Original" vs "_Cleaned": by the time this runs, df[value_col]
    already HAS the imputed value baked in (Modules 26/28/29 overwrite
    it in place) -- so "_Original" is reconstructed as genuinely
    missing (NaN) wherever "_was_imputed" says a cell was filled,
    rather than showing the same post-imputation number under both
    column names, which would misrepresent what was actually observed
    versus estimated.
    """
    if isinstance(time_columns, str):
        time_columns = [time_columns]

    output = pd.DataFrame(index=df.index)
    if len(time_columns) == 1:
        output["Original_Time"] = df[time_columns[0]]
    else:
        for col in time_columns:
            output[f"Original_Time_{col}"] = df[col]

    for _, col_name in component_columns:
        if col_name in df.columns:
            output[col_name] = df[col_name]

    output["Time_Step"] = df["Time_Step"]
    output["Time_Step_Unit"] = df.get("Time_Step_Unit")
    output["Original_Frequency"] = df.get("Original_Frequency")

    for value_col in value_columns:
        was_imputed_col = f"{value_col}_was_imputed"
        method_col = f"{value_col}_imputation_method"

        if was_imputed_col in df.columns:
            was_imputed_series = df[was_imputed_col].fillna(0).astype(bool)
        elif imputed_mask and value_col in imputed_mask:
            was_imputed_series = pd.Series(imputed_mask[value_col], index=df.index).astype(bool)
        else:
            was_imputed_series = pd.Series(False, index=df.index)

        output[f"{value_col}_Cleaned"] = df[value_col]
        output[f"{value_col}_Original"] = df[value_col].where(~was_imputed_series, other=pd.NA)
        output[f"{value_col}_was_imputed"] = was_imputed_series.astype(int)

        if method_col in df.columns:
            output[f"{value_col}_Imputation_Method"] = df[method_col]
        elif imputation_method and value_col in imputation_method:
            output[f"{value_col}_Imputation_Method"] = imputation_method[value_col]
        else:
            output[f"{value_col}_Imputation_Method"] = pd.NA

    output["_time_step_inserted"] = df.get("_time_step_inserted", False)
    output["_original_record_present"] = df.get("_original_record_present", True)

    return output


def build_user_selected_frequency_output(aggregated_df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Thin wrapper around Module T's aggregate_to_frequency() output --
    adds the two fields the spec asks for that Module T's own report
    dict already has the numbers for, but the OUTPUT DATAFRAME itself
    didn't yet carry as columns: which method was used, and a plain-
    English completeness indicator per row (so a client opening this
    file in Excel doesn't need to cross-reference the separate report
    to know a given quarter/year is incomplete).
    """
    output = aggregated_df.copy()
    output["Aggregation_Method"] = method if "_block_id" not in output.columns else method
    output["Number_of_Imputed_Values"] = 0  # always 0 until Phase B exists -- see module docstring
    output["Completeness_Indicator"] = output["Number_of_Missing_Observations"].apply(
        lambda n: "Complete" if n == 0 else f"Incomplete — {n} missing observation(s) in this period"
    )
    return output


if __name__ == "__main__":
    import importlib.util
    import sys

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    split_mod = load("split_mod", "16_time_series_split_validate.py")
    sort_mod = load("sort_mod", "17_time_series_sort.py")
    gaps_mod = load("gaps_mod", "19_time_series_gaps_and_timestep.py")
    freq_mod = load("freq_mod", "20_time_series_frequency_aggregation.py")

    component_meanings = [
        "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
        "Month",
    ]

    df = pd.read_csv("shampoo.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(df, "Month", "-", component_meanings)
    sorted_df, _ = sort_mod.sort_time_series(split_df, split_report["component_columns"])
    complete_df, _ = gaps_mod.build_complete_sequence(sorted_df, split_report["component_columns"], "Month", ["Sales"])
    freq_info = gaps_mod.detect_original_frequency(split_report["component_columns"])

    print("=" * 60)
    print("TEST 1: original-frequency output (19.1)")
    print("=" * 60)
    original_output = build_original_frequency_output(
        complete_df, "Month", split_report["component_columns"], ["Sales"]
    )
    print(original_output.columns.tolist())
    print(original_output.head(3).to_string(index=False))
    print("...")
    print(f"Total rows: {len(original_output)} (should be 36)")
    print(f"All was_imputed False: {(~original_output['Sales_was_imputed']).all()}")

    print("\n" + "=" * 60)
    print("TEST 2: user-selected-frequency output (19.2) -- quarterly, sum")
    print("=" * 60)
    quarterly_agg, agg_report = freq_mod.aggregate_to_frequency(complete_df, freq_info, ["Sales"], multiple=3, method="sum")
    quarterly_output = build_user_selected_frequency_output(quarterly_agg, "sum")
    print(quarterly_output.columns.tolist())
    print(quarterly_output[["Time_Step", "Sales", "Aggregation_Method", "Number_of_Imputed_Values", "Completeness_Indicator"]].to_string(index=False))
