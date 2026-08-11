"""
Module T: Output Frequency Selection & Aggregation
--------------------------------------------------------
Sections 7-8 of the spec. Takes Module S's complete, gap-filled,
Time_Step-numbered series and either:
  (a) keeps it at the original frequency, or
  (b) aggregates it to a coarser frequency the client explicitly chose,
      using an aggregation method the client explicitly chose (never
      picked automatically -- sum and mean mean genuinely different
      things for sales data, per the spec's own example).

FINER FREQUENCY IS STRUCTURALLY IMPOSSIBLE TO REQUEST, not just
blocked after the fact: the question generator only ever offers
multiples >= 1 of the ORIGINAL unit, so "weekly output from monthly
data" never appears as a selectable option in the first place. The
explicit spec warning text is still available as
FINER_FREQUENCY_WARNING, for reuse if a future UI (e.g. a free-text
"custom" field) needs to reject an out-of-range value after the fact.
"""

import pandas as pd

FINER_FREQUENCY_WARNING = (
    "The requested time-step frequency is finer than the frequency at which the original "
    "observations were recorded. Monthly observations do not contain genuine daily or weekly "
    "measurements. Creating finer time steps would require temporal disaggregation and would "
    "generate estimated or synthetic values rather than recover observed data. This is therefore "
    "not supported by the standard machine-learning-ready cleaning workflow. Specialist temporal "
    "disaggregation can be considered as future customised work, but it must not be presented as "
    "ordinary cleaning or imputation."
)

AGGREGATION_METHODS = ["sum", "mean", "median", "min", "max", "first", "last", "count"]


def describe_output_step_length(freq_info: dict, k: int) -> str:
    """
    DESCRIPTIVE ONLY -- never a selectable option. Given the native
    detected interval and a chosen k, returns a short human-readable
    approximation of the resulting output step length (e.g. "≈ 3
    Months" or "≈ 1 Year"), so the client has a sense of what they're
    choosing without the app dictating calendar-flavoured menu
    choices. Per the updated spec: "Calendar equivalents may be
    displayed only as descriptive information."
    """
    if freq_info["mode"] == "datetime":
        total_seconds = freq_info["interval"] * k
        for name, size in [("Day", 86400), ("Hour", 3600), ("Minute", 60), ("Second", 1)]:
            if total_seconds % size == 0:
                value = total_seconds // size
                return f"≈ {value} {name}{'s' if value != 1 else ''}"
        return f"≈ {total_seconds} Seconds"

    unit_name = freq_info["unit_name"]
    total_in_native_unit = freq_info["interval_in_unit"] * k

    if unit_name == "Month" and total_in_native_unit % 12 == 0 and total_in_native_unit > 0:
        years = int(total_in_native_unit // 12)
        return f"≈ {years} Year{'s' if years != 1 else ''}"
    if unit_name == "Quarter" and total_in_native_unit % 4 == 0 and total_in_native_unit > 0:
        years = int(total_in_native_unit // 4)
        return f"≈ {years} Year{'s' if years != 1 else ''}"

    value = total_in_native_unit
    display_value = f"{value:g}"
    return f"≈ {display_value} {unit_name}{'s' if value != 1 else ''}"


def generate_output_frequency_question(freq_info: dict) -> dict:
    """
    Redesigned per the updated spec: NO pre-packaged calendar-labeled
    choices (the old "Quarterly (every 3 months)" style menu) -- the
    client chooses a raw integer k directly ("combine every k
    consecutive native time steps into one"), with calendar
    equivalents shown as descriptive text only (see
    describe_output_step_length() above), never as a menu to pick
    from. k=1 (native frequency, no aggregation) is always the
    default -- nothing is aggregated unless explicitly requested.
    """
    return {
        "id": "ts_output_k",
        "type": "number_input",
        "text": (
            f"This series' native sampling interval is {freq_info['frequency_label']}. How many "
            f"consecutive time steps (k) should be combined into one output time step? "
            f"(k=1 keeps the native frequency; e.g. k=3 combines every 3 native steps into one.)"
        ),
        "default": 1,
        "min_value": 1,
    }


def generate_aggregation_method_question(value_column: str) -> dict:
    """Only relevant (and should only be shown) when the client chose a multiple > 1."""
    return {
        "id": f"ts_aggregation_method__{value_column}",
        "type": "single_select",
        "text": (
            f"Aggregating '{value_column}' to a coarser frequency -- how should values within "
            f"each new period be combined? (e.g. sum = total for the period, mean = average "
            f"per original period within it -- these give different numbers, so pick deliberately.)"
        ),
        "options": AGGREGATION_METHODS,
        "default": None,  # no default -- exactly the spec's "must not choose automatically" rule
    }


def validate_output_multiple(multiple) -> None:
    """Guards the 'custom multiple' free-entry path -- raises with the spec's own warning
    text if someone tries to request a fractional or non-positive multiple (which would
    imply a finer-than-original frequency)."""
    if not isinstance(multiple, int) or multiple < 1:
        raise ValueError(FINER_FREQUENCY_WARNING)


def aggregate_to_frequency(
    df: pd.DataFrame, freq_info: dict, value_columns: list, multiple: int, method: str
) -> tuple:
    """
    Returns (aggregated_df, report). Only operates on rows that made
    it through Module S (i.e. have a real Time_Step) -- Module S's
    excluded invalid rows are passed through unaggregated and
    reported separately, same "never silently absorb a problem row"
    pattern as every prior module.

    multiple == 1 is a legitimate choice (explicit "same frequency"
    confirmation) and short-circuits to a lightweight passthrough
    rather than running a group-of-one aggregation.
    """
    validate_output_multiple(multiple)
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method '{method}'. Choose from {AGGREGATION_METHODS}.")

    year_col, cycle_col = freq_info["year_col"], freq_info["cycle_col"]

    has_timestep = df["Time_Step"].notna()
    steppable_df = df[has_timestep].copy()
    excluded_df = df[~has_timestep].copy()
    steppable_df["_original_record_present"] = steppable_df["_original_record_present"].astype(bool)

    # multiple == 1 ("same frequency") deliberately runs through the
    # EXACT SAME block-grouping logic below rather than a special-cased
    # passthrough -- a block size of 1 naturally produces one output
    # row per input row, but with the SAME clean, curated column shape
    # (Number_of_Original_Observations, Completeness_Indicator, etc.)
    # as any other aggregation, instead of leaking internal diagnostic
    # columns (_time_component_valid, _period_duplicate_count, ...)
    # into a client-facing sheet.

    steppable_df = steppable_df.sort_values("Time_Step", kind="mergesort").reset_index(drop=True)
    steppable_df["_block_id"] = steppable_df["Time_Step"].astype(int) // multiple

    agg_rows = []
    for block_id, group in steppable_df.groupby("_block_id", sort=True):
        row = {"Time_Step": block_id}
        if year_col:
            row[f"{year_col}_Start"] = group.iloc[0][year_col]
            row[f"{year_col}_End"] = group.iloc[-1][year_col]
        if cycle_col:
            row[f"{cycle_col}_Start"] = group.iloc[0][cycle_col]
            row[f"{cycle_col}_End"] = group.iloc[-1][cycle_col]

        row["Number_of_Original_Observations"] = int(group["_original_record_present"].sum())
        n_missing = int((~group["_original_record_present"]).sum()) + int(
            sum((group[vc].isna() & group["_original_record_present"]).sum() for vc in value_columns)
        )
        row["Number_of_Missing_Observations"] = n_missing
        row["Aggregation_Basis"] = (
            "Observed values only (missing observations excluded from aggregation)"
            if n_missing > 0
            else "All observations present"
        )
        # Explicit incomplete-final-group handling, per the updated
        # spec ("the handling of an incomplete final group must be
        # explicitly defined") -- rather than leaving it only
        # implicitly inferable from the observation counts above, this
        # is stated directly: how many native steps actually went into
        # this group, and whether that's the full k or fewer (which
        # only ever happens for the LAST group, when the total number
        # of native steps isn't an exact multiple of k).
        row["Native_Steps_In_Group"] = len(group)
        row["Is_Complete_Group"] = len(group) == multiple

        for vc in value_columns:
            observed = group[vc].dropna()
            if method == "count":
                row[vc] = int(observed.count())
            elif observed.empty:
                row[vc] = pd.NA
            elif method == "first":
                row[vc] = observed.iloc[0]
            elif method == "last":
                row[vc] = observed.iloc[-1]
            else:
                row[vc] = getattr(observed, method)()

        agg_rows.append(row)

    aggregated_df = pd.DataFrame(agg_rows)
    aggregated_df["Time_Step_Unit"] = f"{multiple} x {steppable_df['Time_Step_Unit'].iloc[0]}"
    aggregated_df["Original_Frequency"] = freq_info["frequency_label"]

    final_group_complete = bool(aggregated_df["Is_Complete_Group"].iloc[-1]) if len(aggregated_df) else True
    final_group_size = int(aggregated_df["Native_Steps_In_Group"].iloc[-1]) if len(aggregated_df) else 0

    report = {
        "multiple_applied": multiple,
        "method_applied": method,
        "n_output_periods": len(aggregated_df),
        "n_input_periods": len(steppable_df),
        "n_excluded_invalid_rows": len(excluded_df),
        "final_group_is_complete": final_group_complete,
        "final_group_size": final_group_size,
    }
    return aggregated_df, report


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

    component_meanings = [
        "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
        "Month",
    ]

    df = pd.read_csv("shampoo.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(df, "Month", "-", component_meanings)
    sorted_df, _ = sort_mod.sort_time_series(split_df, split_report["component_columns"])
    complete_df, gap_report = gaps_mod.build_complete_sequence(sorted_df, split_report["component_columns"], "Month", ["Sales"])
    freq_info = gaps_mod.detect_original_frequency(split_report["component_columns"], sorted_df)

    print("=" * 60)
    print("TEST 1: question generation for monthly-origin data")
    print("=" * 60)
    q = generate_output_frequency_question(freq_info)
    print(q["text"])
    print("default k:", q["default"], " min:", q["min_value"])
    print()
    print("Descriptive calendar-equivalents for a few example k values (informational only, not a menu):")
    for k in [1, 3, 6, 12, 24]:
        print(f"  k={k}: {describe_output_step_length(freq_info, k)}")

    print("\n" + "=" * 60)
    print("TEST 2: same frequency (multiple=1) -- passthrough")
    print("=" * 60)
    same_df, same_report = aggregate_to_frequency(complete_df, freq_info, ["Sales"], multiple=1, method="sum")
    print(same_report)
    print("Rows:", len(same_df), "(should be 36)")

    print("\n" + "=" * 60)
    print("TEST 3: quarterly aggregation, method=sum (spec's own worked example: months 1-3, 4-6, 7-9, 10-12)")
    print("=" * 60)
    q_df, q_report = aggregate_to_frequency(complete_df, freq_info, ["Sales"], multiple=3, method="sum")
    print(q_report)
    print(q_df[["Time_Step", "Relative_Year_Start", "Month_Number_Start", "Relative_Year_End", "Month_Number_End", "Sales", "Number_of_Original_Observations", "Number_of_Missing_Observations"]].head(6).to_string(index=False))
    # sanity check against manual sum of first quarter (months 1,2,3 of shampoo.csv: 266.0+145.9+183.1)
    manual_q1_sum = df.iloc[0:3]["Sales"].sum()
    print(f"\nManual check -- Q1 sum should be {manual_q1_sum}, aggregated result is {q_df.iloc[0]['Sales']}")

    print("\n" + "=" * 60)
    print("TEST 4: yearly aggregation, method=mean")
    print("=" * 60)
    y_df, y_report = aggregate_to_frequency(complete_df, freq_info, ["Sales"], multiple=12, method="mean")
    print(y_report)
    print(y_df[["Time_Step", "Relative_Year_Start", "Relative_Year_End", "Sales"]].to_string(index=False))
    manual_y1_mean = df.iloc[0:12]["Sales"].mean()
    print(f"\nManual check -- Year 1 mean should be {manual_y1_mean:.4f}, aggregated result is {y_df.iloc[0]['Sales']:.4f}")

    print("\n" + "=" * 60)
    print("TEST 5: reject a fractional/finer 'custom' multiple")
    print("=" * 60)
    try:
        validate_output_multiple(0.5)
        print("BUG: did not raise!")
    except ValueError as e:
        print("Correctly rejected:", str(e)[:80], "...")

    print("\n" + "=" * 60)
    print("TEST 6: aggregation over the missing-period example (1-05,1-06,[1-07 missing],1-08), quarterly")
    print("=" * 60)
    gap_df = pd.DataFrame({"Month": ["1-05", "1-06", "1-08"], "Sales": [100.0, 110.0, 130.0]})
    gap_split, gap_split_report = split_mod.split_and_validate_time_column(gap_df, "Month", "-", component_meanings)
    gap_sorted, _ = sort_mod.sort_time_series(gap_split, gap_split_report["component_columns"])
    gap_complete, _ = gaps_mod.build_complete_sequence(gap_sorted, gap_split_report["component_columns"], "Month", ["Sales"])
    gap_freq_info = gaps_mod.detect_original_frequency(gap_split_report["component_columns"])
    gap_agg, gap_agg_report = aggregate_to_frequency(gap_complete, gap_freq_info, ["Sales"], multiple=4, method="sum")
    print(gap_agg[["Time_Step", "Sales", "Number_of_Original_Observations", "Number_of_Missing_Observations", "Aggregation_Basis"]].to_string(index=False))
