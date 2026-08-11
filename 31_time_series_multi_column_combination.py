"""
Module AE: Multi-Column Time Stamp Combination
------------------------------------------------------
Generalizes the pipeline beyond "exactly one time column" to "one or
more columns that TOGETHER form the time stamp" -- e.g. a separate
Date column and Time column, or fully pre-split Year/Month/Day
columns each on their own.

DESIGN: reuses Module P's (16_time_series_split_validate.py) existing
split_and_validate_time_column() UNCHANGED, once independently per
configured column, then merges the results. This is deliberate --
every configured column already has its own separator and component
meanings (single-element or multi-element, both already supported),
so there's no new splitting logic to write; the only genuinely new
problem is combining several already-correct per-column results into
one coherent row-level structure.

A row is valid overall only if EVERY configured column's own
components validated successfully for that row -- if the Date column
parses fine but the Time column doesn't for a given row, that row is
not usable for building the time index, and should be reported that
way, not silently treated as "mostly fine."

CONFLICT DETECTION: if two different configured columns both claim
the same meaning (e.g. both are told they contain "Month"), that's
genuinely ambiguous -- which one is authoritative? -- so this raises
a clear error rather than silently letting the second one win.
"""

from importlib import import_module

import pandas as pd

split_mod = import_module("16_time_series_split_validate")


def combine_multiple_time_columns(df: pd.DataFrame, time_column_configs: list) -> tuple:
    """
    time_column_configs: list of dicts, each describing ONE configured
    time-bearing column:
        {"column": "Date", "separator": "-", "component_meanings": [...]}
        {"column": "Time", "separator": ":", "component_meanings": [...]}
    (separator may be None for a single-element column, exactly as
    Module P already supports.)

    Returns (df_with_combined_columns, report). report contains:
      "component_columns": the MERGED list of (meaning, output_name)
        pairs across every configured column, in the order the
        columns were configured (Date's components before Time's, in
        the example above)
      "n_valid_rows" / "n_invalid_rows": combined -- a row counts as
        valid only if valid in EVERY configured column
      "invalid_rows": combined table, one row per originally-invalid
        row, noting WHICH configured column(s) failed and why
      "per_column_reports": the individual Module P report for each
        configured column, kept for anyone who wants the detail
    """
    if not time_column_configs:
        raise ValueError("At least one time column must be configured.")

    seen_meanings = {}
    for config in time_column_configs:
        for meaning in config["component_meanings"]:
            if meaning in ("Component to ignore",):
                continue
            if meaning in seen_meanings and seen_meanings[meaning] != config["column"]:
                raise ValueError(
                    f"Both '{seen_meanings[meaning]}' and '{config['column']}' are configured to "
                    f"mean '{meaning}' -- ambiguous. Only one column can be the authoritative "
                    f"source for a given meaning; reconsider one of these columns' component "
                    f"meanings before continuing."
                )
            seen_meanings[meaning] = config["column"]

    working_df = df.copy()
    combined_component_columns = []
    per_column_reports = {}
    combined_valid = [True] * len(working_df)
    combined_reasons = [None] * len(working_df)

    for config in time_column_configs:
        col = config["column"]
        split_df, report = split_mod.split_and_validate_time_column(
            working_df, col, config["separator"], config["component_meanings"]
        )
        per_column_reports[col] = report
        combined_component_columns.extend(report["component_columns"])

        new_cols = [name for _, name in report["component_columns"]]
        for name in new_cols:
            working_df[name] = split_df[name]

        # Plain Python loop rather than vectorized pandas string ops --
        # confirmed directly that pandas 3.0's Arrow-backed string dtype
        # raises on '+' between Series containing nulls, even for an
        # empty selection (the declared dtype alone is enough to trigger
        # it, regardless of actual row count).
        col_valid_list = split_df["_time_component_valid"].tolist()
        col_reason_list = split_df["_time_component_invalid_reason"].tolist()
        for i in range(len(working_df)):
            if not col_valid_list[i]:
                note = f"'{col}': {col_reason_list[i]}"
                combined_reasons[i] = note if combined_reasons[i] is None else f"{combined_reasons[i]}; {note}"
            combined_valid[i] = combined_valid[i] and col_valid_list[i]

    working_df["_time_component_valid"] = combined_valid
    working_df["_time_component_invalid_reason"] = combined_reasons

    invalid_mask = ~working_df["_time_component_valid"]
    invalid_rows = working_df.loc[
        invalid_mask, [c["column"] for c in time_column_configs] + ["_time_component_invalid_reason"]
    ].copy()
    invalid_rows.insert(0, "Original_Row_Number", invalid_rows.index + 2)

    combined_output_columns = [name for _, name in combined_component_columns]

    report = {
        "component_columns": combined_component_columns,
        "output_columns": combined_output_columns,
        "n_valid_rows": int(working_df["_time_component_valid"].sum()),
        "n_invalid_rows": int(invalid_mask.sum()),
        "invalid_rows": invalid_rows.reset_index(drop=True),
        "per_column_reports": per_column_reports,
    }

    return working_df, report


if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("TEST 1: separate Date + Time columns, combined into one time index")
    print("=" * 60)
    df = pd.DataFrame(
        {
            "Date": ["2020-01-15", "2020-01-16", "2020-01-17"],
            "Time": ["14:30:00", "15:00:00", "09:15:30"],
            "Reading": [10.5, 11.2, 9.8],
        }
    )
    configs = [
        {
            "column": "Date",
            "separator": "-",
            "component_meanings": ["Year (real calendar year)", "Month", "Day"],
        },
        {
            "column": "Time",
            "separator": ":",
            "component_meanings": ["Hour", "Minute", "Second"],
        },
    ]
    combined_df, report = combine_multiple_time_columns(df, configs)
    print("Combined component columns:", report["component_columns"])
    print("n_valid:", report["n_valid_rows"], " n_invalid:", report["n_invalid_rows"])
    print(combined_df[["Date", "Time", "Calendar_Year", "Month_Number", "Day_Number", "Hour", "Minute", "Second"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST 2: a row invalid in ONLY the Time column should make the combined row invalid too")
    print("=" * 60)
    df2 = pd.DataFrame(
        {
            "Date": ["2020-01-15", "2020-01-16"],
            "Time": ["14:30:00", "99:00:00"],  # invalid hour
            "Reading": [10.5, 11.2],
        }
    )
    combined_df2, report2 = combine_multiple_time_columns(df2, configs)
    print("n_valid:", report2["n_valid_rows"], " n_invalid:", report2["n_invalid_rows"], "(should be 1 valid, 1 invalid)")
    print(report2["invalid_rows"].to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST 3: conflict detection -- two columns both claiming 'Month'")
    print("=" * 60)
    df3 = pd.DataFrame({"A": ["1"], "B": ["2"], "Value": [100]})
    conflicting_configs = [
        {"column": "A", "separator": None, "component_meanings": ["Month"]},
        {"column": "B", "separator": None, "component_meanings": ["Month"]},
    ]
    try:
        combine_multiple_time_columns(df3, conflicting_configs)
        print("BUG: did not raise on conflicting meanings")
    except ValueError as e:
        print("Correctly rejected conflicting configuration:", e)

    print("\n" + "=" * 60)
    print("TEST 4: single-element pre-split columns (separate Year, Month, Day columns)")
    print("=" * 60)
    df4 = pd.DataFrame({
        "Yr": ["2021", "2021", "2021"],
        "Mo": ["1", "1", "2"],
        "Dy": ["1", "2", "1"],
        "Value": [1, 2, 3],
    })
    presplit_configs = [
        {"column": "Yr", "separator": None, "component_meanings": ["Year (real calendar year)"]},
        {"column": "Mo", "separator": None, "component_meanings": ["Month"]},
        {"column": "Dy", "separator": None, "component_meanings": ["Day"]},
    ]
    combined_df4, report4 = combine_multiple_time_columns(df4, presplit_configs)
    print("n_valid:", report4["n_valid_rows"])
    print(combined_df4[["Yr", "Mo", "Dy", "Calendar_Year", "Month_Number", "Day_Number"]].to_string(index=False))
