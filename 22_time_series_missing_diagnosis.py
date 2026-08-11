"""
Module V: Missing-Value Diagnosis
--------------------------------------
Section 10 of the spec -- and the module that directly implements the
very first instruction given for this whole build: "missing value
option should be only shown when the data has some missing value."

diagnose_missing_values() is the single source of truth every caller
(app.py included) should check before deciding whether to render ANY
imputation UI at all. columns_needing_imputation() below is the one-
line gate: an empty list means skip the entire imputation section,
exactly like shampoo.csv's Sales column does today.

Distinguishes missingness by WHERE it came from, per the spec:
  - missing because Module S inserted an entirely absent period
    (_time_step_inserted == True)
  - missing because the row genuinely existed in the source file but
    the cell itself was blank / became NaN via placeholder or
    non-numeric resolution (_original_record_present == True, value
    still NaN)
These are different problems with different implications for which
imputation method makes sense later, so they're reported separately
rather than folded into one undifferentiated missing count.
"""

import pandas as pd


def _find_missing_runs(is_missing: pd.Series) -> list:
    """
    is_missing: a boolean Series in Time_Step order. Returns a list of
    run lengths for every MAXIMAL consecutive stretch of missing
    values (e.g. [False, True, True, False, True] -> [2, 1]).
    """
    runs = []
    current_run = 0
    for missing in is_missing:
        if missing:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    return runs


def diagnose_missing_values(df: pd.DataFrame, value_columns: list) -> dict:
    """
    Returns {column_name: diagnosis_dict} for every value column,
    where diagnosis_dict has exactly the fields the spec's worked
    example lists, plus the two "where did this missingness come
    from" breakdowns.

    Assumes df is already sorted by Time_Step (true after Module Q/S)
    and restricted to rows that have a genuine Time_Step (i.e. Module
    P's invalid rows, which never got one, are excluded here --
    they're a different, already-reported problem).
    """
    steppable = df[df["Time_Step"].notna()].sort_values("Time_Step", kind="mergesort")
    n_rows = len(steppable)

    report = {}
    for col in value_columns:
        if col not in steppable.columns:
            continue

        is_missing = steppable[col].isna()
        missing_count = int(is_missing.sum())
        observed_count = n_rows - missing_count

        runs = _find_missing_runs(is_missing)
        isolated_count = sum(1 for r in runs if r == 1)
        consecutive_runs = [r for r in runs if r >= 2]
        longest_run = max(runs) if runs else 0

        # Leading: missing rows from the very start until the first observed value.
        leading = 0
        for v in is_missing:
            if v:
                leading += 1
            else:
                break

        # Trailing: missing rows from the very end back to the last observed value.
        trailing = 0
        for v in reversed(is_missing.tolist()):
            if v:
                trailing += 1
            else:
                break

        inserted_period_missing = int((is_missing & (~steppable["_original_record_present"])).sum())
        original_cell_missing = int((is_missing & steppable["_original_record_present"]).sum())

        report[col] = {
            "observed_values": observed_count,
            "missing_values": missing_count,
            "missing_percentage": round((missing_count / n_rows * 100) if n_rows else 0.0, 2),
            "isolated_missing_observations": isolated_count,
            "consecutive_missing_runs": consecutive_runs,
            "longest_missing_run": longest_run,
            "missing_at_beginning": leading,
            "missing_at_end": trailing,
            "missing_from_inserted_periods": inserted_period_missing,
            "missing_originally_in_dataset": original_cell_missing,
            "imputation_required": missing_count > 0,
        }

    return report


def columns_needing_imputation(diagnosis: dict) -> list:
    """
    THE gate. Empty list -> skip the entire imputation section, same
    as the cross-sectional pipeline's equivalent check. Never render
    imputation-method questions for a column that doesn't need them.
    """
    return [col for col, d in diagnosis.items() if d["imputation_required"]]


def format_diagnosis_report(diagnosis: dict) -> str:
    """Plain-text rendering matching the spec's own worked-example format for shampoo.csv."""
    lines = []
    for col, d in diagnosis.items():
        lines.append(f"{col}:")
        lines.append(f"  Observed values: {d['observed_values']}")
        lines.append(f"  Missing values: {d['missing_values']}")
        lines.append(f"  Missing percentage: {d['missing_percentage']}%")
        lines.append(f"  Inserted monthly periods (missing): {d['missing_from_inserted_periods']}")
        lines.append(f"  Imputation required: {'Yes' if d['imputation_required'] else 'No'}")
    return "\n".join(lines)


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

    print("=" * 60)
    print("TEST 1: real shampoo.csv -- must match the spec's own literal expected output")
    print("=" * 60)
    df = pd.read_csv("shampoo.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(df, "Month", "-", component_meanings)
    sorted_df, _ = sort_mod.sort_time_series(split_df, split_report["component_columns"])
    complete_df, _ = gaps_mod.build_complete_sequence(sorted_df, split_report["component_columns"], "Month", ["Sales"])

    diagnosis = diagnose_missing_values(complete_df, ["Sales"])
    print(format_diagnosis_report(diagnosis))
    print()
    print("Columns needing imputation:", columns_needing_imputation(diagnosis), "(should be [] -- empty)")
    print("Full diagnosis dict:", diagnosis["Sales"])

    print("\n" + "=" * 60)
    print("TEST 2: synthetic series with isolated gap + a run + leading + trailing + mixed causes")
    print("=" * 60)
    # 12 months: leading gap at steps 1-2, isolated gap at step 5, a
    # 3-run at steps 8-10, trailing gap at step 12. Steps 8-9 are
    # INSERTED periods (never existed in source), step 10 is a
    # genuinely blank cell in an original row.
    synth = pd.DataFrame(
        {
            "Sales": [pd.NA, pd.NA, 100.0, 110.0, pd.NA, 130.0, 140.0, pd.NA, pd.NA, pd.NA, 200.0, pd.NA],
            "Time_Step": list(range(1, 13)),
            "_original_record_present": [True, True, True, True, True, True, True, True, False, False, True, True],
        }
    )
    diagnosis2 = diagnose_missing_values(synth, ["Sales"])
    print(format_diagnosis_report(diagnosis2))
    print()
    d = diagnosis2["Sales"]
    print("Isolated missing (should be 1 -- step 5):", d["isolated_missing_observations"])
    print("Consecutive runs (should be [2, 3] -- steps 1-2 leading run, steps 8-10 run):", d["consecutive_missing_runs"])
    print("Longest run (should be 3):", d["longest_missing_run"])
    print("Missing at beginning (should be 2):", d["missing_at_beginning"])
    print("Missing at end (should be 1):", d["missing_at_end"])
    print("Missing from inserted periods (should be 2 -- steps 8,9):", d["missing_from_inserted_periods"])
    print("Missing originally in dataset (should be 4 -- steps 1,2,5,10... wait check):", d["missing_originally_in_dataset"])
    print()
    print("Columns needing imputation:", columns_needing_imputation(diagnosis2), "(should be ['Sales'])")
