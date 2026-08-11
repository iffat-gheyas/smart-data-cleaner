"""
Module R: Duplicate Time-Period Detection & Resolution
----------------------------------------------------------
Section 5 of the spec. A "duplicate period" means two or more VALID
rows share the exact same temporal-ordering values (e.g. two rows
both '1-05') -- this is a different problem from Section 6's missing
periods, and a different problem from ordinary exact-duplicate-row
detection (Module C, 03_duplicate_removal.py): two shampoo-sales rows
for the same month could have genuinely different Sales values (e.g.
two regions' figures that need summing), so this is never safe to
resolve the way Module C resolves plain duplicate rows.

CORE RULE: the cleaner must not automatically remove or collapse
duplicates -- the client chooses the method, because (per the spec's
own example) sum vs mean depends on what the duplicate rows actually
MEAN, which the cleaner has no way to know on its own.

Only VALID rows (per Module P's validation) participate -- a row
whose time components failed validation has no reliable period to
compare against anything, so it's excluded here the same way it was
excluded from sorting in Module Q, and stays untouched.
"""

import pandas as pd

RESOLUTION_METHODS = [
    "keep_first",
    "keep_last",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "manual",
    "stop",
]


class StopProcessingRequested(Exception):
    """Raised when the client explicitly chooses 'stop processing' for duplicate periods."""
    pass


def detect_duplicate_periods(df: pd.DataFrame, sort_columns: list) -> dict:
    """
    Looks only at rows where _time_component_valid is True. Returns a
    report describing which periods are duplicated and how many rows
    each has -- detection only, no resolution yet, so the client can
    see exactly what they're deciding about before picking a method.
    """
    valid_df = df[df["_time_component_valid"]]
    group_sizes = valid_df.groupby(sort_columns, dropna=False).size()
    duplicated_periods = group_sizes[group_sizes > 1]

    duplicate_details = []
    for period_key, count in duplicated_periods.items():
        if not isinstance(period_key, tuple):
            period_key = (period_key,)
        mask = pd.Series(True, index=valid_df.index)
        for col, val in zip(sort_columns, period_key):
            mask &= valid_df[col] == val
        matching_rows = valid_df[mask]
        duplicate_details.append(
            {
                "period": dict(zip(sort_columns, period_key)),
                "n_rows": int(count),
                "row_positions": (matching_rows.index + 2).tolist(),  # +2: header row, 1-indexed
            }
        )

    return {
        "n_duplicate_periods": len(duplicate_details),
        "n_rows_involved": sum(d["n_rows"] for d in duplicate_details),
        "duplicate_details": duplicate_details,
    }


def resolve_duplicate_periods(
    df: pd.DataFrame, sort_columns: list, value_columns: list, method: str
) -> tuple:
    """
    Returns (resolved_df, report). Raises StopProcessingRequested if
    method == "stop", so the caller (app.py) can halt the pipeline
    cleanly and explain why, rather than silently continuing.

    "manual": nothing is resolved -- every row in a duplicate group is
    left exactly as-is, flagged, and returned for the client to sort
    out directly (e.g. by editing the source file), same philosophy
    as invalid rows in Module P.

    "keep_first" / "keep_last": relies on Module Q having already
    sorted with a STABLE sort, so rows within the same period are
    still in their original relative order -- "first"/"last" means
    first/last as they appeared in the client's own file, since there's
    no other basis to order same-period rows by.

    Aggregation methods (sum/mean/median/min/max): collapse each
    duplicate group into ONE row. The temporal component columns are
    identical across the group by definition, so they're kept as-is;
    only the value column(s) get aggregated. A "_period_duplicate_count"
    column records how many original rows fed into each output row --
    1 for rows that were never duplicated, 2+ for resolved groups --
    so this is visible in the delivered file, not just the report.
    """
    if method not in RESOLUTION_METHODS:
        raise ValueError(f"Unknown duplicate-resolution method '{method}'. Choose from {RESOLUTION_METHODS}.")

    if method == "stop":
        raise StopProcessingRequested(
            "Processing stopped at the client's request: duplicate time periods were found and "
            "'stop processing' was chosen instead of a resolution method."
        )

    detection = detect_duplicate_periods(df, sort_columns)

    if detection["n_duplicate_periods"] == 0:
        result = df.copy()
        result["_period_duplicate_count"] = 1
        return result, {**detection, "method_applied": method, "n_groups_resolved": 0}

    invalid_df = df[~df["_time_component_valid"]].copy()
    invalid_df["_period_duplicate_count"] = 1

    valid_df = df[df["_time_component_valid"]].copy()
    group_keys = list(zip(*[valid_df[c] for c in sort_columns])) if len(sort_columns) > 1 else valid_df[sort_columns[0]].tolist()
    valid_df["_period_group_key"] = group_keys

    if method == "manual":
        valid_df["_period_duplicate_count"] = valid_df.groupby("_period_group_key")["_period_group_key"].transform("count")
        result = pd.concat([valid_df.drop(columns=["_period_group_key"]), invalid_df], ignore_index=True)
        return result, {**detection, "method_applied": "manual", "n_groups_resolved": 0}

    if method in ("keep_first", "keep_last"):
        keep = "first" if method == "keep_first" else "last"
        counts = valid_df.groupby("_period_group_key")["_period_group_key"].transform("count")
        valid_df["_period_duplicate_count"] = counts
        resolved = valid_df.drop_duplicates(subset="_period_group_key", keep=keep)
        resolved = resolved.drop(columns=["_period_group_key"])
        result = pd.concat([resolved, invalid_df], ignore_index=True)
        return result, {**detection, "method_applied": method, "n_groups_resolved": detection["n_duplicate_periods"]}

    # Aggregation methods: sum / mean / median / min / max
    agg_map = {"sum": "sum", "mean": "mean", "median": "median", "min": "min", "max": "max"}
    non_value_cols = [c for c in valid_df.columns if c not in value_columns and c != "_period_group_key"]

    agg_rows = []
    for group_key, group_df in valid_df.groupby("_period_group_key", sort=False):
        row = group_df.iloc[0][non_value_cols].to_dict()
        for value_col in value_columns:
            row[value_col] = group_df[value_col].agg(agg_map[method])
        row["_period_duplicate_count"] = len(group_df)
        agg_rows.append(row)

    resolved = pd.DataFrame(agg_rows)
    # Restore original column order (dict-based row construction above doesn't guarantee it)
    resolved = resolved[[c for c in df.columns if c in resolved.columns] + ["_period_duplicate_count"]]
    result = pd.concat([resolved, invalid_df], ignore_index=True)

    return result, {**detection, "method_applied": method, "n_groups_resolved": detection["n_duplicate_periods"]}


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

    component_meanings = [
        "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
        "Month",
    ]

    print("=" * 60)
    print("TEST 1: real shampoo.csv -- no duplicate periods expected")
    print("=" * 60)
    df = pd.read_csv("shampoo.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(df, "Month", "-", component_meanings)
    sorted_df, sort_report = sort_mod.sort_time_series(split_df, split_report["component_columns"])
    sort_columns = sort_mod.get_temporal_sort_columns(split_report["component_columns"])
    detection = detect_duplicate_periods(sorted_df, sort_columns)
    print("Duplicate periods found:", detection["n_duplicate_periods"], "(should be 0)")

    print("\n" + "=" * 60)
    print("TEST 2: synthetic duplicate periods -- test every resolution method")
    print("=" * 60)
    dup_df = pd.DataFrame(
        {
            "Month": ["1-01", "1-02", "1-02", "1-03", "1-02"],
            "Sales": [100.0, 50.0, 70.0, 200.0, 30.0],
        }
    )
    dup_split, dup_split_report = split_mod.split_and_validate_time_column(dup_df, "Month", "-", component_meanings)
    dup_sorted, _ = sort_mod.sort_time_series(dup_split, dup_split_report["component_columns"])
    dup_sort_cols = sort_mod.get_temporal_sort_columns(dup_split_report["component_columns"])

    detection = detect_duplicate_periods(dup_sorted, dup_sort_cols)
    print("Duplicate periods found:", detection["n_duplicate_periods"])
    print("Details:", detection["duplicate_details"])

    for method in ["sum", "mean", "median", "min", "max", "keep_first", "keep_last"]:
        resolved, report = resolve_duplicate_periods(dup_sorted, dup_sort_cols, ["Sales"], method)
        print(f"\nMethod: {method}")
        print(resolved[["Month", "Relative_Year", "Month_Number", "Sales", "_period_duplicate_count"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST 3: 'manual' -- nothing resolved, all rows kept, flagged")
    print("=" * 60)
    resolved_manual, _ = resolve_duplicate_periods(dup_sorted, dup_sort_cols, ["Sales"], "manual")
    print(resolved_manual[["Month", "Sales", "_period_duplicate_count"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST 4: 'stop' -- must raise, must not silently continue")
    print("=" * 60)
    try:
        resolve_duplicate_periods(dup_sorted, dup_sort_cols, ["Sales"], "stop")
        print("BUG: did not raise!")
    except StopProcessingRequested as e:
        print("Correctly stopped:", e)
