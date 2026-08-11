"""
Module U: Time-Series Value Column Validation
--------------------------------------------------
Section 9 of the spec. Two stages, matching the spec's own order:
  1. Confirm missing-value PLACEHOLDERS (e.g. 'NA', '999', '?') and
     convert confirmed ones to genuine missing values.
  2. Whatever non-numeric text is STILL left after that gets resolved
     one unique value at a time, with the client choosing what to do
     with it -- never a silent guess.

REUSE, not reinvention: stage 1 uses the EXACT SAME detection logic
already built for the cross-sectional pipeline (Module J's
detect_disguised_missing() and its DISGUISED_MISSING_CANDIDATES /
NUMERIC_DISGUISED_MISSING_CANDIDATES lists) via the same import_module
pattern Module L (12_apply_answers_and_run.py) already uses to wire
modules together. This keeps placeholder-recognition behaviour
IDENTICAL between the cross-sectional and time-series pipelines
rather than maintaining two subtly-different lists that drift apart
over time -- and it already covers essentially every placeholder the
spec names by name (empty, NA, N/A, NaN, null, None, missing, ?, -,
999, -999, 9999) plus more found from real client files.
"""

from importlib import import_module

import pandas as pd

profiling = import_module("10_column_profiling")

NON_NUMERIC_RESOLUTION_OPTIONS = [
    "Convert to a specific number",
    "Classify as missing",
    "Remove the affected row(s)",
    "Leave column non-numeric and exclude it from cleaning",
    "I'll correct the source data manually (leave as-is for now)",
]


def generate_missing_placeholder_question(series: pd.Series, column_name: str) -> dict:
    """
    Returns None (no question needed) when nothing looks like a
    placeholder -- same dynamic-question principle as the rest of the
    app: don't ask about something that isn't there.
    """
    markers = profiling.detect_disguised_missing(series)
    if not markers:
        return None
    return {
        "id": f"ts_missing_placeholders__{column_name}",
        "type": "multi_select",
        "text": (
            f"'{column_name}' contains these repeated values that might mean \"missing\" rather "
            f"than a real value. Select any that should be treated as missing:"
        ),
        "options": [f"'{marker}' ({count} occurrences)" for marker, count in markers.items()],
        "option_values": list(markers.keys()),  # positional match to "options" above
        "default_selected_all": True,  # every detected marker starts pre-selected, same as Module K
    }


def apply_confirmed_placeholders(df: pd.DataFrame, confirmed_placeholders: dict) -> tuple:
    """
    confirmed_placeholders: {column_name: [marker1, marker2, ...]}.
    Replaces confirmed markers with genuine NaN. Text match is
    case-insensitive/whitespace-trimmed for text columns; exact match
    for numeric columns -- same convention as the cross-sectional
    pipeline's equivalent function.
    """
    df = df.copy()
    log = []
    for col, markers in confirmed_placeholders.items():
        if not markers or col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_markers = set()
            for m in markers:
                try:
                    numeric_markers.add(float(m))
                except (TypeError, ValueError):
                    continue
            if numeric_markers:
                mask = df[col].isin(numeric_markers)
        else:
            text_markers = {str(m).strip().lower() for m in markers}
            mask = df[col].astype(str).str.strip().str.lower().isin(text_markers) & df[col].notna()

        n_replaced = int(mask.sum())
        if n_replaced:
            df.loc[mask, col] = pd.NA
            log.append(f"'{col}': converted {n_replaced} confirmed placeholder value(s) to missing")

    return df, log


def detect_remaining_non_numeric_values(df: pd.DataFrame, value_columns: list) -> dict:
    """
    For each value column, finds every unique value that still can't
    be parsed as a number after placeholder conversion. Reports count,
    affected Time_Step values, and sample row positions -- exactly the
    four things the spec asks to display before asking the client
    what to do.
    """
    report = {}
    for col in value_columns:
        if col not in df.columns:
            continue
        numeric_attempt = pd.to_numeric(df[col], errors="coerce")
        non_numeric_mask = numeric_attempt.isna() & df[col].notna()
        if not non_numeric_mask.any():
            continue

        affected = df[non_numeric_mask]
        entries = []
        for value, group in affected.groupby(col):
            entries.append(
                {
                    "value": value,
                    "count": len(group),
                    "affected_time_steps": group["Time_Step"].tolist() if "Time_Step" in group.columns else [],
                    "sample_row_positions": (group.index[:5] + 2).tolist(),
                }
            )
        report[col] = entries
    return report


def generate_non_numeric_resolution_question(column_name: str, value, count: int, affected_time_steps: list) -> dict:
    steps_preview = affected_time_steps[:5]
    return {
        "id": f"ts_non_numeric__{column_name}__{value}",
        "type": "single_select",
        "text": (
            f"'{column_name}' contains the non-numeric value '{value}' in {count} row(s) "
            f"(Time_Step {steps_preview}{'...' if len(affected_time_steps) > 5 else ''}). "
            f"How should this be handled?"
        ),
        "options": NON_NUMERIC_RESOLUTION_OPTIONS,
        "default": None,  # spec: must not choose automatically
    }


def apply_non_numeric_resolutions(df: pd.DataFrame, resolutions: dict) -> tuple:
    """
    resolutions: {(column_name, raw_value): {"action": ..., "replacement_number": optional float}}
    action in {"convert", "missing", "remove_row", "exclude_column", "manual"}.

    Returns (df, excluded_columns, log). excluded_columns is returned
    separately (not silently dropped from value_columns internally)
    so the caller keeps explicit control over what still gets treated
    as a value column for every later step (imputation, aggregation).
    """
    df = df.copy()
    excluded_columns = set()
    manual_columns = set()
    log = []

    # pandas' newer strict string dtype rejects assigning a float (or
    # pd.NA-as-missing) directly into a column still typed as string --
    # cast every column this function will touch to plain `object`
    # dtype first, so 'convert to a number' and 'classify as missing'
    # can actually hold mixed types during resolution. The LAST step,
    # finalize_numeric_conversion(), is what produces the real,
    # clean numeric dtype -- this function's job is just to resolve
    # each flagged value without pandas rejecting the assignment.
    touched_columns = {col for col, _ in resolutions.keys()}
    for col in touched_columns:
        if col in df.columns:
            df[col] = df[col].astype(object)

    for (col, raw_value), resolution in resolutions.items():
        action = resolution["action"]
        mask = df[col].astype(str) == str(raw_value)
        n_affected = int(mask.sum())
        if n_affected == 0:
            continue

        if action == "convert":
            replacement = resolution["replacement_number"]
            df.loc[mask, col] = replacement
            log.append(f"'{col}': converted {n_affected} occurrence(s) of '{raw_value}' to {replacement}")
        elif action == "missing":
            df.loc[mask, col] = pd.NA
            log.append(f"'{col}': classified {n_affected} occurrence(s) of '{raw_value}' as missing")
        elif action == "remove_row":
            df = df[~mask]
            log.append(f"'{col}': removed {n_affected} row(s) containing '{raw_value}'")
        elif action == "exclude_column":
            excluded_columns.add(col)
            log.append(f"'{col}': excluded from cleaning entirely due to non-numeric value '{raw_value}'")
        elif action == "manual":
            manual_columns.add(col)
            log.append(f"'{col}': '{raw_value}' left as-is for manual correction by the client")
        else:
            raise ValueError(f"Unknown resolution action '{action}' for {col}='{raw_value}'")

    df = df.reset_index(drop=True)
    return df, sorted(excluded_columns), sorted(manual_columns), log


def finalize_numeric_conversion(df: pd.DataFrame, value_columns: list, excluded_columns: list, manual_columns: list) -> tuple:
    """
    Casts every value column that ISN'T excluded or flagged for manual
    correction to genuine numeric dtype. Uses errors='coerce' as a
    safety net (should be a no-op if every non-numeric value was
    already resolved in the prior step) but reports anything that
    still failed, rather than pretending nothing could go wrong.
    """
    df = df.copy()
    log = []
    unexpected_failures = {}

    for col in value_columns:
        if col in excluded_columns or col in manual_columns:
            continue
        before_non_null = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_non_null = df[col].notna().sum()
        if after_non_null < before_non_null:
            unexpected_failures[col] = int(before_non_null - after_non_null)
        else:
            log.append(f"'{col}': confirmed fully numeric")

    return df, {"log": log, "unexpected_failures": unexpected_failures}


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: real shampoo.csv Sales column -- already clean, expect no questions")
    print("=" * 60)
    df = pd.read_csv("shampoo.csv")
    q = generate_missing_placeholder_question(df["Sales"], "Sales")
    print("Placeholder question:", q, "(should be None -- no markers detected)")
    remaining = detect_remaining_non_numeric_values(df.assign(Time_Step=range(1, 37)), ["Sales"])
    print("Remaining non-numeric values report:", remaining, "(should be empty dict)")

    print("\n" + "=" * 60)
    print("TEST 2: synthetic messy Sales column -- placeholders + genuine text junk")
    print("=" * 60)
    messy_df = pd.DataFrame(
        {
            "Sales": ["100.5", "NA", "150.2", "999", "twenty", "200.0", "NA", "ERR", "180.3", "NA"],
            "Time_Step": list(range(1, 11)),
        }
    )
    q2 = generate_missing_placeholder_question(messy_df["Sales"], "Sales")
    print("Placeholder question:", q2["text"] if q2 else None)
    if q2:
        print("Options:", q2["options"])

    # Simulate the client confirming 'NA' (not '999', since maybe 999 is a real sales figure this time -- testing that the client's CHOICE matters, not an automatic assumption)
    confirmed = {"Sales": ["NA"]}
    after_placeholders, placeholder_log = apply_confirmed_placeholders(messy_df, confirmed)
    print("\nAfter placeholder conversion:", placeholder_log)
    print(after_placeholders["Sales"].tolist())

    remaining2 = detect_remaining_non_numeric_values(after_placeholders, ["Sales"])
    print("\nRemaining non-numeric values:")
    for col, entries in remaining2.items():
        for e in entries:
            print(" ", e)

    print("\nGenerating resolution questions for each remaining non-numeric value:")
    resolutions = {}
    for col, entries in remaining2.items():
        for e in entries:
            rq = generate_non_numeric_resolution_question(col, e["value"], e["count"], e["affected_time_steps"])
            print(" -", rq["text"])
            # simulate client choices: '999' -> keep as real number (convert to itself); 'twenty' -> convert to 20; 'ERR' -> remove row
            if e["value"] == "999":
                resolutions[(col, e["value"])] = {"action": "convert", "replacement_number": 999.0}
            elif e["value"] == "twenty":
                resolutions[(col, e["value"])] = {"action": "convert", "replacement_number": 20.0}
            elif e["value"] == "ERR":
                resolutions[(col, e["value"])] = {"action": "remove_row"}

    resolved_df, excluded_cols, manual_cols, resolution_log = apply_non_numeric_resolutions(after_placeholders, resolutions)
    print("\nResolution log:", resolution_log)
    print("Excluded columns:", excluded_cols, " Manual columns:", manual_cols)
    print(resolved_df[["Sales", "Time_Step"]].to_string(index=False))

    final_df, final_report = finalize_numeric_conversion(resolved_df, ["Sales"], excluded_cols, manual_cols)
    print("\nFinal numeric conversion:", final_report)
    print("Final dtype:", final_df["Sales"].dtype)
    print(final_df[["Sales", "Time_Step"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST 3: 'exclude column' and 'manual' actions -- must not force-convert")
    print("=" * 60)
    junk_df = pd.DataFrame({"Notes_Value": ["ok", "bad_data", "also_bad"], "Time_Step": [1, 2, 3]})
    junk_resolutions = {
        ("Notes_Value", "bad_data"): {"action": "exclude_column"},
    }
    junk_resolved, junk_excluded, junk_manual, junk_log = apply_non_numeric_resolutions(junk_df, junk_resolutions)
    print("Excluded:", junk_excluded)
    final_junk, final_junk_report = finalize_numeric_conversion(junk_resolved, ["Notes_Value"], junk_excluded, junk_manual)
    print("Column untouched (still text, not coerced to NaN):", final_junk["Notes_Value"].tolist())
