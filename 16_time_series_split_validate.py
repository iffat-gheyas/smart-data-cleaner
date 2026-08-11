"""
Module P: Time Stamp Splitting & Validation
----------------------------------------------
Section 2 of the spec. Takes the CONFIRMED answers from Module O
(separator, and each component's meaning) and actually splits the raw
time column into validated, meaningfully-named columns.

CORE RULE, directly from the spec: "Do not silently correct invalid
time components." So this module never clamps, guesses, or drops a
bad value on its own. Every row is classified as valid or invalid;
invalid rows get their component columns left as NaN plus a specific
reason, but the row itself stays in the dataframe and the ORIGINAL raw
value is always preserved untouched (spec: "the original time column
must always be retained for auditing"). What to actually DO about
invalid rows (exclude, fix manually, ask the client) is a decision for
a later step, once duplicate/gap detection also needs to account for
them -- this module's job is only to detect and report, honestly.

Month/quarter parsing (numbers, names, abbreviations, quarter labels)
now comes from Module AD (30_time_series_unit_hierarchy) -- the single
source of truth, rather than a duplicated copy here that could drift
out of sync.
"""

from importlib import import_module

import pandas as pd

unit_hierarchy = import_module("30_time_series_unit_hierarchy")

# Maps a component's confirmed MEANING to (output column name, validator).
# Each validator takes a single raw string and returns (parsed_value, error_message).
# error_message is None when valid.


def _validate_int_range(raw: str, low: int, high: int, label: str):
    raw = raw.strip()
    if raw == "":
        return None, f"{label} is empty"
    try:
        value = int(raw)
    except ValueError:
        return None, f"{label} '{raw}' is not a whole number"
    if not (low <= value <= high):
        return None, f"{label} '{value}' is outside the valid range {low}-{high}"
    return value, None


def _validate_positive_int(raw: str, label: str):
    raw = raw.strip()
    if raw == "":
        return None, f"{label} is empty"
    try:
        value = int(raw)
    except ValueError:
        return None, f"{label} '{raw}' is not a whole number"
    if value <= 0:
        return None, f"{label} '{value}' must be a positive whole number"
    return value, None


def _validate_none(raw: str):
    # Identifiers and "custom" components are kept as-is, unvalidated --
    # they're explicitly not part of the time ORDERING, so there's
    # nothing objective to validate them against.
    return raw, None


COMPONENT_MEANING_RULES = {
    "Year (real calendar year)": ("Calendar_Year", lambda raw: _validate_positive_int(raw, "Calendar year")),
    "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)": (
        "Relative_Year",
        lambda raw: _validate_positive_int(raw, "Relative year"),
    ),
    "Quarter": ("Quarter_Number", lambda raw: unit_hierarchy.parse_quarter(raw)),
    "Month": ("Month_Number", lambda raw: unit_hierarchy.parse_month(raw)),
    "Week": ("Week_Number", lambda raw: _validate_int_range(raw, 1, 53, "Week")),
    "Day": ("Day_Number", lambda raw: _validate_int_range(raw, 1, 31, "Day")),
    "Hour": ("Hour", lambda raw: _validate_int_range(raw, 0, 23, "Hour")),
    "Minute": ("Minute", lambda raw: _validate_int_range(raw, 0, 59, "Minute")),
    "Second": ("Second", lambda raw: _validate_int_range(raw, 0, 59, "Second")),
    "Relative period index (e.g. period 1, 2, 3 ...)": (
        "Relative_Period_Index",
        lambda raw: _validate_positive_int(raw, "Relative period index"),
    ),
    "Identifier (not part of the time ordering)": ("Identifier", _validate_none),
    "Custom / other": ("Custom_Component", _validate_none),
    # "Component to ignore" deliberately has NO entry here -- handled
    # as a special case below (produces no output column at all).
}


def split_and_validate_time_column(
    df: pd.DataFrame, time_column: str, separator: str, component_meanings: list
) -> tuple:
    """
    component_meanings: list of meaning strings in POSITIONAL order,
    e.g. ["Relative year...", "Month"] for '1-01' -> Relative_Year, Month_Number.

    Returns (df_with_new_columns, report) where report is a dict with:
      - "output_columns": the new component column names actually created
      - "n_valid_rows", "n_invalid_rows"
      - "invalid_rows": a DataFrame of just the problem rows, WHY each
        one failed, and its original row position (for the client to
        locate it in their source file)
    """
    df = df.copy()
    original_time_series = df[time_column].copy()  # captured BEFORE any mutation -- see the collision note below
    n_components = len(component_meanings)

    # Column names: duplicate meanings (e.g. two "Custom" components)
    # need distinguishing suffixes so they don't collide.
    output_names = []
    seen = {}
    for meaning in component_meanings:
        if meaning == "Component to ignore":
            output_names.append(None)
            continue
        base_name, _ = COMPONENT_MEANING_RULES[meaning]
        seen[base_name] = seen.get(base_name, 0) + 1
        output_names.append(base_name if seen[base_name] == 1 else f"{base_name}_{seen[base_name]}")

    for name in output_names:
        if name is not None:
            df[name] = pd.NA

    invalid_reasons = [None] * len(df)
    # Read raw_values from the ORIGINAL series, captured before the
    # loop above ever touches df -- a raw column can share its name
    # with its own canonical output column (e.g. a client's column
    # literally named "Hour", mapped to the "Hour" meaning, whose
    # output name is ALSO "Hour"). Reading df[time_column] AFTER the
    # output-column creation loop would silently read back pd.NA
    # instead of the real raw values in that exact collision case --
    # confirmed as a real bug via a genuine electricity-data test.
    raw_values = original_time_series.astype(str)

    for row_idx, raw_value in raw_values.items():
        parts = raw_value.split(separator) if separator else [raw_value]

        if len(parts) != n_components:
            invalid_reasons[df.index.get_loc(row_idx)] = (
                f"expected {n_components} component(s) separated by '{separator}', "
                f"found {len(parts)} in '{raw_value}'"
            )
            continue

        row_values = {}
        row_error = None
        for part, meaning, out_name in zip(parts, component_meanings, output_names):
            if meaning == "Component to ignore":
                continue
            _, validator = COMPONENT_MEANING_RULES[meaning]
            value, error = validator(part)
            if error is not None:
                row_error = error
                break
            row_values[out_name] = value

        if row_error is not None:
            invalid_reasons[df.index.get_loc(row_idx)] = row_error
            continue

        for out_name, value in row_values.items():
            df.at[row_idx, out_name] = value

    df["_time_component_invalid_reason"] = invalid_reasons
    df["_time_component_valid"] = [r is None for r in invalid_reasons]

    invalid_mask = ~df["_time_component_valid"]
    invalid_rows = df.loc[invalid_mask, [time_column, "_time_component_invalid_reason"]].copy()
    invalid_rows.insert(0, "Original_Row_Number", invalid_rows.index + 2)  # +2: header row + 1-indexing, matches Module F's convention

    report = {
        "output_columns": [n for n in output_names if n is not None],
        "component_columns": list(zip(component_meanings, output_names)),
        "n_valid_rows": int(df["_time_component_valid"].sum()),
        "n_invalid_rows": int(invalid_mask.sum()),
        "invalid_rows": invalid_rows.reset_index(drop=True),
    }

    return df, report


if __name__ == "__main__":
    df = pd.read_csv("shampoo.csv")

    print("=" * 60)
    print("TEST 1: real shampoo.csv (should be 100% valid)")
    print("=" * 60)
    result_df, report = split_and_validate_time_column(
        df,
        time_column="Month",
        separator="-",
        component_meanings=[
            "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
            "Month",
        ],
    )
    print("Output columns created:", report["output_columns"])
    print("Valid rows:", report["n_valid_rows"], " Invalid rows:", report["n_invalid_rows"])
    print(result_df[["Month", "Relative_Year", "Month_Number", "_time_component_valid"]].head(5))
    print(result_df[["Month", "Relative_Year", "Month_Number"]].tail(3))

    print("\n" + "=" * 60)
    print("TEST 2: every invalid example from the spec itself")
    print("=" * 60)
    bad_df = pd.DataFrame(
        {
            "Month": ["1-00", "1-13", "A-03", "2-March", "2-", "1-01-05", "1-06"],
            "Sales": [100, 100, 100, 100, 100, 100, 100],
        }
    )
    bad_result, bad_report = split_and_validate_time_column(
        bad_df,
        time_column="Month",
        separator="-",
        component_meanings=[
            "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
            "Month",
        ],
    )
    print("Valid rows:", bad_report["n_valid_rows"], " Invalid rows:", bad_report["n_invalid_rows"])
    print()
    print(bad_report["invalid_rows"].to_string(index=False))
