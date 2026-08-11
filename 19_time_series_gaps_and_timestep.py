"""
Module S: Missing Time-Step Detection & Canonical Time_Step
----------------------------------------------------------------
Section 6 of the spec (missing PERIODS, distinct from missing VALUES
-- a period can be entirely absent from the file, or present but with
a blank Sales cell; these are different problems and this module only
handles the first one). Also finishes Section 3's canonical Time_Step,
since Section 21's own build order requires the complete gap-filled
sequence to exist FIRST.

TWO STRUCTURAL MODES:
  "modular" -- the ORIGINAL logic, completely unchanged: Year+Month,
    Year+Quarter, Year-only, or Relative_Period_Index-only. Simple
    modular arithmetic (year * cycle_length + cycle_value), no real
    calendar math needed since these don't have variable-length
    sub-periods to worry about.
  "datetime" -- NEW: anything involving Day, Hour, Minute, or Second.
    Uses Module AD's (30_time_series_unit_hierarchy) fixed-calendar
    arithmetic to build a genuine sequential tick count, correctly
    rolling over day->month->year boundaries (23:00 -> 00:00 the next
    day, Dec 31 -> Jan 1 the next year, etc.) -- this is exactly why
    real calendar math is needed here and wasn't for the modular case:
    months and years don't have fixed lengths the way "12 months in a
    year" or "4 quarters in a year" do.

The FINEST unit actually present (via Module AD's unit_rank ordering)
determines BOTH which mode applies and the frequency label -- e.g.
Year+Month+Day+Hour+Minute+Second all present at once correctly
detects "Secondly" as the true sampling frequency, not "Monthly" just
because Year+Month also happen to be present (a real, confirmed bug
in the pre-datetime-mode version of this module).

For each inserted (previously-absent) period, the row's value
column(s) are left genuinely missing (NaN) -- never estimated here.
Section 21 is explicit: imputation only happens later, after the
client has chosen a method.
"""

from importlib import import_module

import pandas as pd

unit_hierarchy = import_module("30_time_series_unit_hierarchy")
interval_mod = import_module("32_time_series_interval_detection")

CYCLE_LENGTHS = {
    "Quarter_Number": 4,
    "Month_Number": 12,
}
CYCLE_LABELS = {
    "Quarter_Number": "Quarter",
    "Month_Number": "Month",
}
YEAR_COLUMNS = {"Calendar_Year", "Relative_Year"}

# Which output column names correspond to which datetime unit, for the
# NEW datetime mode. Checked in this order; first match wins per unit.
DATETIME_UNIT_CANDIDATES = {
    "year": ["Calendar_Year", "Relative_Year"],
    "month": ["Month_Number"],
    "day": ["Day_Number"],
    "hour": ["Hour"],
    "minute": ["Minute"],
    "second": ["Second"],
}
# Finest-to-coarsest, for picking the frequency label -- deliberately
# only the sub-monthly units here, since Month/Year alone are still
# handled by the "modular" mode above (unchanged).
DATETIME_FINENESS_ORDER = ["second", "minute", "hour", "day"]
UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


class UnsupportedTemporalStructure(Exception):
    """Raised when the confirmed component structure isn't one of the currently-supported patterns."""
    pass


def _find_datetime_unit_columns(output_names: list) -> dict:
    """Returns {unit_name: output_column_name} for whichever datetime units are actually present."""
    found = {}
    for unit, candidates in DATETIME_UNIT_CANDIDATES.items():
        for candidate in candidates:
            if candidate in output_names:
                found[unit] = candidate
                break
    return found


def _compute_position_column(df: pd.DataFrame, freq_mode_info: dict) -> pd.Series:
    """
    Computes each row's position in the relevant NATIVE counting
    system -- shared by detect_original_frequency() (for interval
    detection) and build_complete_sequence() (for actually building
    the grid), so the two can never drift apart on how positions are
    computed.

    "modular" mode: exact calendar units (year*cycle_length+cycle, or
    year alone, or the raw Relative_Period_Index value) -- these are
    EXACT, no rounding, since months/quarters/years are being counted
    directly rather than approximated as a fixed number of days.

    "datetime" mode: RAW absolute seconds (Module AD's fixed-calendar
    arithmetic) -- deliberately NOT pre-divided by any unit. Interval
    detection (Module AF/32) works directly on these raw second values
    and finds whatever the true interval is (e.g. 900 seconds for
    genuinely 15-minute data), rather than assuming the step is
    exactly 1 tick of whichever unit happens to vary.
    """
    if freq_mode_info["mode"] == "datetime":
        units = freq_mode_info["units"]
        return df.apply(lambda row: _row_to_absolute_seconds(row, units), axis=1)

    year_col, cycle_col, cycle_length = freq_mode_info["year_col"], freq_mode_info["cycle_col"], freq_mode_info["cycle_length"]

    def period_index(row):
        if cycle_col is None:
            return int(row[year_col])
        return int(row[year_col]) * cycle_length + int(row[cycle_col])

    return df.apply(period_index, axis=1)


def detect_original_frequency(component_columns: list, df: pd.DataFrame = None) -> dict:
    """
    component_columns: (meaning, output_name) pairs from Module P.
    df: the actual data. Used for TWO things now:
      1. Checking whether each candidate datetime unit's column
         genuinely VARIES (as before) -- a constant column (e.g. Day
         always =1, a monthly-data convention) is excluded from mode
         selection.
      2. NEW: computing the actual position of every valid row and
         running Module AF/32's dominant-interval detection on the
         real consecutive differences -- replacing the old assumption
         that consecutive observations are exactly 1 tick apart. This
         is the fix for the confirmed flaw where genuinely 15-minute
         (or genuinely quarterly-but-expressed-via-Month) data was
         misdetected as 1-minute (or monthly).

    Returns a dict with "frequency_label", "mode" ("modular" or
    "datetime"), the mode-specific driver columns as before, PLUS (when
    df is provided and has "_time_component_valid"): "interval" (the
    detected step size, in the mode's native units -- exact calendar
    units for modular, raw seconds for datetime), "confidence"
    ("high"/"low"), and "consistency_ratio". Without df (or without
    enough valid rows), falls back to interval=1 with confidence=None,
    for backward compatibility and to avoid crashing on an unexpected
    caller.
    """
    output_names = [name for _, name in component_columns]

    datetime_units = _find_datetime_unit_columns(output_names)

    if df is not None:
        varying_datetime_units = {
            unit: col for unit, col in datetime_units.items()
            if col in df.columns and df[col].nunique(dropna=True) > 1
        }
    else:
        varying_datetime_units = datetime_units

    finest_datetime_unit = next((u for u in DATETIME_FINENESS_ORDER if u in varying_datetime_units), None)

    if finest_datetime_unit is not None:
        freq_info = {
            "mode": "datetime",
            "units": datetime_units,  # ALL present units, including constant ones -- still needed for absolute-position arithmetic
            "finest_unit": finest_datetime_unit,
            # kept as None so any code that still reads the old modular
            # keys directly (rather than checking "mode" first) fails
            # loudly instead of silently using nonsense values.
            "year_col": None,
            "cycle_col": None,
            "cycle_length": None,
        }
    else:
        # --- Unchanged "modular" mode below (Year+Month / Year+Quarter / Year-only / Relative_Period_Index-only) ---
        year_col = next((c for c in output_names if c in YEAR_COLUMNS), None)
        cycle_col = next((c for c in output_names if c in CYCLE_LENGTHS), None)

        if year_col and cycle_col:
            freq_info = {
                "mode": "modular",
                "year_col": year_col,
                "cycle_col": cycle_col,
                "cycle_length": CYCLE_LENGTHS[cycle_col],
            }
        elif year_col:
            freq_info = {"mode": "modular", "year_col": year_col, "cycle_col": None, "cycle_length": None}
        elif "Relative_Period_Index" in output_names:
            freq_info = {"mode": "modular", "year_col": "Relative_Period_Index", "cycle_col": None, "cycle_length": None}
        else:
            raise UnsupportedTemporalStructure(
                f"This module currently supports Year+Month, Year+Quarter, Year-only, "
                f"Relative_Period_Index-only, or any combination involving Day/Hour/Minute/Second. "
                f"Detected component columns: {output_names}. Week-based sequences (without any finer "
                f"unit alongside) aren't built yet -- flag this dataset rather than guessing at its "
                f"regular sequence."
            )

    # --- NEW: detect the actual interval from real consecutive differences ---
    if df is not None and "_time_component_valid" in df.columns:
        valid_df = df[df["_time_component_valid"]]
        if len(valid_df) >= 2:
            positions = _compute_position_column(valid_df, freq_info)
            sorted_unique_positions = sorted(positions.unique())
            interval_result = interval_mod.detect_dominant_interval(sorted_unique_positions)
            freq_info["interval"] = interval_result["interval"]
            freq_info["confidence"] = interval_result["confidence"]
            freq_info["consistency_ratio"] = interval_result["consistency_ratio"]
        else:
            freq_info["interval"] = 1
            freq_info["confidence"] = None
            freq_info["consistency_ratio"] = None
    else:
        freq_info["interval"] = 1
        freq_info["confidence"] = None
        freq_info["consistency_ratio"] = None

    # --- Frequency label reflects the ACTUAL detected interval, not just the unit name ---
    if freq_info["mode"] == "datetime":
        unit_name = finest_datetime_unit.capitalize()
        interval_in_unit = freq_info["interval"] / UNIT_SECONDS[finest_datetime_unit]
    elif freq_info["cycle_col"]:
        unit_name = CYCLE_LABELS[freq_info["cycle_col"]]
        interval_in_unit = freq_info["interval"]
    elif freq_info["year_col"] == "Relative_Period_Index":
        unit_name = "Period"
        interval_in_unit = freq_info["interval"]
    else:
        unit_name = "Year"
        interval_in_unit = freq_info["interval"]

    interval_display = f"{interval_in_unit:g}"
    plural = "" if interval_in_unit == 1 else "s"
    freq_info["frequency_label"] = f"{interval_display} {unit_name}{plural}"
    freq_info["unit_name"] = unit_name
    freq_info["interval_in_unit"] = interval_in_unit

    return freq_info


def _row_to_absolute_seconds(row, units: dict) -> int:
    """
    Converts one row's present datetime components into a single
    absolute second count, using Module AD's fixed 365-day calendar.
    Missing coarser components default sensibly (month=1, day=1,
    hour=minute=second=0) -- e.g. a Day+Hour+Minute+Second dataset
    with no Year/Month present just treats every day as being in the
    same implicit year/month, which is fine since only DIFFERENCES
    between rows matter for building the sequence, not the absolute
    value itself.
    """
    year = int(row[units["year"]]) if "year" in units else 0
    month = int(row[units["month"]]) if "month" in units else 1
    day = int(row[units["day"]]) if "day" in units else 1
    hour = int(row[units["hour"]]) if "hour" in units else 0
    minute = int(row[units["minute"]]) if "minute" in units else 0
    second = int(row[units["second"]]) if "second" in units else 0

    day_in_year = unit_hierarchy.day_of_year(month, day)
    total_days = year * unit_hierarchy.DAYS_PER_YEAR + (day_in_year - 1)
    return (
        total_days * unit_hierarchy.SECONDS_PER_DAY
        + hour * unit_hierarchy.SECONDS_PER_HOUR
        + minute * unit_hierarchy.SECONDS_PER_MINUTE
        + second
    )


def _absolute_seconds_to_components(total_seconds: int, units: dict) -> dict:
    """
    The inverse of _row_to_absolute_seconds() -- given an absolute
    second count, reconstructs values for whichever datetime columns
    are actually configured. Used to fill in an inserted (previously-
    absent) row's time columns with a genuine, correctly-computed
    date/time, not a placeholder.
    """
    total_days, remainder_seconds = divmod(total_seconds, unit_hierarchy.SECONDS_PER_DAY)
    hour, rem = divmod(remainder_seconds, unit_hierarchy.SECONDS_PER_HOUR)
    minute, second = divmod(rem, unit_hierarchy.SECONDS_PER_MINUTE)
    year, day_in_year_0indexed = divmod(total_days, unit_hierarchy.DAYS_PER_YEAR)
    month, day = unit_hierarchy.month_day_from_day_of_year(day_in_year_0indexed + 1)

    result = {}
    if "year" in units:
        result[units["year"]] = year
    if "month" in units:
        result[units["month"]] = month
    if "day" in units:
        result[units["day"]] = day
    if "hour" in units:
        result[units["hour"]] = hour
    if "minute" in units:
        result[units["minute"]] = minute
    if "second" in units:
        result[units["second"]] = second
    return result


def build_complete_sequence(
    df: pd.DataFrame, component_columns: list, time_columns, value_columns: list
) -> tuple:
    """
    Returns (result_df, report).

    time_columns: a single column name (str) or a list of column
    names -- accepts both for backward compatibility with the
    single-time-column callers. Functionally this barely matters: the
    inserted-row dict below already initializes EVERY column to NA,
    time column(s) included, so there's no real per-column logic
    needed here regardless of how many time columns exist.

    result_df: every originally-valid row, PLUS one inserted row for
    every absent period between the observed minimum and maximum,
    sorted, with Time_Step = 0..N-1 assigned across the complete
    sequence (0-indexed, per the updated spec -- the earliest
    timestamp is Time_Step 0). Steps by the ACTUAL DETECTED interval
    (Module AF/32's dominant-interval detection), not an assumed 1
    tick -- this is what correctly handles e.g. genuinely 15-minute
    data (steps of 15) or genuinely-quarterly-but-expressed-via-Month
    data (steps of 3), instead of always assuming step=1.
    Originally-invalid rows (Module P) are appended at the end,
    untouched, with no Time_Step (their true position is still
    unknown, same reasoning as Module Q).
    """
    freq_info = detect_original_frequency(component_columns, df)

    valid_df = df[df["_time_component_valid"]].copy()
    invalid_df = df[~df["_time_component_valid"]].copy()

    if valid_df.empty:
        raise ValueError(
            "No rows have a validly-parsed time stamp -- every single row failed time-component "
            "validation (see the invalid-rows table above for why). There's nothing to build a "
            "time sequence from until at least some rows are fixed at the source and re-uploaded, "
            "or the component meanings above are reconsidered."
        )

    valid_df["_period_index"] = _compute_position_column(valid_df, freq_info)
    interval = freq_info["interval"]

    min_idx = int(valid_df["_period_index"].min())
    max_idx = int(valid_df["_period_index"].max())
    expected_indices = set(range(min_idx, max_idx + 1, interval))
    observed_indices = set(valid_df["_period_index"])
    missing_indices = sorted(expected_indices - observed_indices)

    valid_df["_time_step_inserted"] = False
    valid_df["_original_record_present"] = True

    # Any OTHER component column (not directly driven by the mode's
    # own step logic above) that happens to be CONSTANT across every
    # valid row -- e.g. a Day column always =1 in monthly data, or a
    # Year column when Month/Day/Hour/etc. are the datetime-mode
    # drivers -- gets backfilled with that known constant value on
    # inserted rows too. This is filling in something we know with
    # certainty (every real row agrees), not guessing at genuinely
    # unknown data -- leaving it NA there would just be needlessly
    # inconsistent with the rest of the column.
    all_component_output_names = [name for _, name in component_columns if name is not None]
    driven_by_mode = set(freq_info["units"].values()) if freq_info["mode"] == "datetime" else {
        c for c in (freq_info.get("year_col"), freq_info.get("cycle_col")) if c
    }
    constant_backfill = {}
    for col in all_component_output_names:
        if col in driven_by_mode or col not in valid_df.columns:
            continue
        unique_values = valid_df[col].dropna().unique()
        if len(unique_values) == 1:
            constant_backfill[col] = unique_values[0]

    inserted_rows = []
    for idx in missing_indices:
        row = {c: pd.NA for c in df.columns}  # already covers every time column, whatever its name(s)
        if freq_info["mode"] == "datetime":
            reconstructed = _absolute_seconds_to_components(idx, freq_info["units"])
            row.update(reconstructed)
        elif freq_info["cycle_col"] is None:
            row[freq_info["year_col"]] = idx
        else:
            row[freq_info["year_col"]] = idx // freq_info["cycle_length"]
            row[freq_info["cycle_col"]] = idx % freq_info["cycle_length"]
        for col, value in constant_backfill.items():
            row[col] = value
        for vc in value_columns:
            row[vc] = pd.NA
        row["_time_component_valid"] = True  # the PERIOD is validly defined; only the value is missing
        row["_time_component_invalid_reason"] = None
        row["_time_step_inserted"] = True
        row["_original_record_present"] = False
        row["_period_index"] = idx
        inserted_rows.append(row)

    inserted_df = pd.DataFrame(inserted_rows) if inserted_rows else pd.DataFrame(columns=valid_df.columns)

    complete_df = pd.concat([valid_df, inserted_df], ignore_index=True)
    complete_df = complete_df.sort_values("_period_index", kind="mergesort").reset_index(drop=True)

    # Time_Step is 0-indexed (the earliest timestamp is Time_Step 0),
    # per the updated spec -- since the grid is complete and evenly
    # spaced by construction (every missing multiple of `interval` was
    # just inserted above), plain sequential numbering after sorting
    # is exactly equivalent to (position - min_idx) / interval for
    # every row, just simpler to compute.
    complete_df["Time_Step"] = range(0, len(complete_df))
    complete_df["Time_Step_Unit"] = freq_info["unit_name"]
    complete_df["Original_Frequency"] = freq_info["frequency_label"]
    complete_df = complete_df.drop(columns=["_period_index"])
    # Explicit dtype fix: concatenating with an EMPTY inserted_df (the
    # common case -- most files have zero missing periods) silently
    # degrades these two flag columns to `object` dtype. On an object
    # column, `~True` does bitwise-NOT on the Python int underneath
    # (giving -2), not logical negation (False) -- a real bug that
    # only shows up once something downstream actually negates the
    # flag, so forcing proper bool dtype here rather than trusting
    # pandas to infer it correctly through the concat.
    complete_df["_time_step_inserted"] = complete_df["_time_step_inserted"].astype(bool)
    complete_df["_original_record_present"] = complete_df["_original_record_present"].astype(bool)

    invalid_df["_time_step_inserted"] = False
    invalid_df["_original_record_present"] = True
    invalid_df["Time_Step"] = pd.NA
    invalid_df["Time_Step_Unit"] = freq_info["unit_name"]
    invalid_df["Original_Frequency"] = freq_info["frequency_label"]

    result_df = pd.concat([complete_df, invalid_df], ignore_index=True)

    report = {
        "original_frequency": freq_info["frequency_label"],
        "n_observed_periods": len(observed_indices),
        "n_expected_periods": len(expected_indices),
        "n_missing_periods_inserted": len(missing_indices),
        "n_invalid_rows_excluded": len(invalid_df),
        "sequence_min_to_max": (min_idx, max_idx),
    }
    return result_df, report


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
    print("TEST 1: real shampoo.csv -- 0 missing periods, Time_Step should run 1..36")
    print("=" * 60)
    df = pd.read_csv("shampoo.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(df, "Month", "-", component_meanings)
    sorted_df, _ = sort_mod.sort_time_series(split_df, split_report["component_columns"])
    complete_df, report = build_complete_sequence(sorted_df, split_report["component_columns"], "Month", ["Sales"])
    print(report)
    print("Time_Step range:", complete_df["Time_Step"].min(), "-", complete_df["Time_Step"].max())
    print("Matches expected result (0 to 35, 0 missing):", report["n_missing_periods_inserted"] == 0)

    print("\n" + "=" * 60)
    print("TEST 2: the spec's own missing-period example (1-05, 1-06, 1-08 -- 1-07 absent)")
    print("=" * 60)
    gap_df = pd.DataFrame({"Month": ["1-05", "1-06", "1-08"], "Sales": [100.0, 110.0, 130.0]})
    gap_split, gap_split_report = split_mod.split_and_validate_time_column(gap_df, "Month", "-", component_meanings)
    gap_sorted, _ = sort_mod.sort_time_series(gap_split, gap_split_report["component_columns"])
    gap_complete, gap_report = build_complete_sequence(gap_sorted, gap_split_report["component_columns"], "Month", ["Sales"])
    print(gap_report)
    print(gap_complete[["Month", "Relative_Year", "Month_Number", "Sales", "Time_Step", "_time_step_inserted", "_original_record_present"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST 3: gap spanning a year boundary (1-11, 1-12, [2-01 missing], 2-02)")
    print("=" * 60)
    yr_gap_df = pd.DataFrame({"Month": ["1-11", "1-12", "2-02"], "Sales": [300.0, 310.0, 250.0]})
    yr_split, yr_split_report = split_mod.split_and_validate_time_column(yr_gap_df, "Month", "-", component_meanings)
    yr_sorted, _ = sort_mod.sort_time_series(yr_split, yr_split_report["component_columns"])
    yr_complete, yr_report = build_complete_sequence(yr_sorted, yr_split_report["component_columns"], "Month", ["Sales"])
    print(yr_report)
    print(yr_complete[["Month", "Relative_Year", "Month_Number", "Sales", "Time_Step", "_time_step_inserted"]].to_string(index=False))
