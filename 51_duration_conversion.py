"""
Module: Duration/Age Standardisation -- Apply Conversion
------------------------------------------------------------
Sibling to Module 46 (weight/length). Built WITH the default-gate +
per-value design from the start (unlike weight/length and currency,
which both got this treatment as a later redesign) -- see Module 50's
docstring for the full reasoning, identical here:
  - confirm_duration__<col>       (yes/no -- is this really a duration column)
  - standardize_duration__<col>   (yes/no -- convert to one unit, or leave as-is)
  - target_duration_unit__<col>   (only when genuinely mixed units)
  - has_default_duration_unit__<col> / default_duration_unit__<col>
        (yes/no + picklist -- a blanket default for values with NO unit
        mentioned at all, e.g. every "42" in an Age column defaults to
        "years")
  - duration_value_resolution__<col>__<i>
        (per-value dropdown, case-by-case for anything the default
        doesn't cover -- ambiguous/unrecognised unit tokens ALWAYS go
        here regardless of any default, same reasoning as Modules 46/48)

Conversion ratios are FIXED (like Module 45's weight/length ratios, NOT
like Module 48's currency exchange rates) -- no rate question needed,
since 60 seconds always makes a minute. Month/year ratios are averages
(see Module 50's docstring) rather than exact calendar lengths.
"""

from importlib import import_module

import numpy as np
import pandas as pd

duration_std = import_module("50_duration_standardisation")
measurement_conv = import_module("46_unit_standardisation_conversion")

# Reused directly rather than duplicated -- same mixed-type-column fix
# and the same fully generic end-of-pipeline rename function.
_to_display_string = measurement_conv._to_display_string
apply_final_column_renames = measurement_conv.apply_final_column_renames


def convert_duration_value(number: float, from_unit: str, to_unit: str) -> float:
    """Converts a duration amount between two units via the fixed seconds-based ratios."""
    table = duration_std.DURATION_UNIT_TO_SECONDS
    if from_unit not in table or to_unit not in table:
        raise ValueError(f"Unrecognised duration unit: '{from_unit}' or '{to_unit}'. Known: {sorted(table)}")
    return number * table[from_unit] / table[to_unit]


def _resolve_effective_target_unit(col_info: dict, answers: dict, col: str):
    """Same priority order as Module 46's version, for the same reasoning."""
    answered = answers.get(f"target_duration_unit__{col}")
    if answered:
        return answered
    if col_info.get("implied_unit_from_name"):
        return col_info["implied_unit_from_name"]
    detected = col_info["detected_units"]
    if len(detected) == 1:
        return next(iter(detected))
    return None


def apply_duration_conversion(df: pd.DataFrame, profile: dict, answers: dict) -> tuple:
    """
    Runs Module 50's detection fresh against the CURRENT state of df,
    then, for every confirmed AND client-approved duration column,
    converts every value to a single target unit. Mirrors Module 46's
    apply_measurement_unit_conversion structure and safeguards exactly
    -- see that module for the fuller reasoning behind each one (the
    mixed-type mainlining fix, the "already documents this unit, skip
    rename" check, etc.), summarised briefly here.

    Returns (df, actions_log, standardized_units).
    """
    df = df.copy()
    actions_log = []
    standardized_units = {}

    detected_durations = duration_std.detect_duration_columns(df)

    for col, info in detected_durations.items():
        if col not in df.columns:
            continue

        confirm_answer = answers.get(f"confirm_duration__{col}")
        if confirm_answer is False:
            continue

        if not duration_std.needs_standardization(info):
            if pd.api.types.is_numeric_dtype(df[col]):
                profile[col]["dtype"] = "numeric"
                profile[col]["semantic_role"] = "numeric_measure"
                profile[col]["role_confidence"] = 1.0
            continue

        standardize_answer = answers.get(f"standardize_duration__{col}")
        if standardize_answer is False:
            df[col] = df[col].apply(lambda v: _to_display_string(v) if pd.notna(v) else v)
            actions_log.append(
                f"Left '{col}' as original text, per client's choice not to standardise duration "
                f"units (e.g. values like {info['detected_units'] or info['bare_number_examples']} "
                f"remain exactly as entered)"
            )
            continue

        implied_unit = info.get("implied_unit_from_name")

        has_default_answer = answers.get(f"has_default_duration_unit__{col}")
        blanket_default_unit = (
            answers.get(f"default_duration_unit__{col}") if has_default_answer is True else None
        )

        target_unit = _resolve_effective_target_unit(info, answers, col)
        if target_unit is None and blanket_default_unit and blanket_default_unit != duration_std.TREAT_AS_MISSING:
            # Same bug/fix as Modules 46 and 48: a column that's
            # ENTIRELY bare numbers with zero embedded duration-unit
            # tokens anywhere (e.g. an "Age" column that's just "42",
            # "35", ... flagged purely via the strong name-keyword
            # signal) gives _resolve_effective_target_unit nothing to
            # work with. The blanket default is the obvious target
            # here -- if every bare value defaults to the same unit,
            # that unit IS the column's unit.
            target_unit = blanket_default_unit

        if target_unit is None:
            continue

        unresolved_details = duration_std.find_unresolved_duration_details(df, col, info)
        value_to_resolution = {
            str(detail["raw_value"]): answers.get(f"duration_value_resolution__{col}__{i}")
            for i, detail in enumerate(unresolved_details)
        }

        # ADDED for the client's per-token redesign: maps each
        # unrecognised unit TOKEN (e.g. "m") to whatever the client
        # answered for it (a canonical unit, or TREAT_AS_MISSING) --
        # passed into every parse_duration_value() call below so a
        # value like "5 m 6 d" resolves correctly once "m" has been
        # clarified, without needing to re-teach the alias table
        # itself. Built fresh each run from Module 50's own token
        # enumeration, in the SAME order Module 11 used to generate
        # the questions, so the positional IDs line up.
        token_details = duration_std.find_unresolved_duration_tokens(df, col, info)
        token_overrides = {
            detail["token"]: answers.get(f"duration_token_resolution__{col}__{i}")
            for i, detail in enumerate(token_details)
        }
        token_overrides = {k: v for k, v in token_overrides.items() if v is not None}

        converted_values = []
        source_unit_counts = {}
        n_converted = 0
        n_already_target = 0
        n_unresolved = 0
        n_marked_missing = 0
        n_left_as_is = 0
        n_default_applied = 0
        unresolved_examples = []

        for raw in df[col]:
            number, canonical_unit, raw_token = duration_std.parse_duration_value(raw, token_overrides=token_overrides)

            if number is None:
                converted_values.append(np.nan)
                continue

            if raw_token is not None and duration_std.LEAVE_AS_IS in raw_token:
                # ADDED per explicit client request: at least one unit
                # token in this compound was explicitly marked "leave
                # as is" -- keeps the value EXACTLY as originally
                # entered (full text, e.g. "5 m 6 d" stays exactly
                # that), rather than attempting a partial conversion
                # or blanking it. Checked BEFORE the TREAT_AS_MISSING
                # check below since both sentinels could in principle
                # appear together across different tokens in the same
                # compound -- "leave as is" is the more conservative,
                # less destructive choice, so it wins if both are present.
                converted_values.append(raw)
                n_left_as_is += 1
                continue

            if raw_token is not None and duration_std.TREAT_AS_MISSING in raw_token:
                # At least one unit token in this value was explicitly
                # marked "treat as missing" by the client (Module 11's
                # per-token clarification question) -- a duration
                # missing even one component of a compound can't be
                # safely computed, so the WHOLE value becomes missing,
                # not partially converted.
                converted_values.append(np.nan)
                n_marked_missing += 1
                continue

            source_unit = canonical_unit or implied_unit
            is_bare_cell = source_unit is None and canonical_unit is None and raw_token is None

            if source_unit is None and is_bare_cell and blanket_default_unit:
                clarified_unit = blanket_default_unit
                if clarified_unit not in (duration_std.TREAT_AS_MISSING, duration_std.LEAVE_AS_IS):
                    n_default_applied += 1
            else:
                # Restricted to genuinely bare cells -- a value with
                # unresolved TOKEN(s) that still has no override yet
                # falls straight through to the "unresolved" branch
                # below instead, rather than incorrectly matching a
                # stale per-VALUE answer that no longer applies to it
                # (see Module 50's find_unresolved_duration_details,
                # which no longer covers token-issue values at all).
                clarified_unit = value_to_resolution.get(str(raw)) if source_unit is None and is_bare_cell else None

            if source_unit is None and clarified_unit == duration_std.LEAVE_AS_IS:
                converted_values.append(raw)
                n_left_as_is += 1
                continue

            if source_unit is None and clarified_unit == duration_std.TREAT_AS_MISSING:
                converted_values.append(np.nan)
                n_marked_missing += 1
                continue

            source_unit = source_unit or clarified_unit

            if source_unit is None:
                converted_values.append(number)
                n_unresolved += 1
                if len(unresolved_examples) < 5:
                    unresolved_examples.append(raw)
                continue

            converted = convert_duration_value(number, source_unit, target_unit)
            converted_values.append(converted)
            source_unit_counts[source_unit] = source_unit_counts.get(source_unit, 0) + 1
            if source_unit == target_unit:
                n_already_target += 1
            else:
                n_converted += 1

        df[col] = converted_values

        source_breakdown = ", ".join(f"{n} {u}" for u, n in sorted(source_unit_counts.items()))

        if implied_unit == target_unit:
            rename_note = "column name already documents this unit, no rename needed"
        else:
            standardized_units[col] = target_unit
            rename_note = f"column will be labelled '{col} ({target_unit})' in the final output"

        profile[col]["dtype"] = "numeric"
        profile[col]["semantic_role"] = "numeric_measure"
        profile[col]["role_confidence"] = 1.0
        profile[col]["variable_type"] = "continuous"
        profile[col]["variable_type_confidence"] = 0.9

        log_line = (
            f"Standardised '{col}' to {target_unit} ({rename_note}): {n_converted} value(s) "
            f"converted, {n_already_target} value(s) already in {target_unit} -- source units "
            f"were: {source_breakdown or 'none'}"
        )
        if n_default_applied:
            log_line += (
                f"; {n_default_applied} value(s) with no unit mentioned were assumed to be "
                f"{blanket_default_unit}, per client's confirmed default for '{col}'"
            )
        if n_marked_missing:
            log_line += (
                f"; {n_marked_missing} value(s) set to MISSING per client's 'I don't know' answer "
                f"to the unresolved-unit question"
            )
        if n_left_as_is:
            log_line += (
                f"; {n_left_as_is} value(s) left EXACTLY as originally entered, per client's "
                f"'I don't know' answer to the unresolved-unit question"
            )
        if n_unresolved:
            log_line += (
                f"; {n_unresolved} value(s) left UNCONVERTED -- original number kept as-is because "
                f"no unit could be determined (e.g. {unresolved_examples})"
            )
        actions_log.append(log_line)

    return df, actions_log, standardized_units


if __name__ == "__main__":
    df = pd.DataFrame({
        "Age": ["42", "35", "28", "61", "19", "37"],
        "Tenure": ["3 years", "5 years 2 months", "18 months", "2 yrs", "10 fortnights", "9 months"],
    })
    profile = {
        "Age": {"dtype": "text", "semantic_role": "categorical"},
        "Tenure": {"dtype": "text", "semantic_role": "categorical"},
    }

    print("=== BEFORE ===")
    print(df)

    answers = {
        "standardize_duration__Age": True,
        "has_default_duration_unit__Age": True,
        "default_duration_unit__Age": "year",
        "standardize_duration__Tenure": True,
        "target_duration_unit__Tenure": "year",
        "duration_token_resolution__Tenure__0": duration_std.TREAT_AS_MISSING,  # "fortnights" (in "10 fortnights")
    }
    converted_df, log, standardized = apply_duration_conversion(df, profile, answers)

    print("\n=== AFTER ===")
    print(converted_df)
    print("\n=== Actions log ===")
    for line in log:
        print(" -", line)

    final_df = apply_final_column_renames(converted_df, standardized)
    print("\n=== AFTER final rename ===")
    print(final_df)
    print("Final columns:", list(final_df.columns))
