"""
Module: Phone Standardisation -- Apply Conversion + Batch Review
----------------------------------------------------------------------
Sibling to Module 46/48/51's apply modules, but with a genuinely
different shape, following the client's own spec (Section 5): instead
of asking one dropdown question per unresolved value (the pattern
Modules 46/48/50 use), invalid/ambiguous phone values are collected
into ONE BATCH REVIEW TABLE the client fills in, since a per-value
question would be far too disruptive on a real file with many bad
numbers.

This means the apply step is split into two functions, used at two
different points in the app's flow:
  - find_unresolved_phone_values(): a read-only, side-effect-free scan
    that returns the list of problem values needing review -- called
    BEFORE the main pipeline runs, so app.py can render the batch
    review table and collect decisions.
  - apply_phone_standardization(): the real apply step, run as part of
    the main pipeline (Module 12) -- does the SAFE automatic
    formatting, then applies whatever review decisions exist in
    `answers` (keyed f"phone_review__{col}__{row_index}"), and honours
    the spec's explicit rule: "Don't know" -> treated as permission to
    make the value missing.
"""

from importlib import import_module

import numpy as np
import pandas as pd
import phonenumbers

phone_std = import_module("52_phone_standardisation")

# The four decision options from the client's own spec, plus the
# explicit rule: "Don't know" is NOT a fifth do-nothing option -- it's
# treated exactly the same as "Mark as missing".
REVIEW_ACTION_CORRECT = "correct"
REVIEW_ACTION_KEEP = "keep"
REVIEW_ACTION_MISSING = "missing"
REVIEW_ACTION_DONT_KNOW = "dont_know"


def _expected_format_hint(default_region: str, target_format: str) -> str:
    """
    Builds an illustrative "this is what a valid number looks like"
    example for the review table's "Expected format" column, using
    phonenumbers' own example-number generator for the region -- never
    a fabricated template, always a genuine example the library itself
    considers valid for that country.
    """
    if not default_region:
        return "(no default region set)"
    try:
        example = phonenumbers.example_number(default_region)
        if example is None:
            return "(no example available for this region)"
        fmt = phone_std.FORMAT_NAME_TO_ENUM.get(target_format, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        return phonenumbers.format_number(example, fmt)
    except Exception:
        return "(no example available for this region)"


def find_unresolved_phone_values(df: pd.DataFrame, profile: dict, answers: dict) -> list:
    """
    Read-only scan for problem values across every confirmed AND
    standardize=True phone column -- does NOT modify df, safe to call
    on every Streamlit rerun before the main pipeline executes.

    Skips any (column, row) that already has a stored review decision
    (f"phone_review__{col}__{idx}" present in answers) -- once the
    client has decided, it's resolved, not re-shown.

    Returns a list of dicts: column, row_index, original_value,
    problem, expected_format -- exactly the columns the client's spec
    asks the batch review table to show.
    """
    review_items = []
    detected = phone_std.detect_phone_columns(df)

    for col, info in detected.items():
        if col not in df.columns:
            continue
        if answers.get(f"confirm_phone__{col}") is False:
            continue
        if answers.get(f"standardize_phone__{col}") is False:
            continue

        default_region = answers.get(f"default_phone_region__{col}") or info.get("guessed_region")
        target_format = answers.get(f"target_phone_format__{col}", "INTERNATIONAL")
        expected_hint = _expected_format_hint(default_region, target_format)

        for idx, raw in df[col].items():
            if pd.isna(raw):
                continue
            if f"phone_review__{col}__{idx}" in answers:
                continue  # already resolved
            parsed = phone_std.parse_phone_value(raw, default_region)
            if parsed["is_valid"]:
                continue
            review_items.append({
                "column": col,
                "row_index": idx,
                "original_value": raw,
                "problem": parsed["reason"],
                "expected_format": expected_hint,
            })

    return review_items


def apply_phone_standardization(df: pd.DataFrame, profile: dict, answers: dict) -> tuple:
    """
    The real apply step -- part of the main Module 12 pipeline.

    For each confirmed, standardize=True phone column:
      - SAFE values (phonenumbers considers them genuinely valid) are
        reformatted into the client's chosen target format
        automatically -- no confirmation needed, matching the spec's
        "Confidence Categories" (Section 6): only this category is
        changed automatically.
      - Values with a stored review decision (from the batch review
        table) are resolved per that decision: "correct" re-parses
        the client's typed replacement (falls back to leaving it
        as-is, still flagged, if the replacement is ITSELF invalid --
        never silently accepted just because the client typed
        something); "keep" leaves the original text untouched; "missing"
        or "dont_know" become NaN (the spec's explicit rule).
      - Anything with NEITHER a valid parse NOR a stored decision yet
        is left completely untouched (the client hasn't reached the
        review table for it yet, or standardisation is still pending)
        -- never guessed at.

    Phone numbers keep their column NAME unchanged (unlike weight/
    currency/duration) -- there's no unit ambiguity being resolved by
    a header rename here, just a formatting choice.

    Returns (df, actions_log).
    """
    df = df.copy()
    actions_log = []

    detected = phone_std.detect_phone_columns(df)

    for col, info in detected.items():
        if col not in df.columns:
            continue

        confirm_answer = answers.get(f"confirm_phone__{col}")
        if confirm_answer is False:
            continue

        standardize_answer = answers.get(f"standardize_phone__{col}")
        if standardize_answer is False:
            actions_log.append(
                f"Left '{col}' as original text, per client's choice not to standardise phone formatting"
            )
            continue

        target_format = answers.get(f"target_phone_format__{col}", "INTERNATIONAL")
        default_region = answers.get(f"default_phone_region__{col}") or info.get("guessed_region")

        converted_values = []
        n_reformatted = 0
        n_already_target_format = 0
        n_user_corrected = 0
        n_kept_original = 0
        n_marked_missing = 0
        n_unresolved = 0

        for idx, raw in df[col].items():
            if pd.isna(raw):
                converted_values.append(np.nan)
                continue

            parsed = phone_std.parse_phone_value(raw, default_region)

            if parsed["is_valid"]:
                formatted = parsed.get(target_format.lower()) or parsed["international"]
                converted_values.append(formatted)
                if str(raw).strip() == formatted:
                    n_already_target_format += 1
                else:
                    n_reformatted += 1
                continue

            review = answers.get(f"phone_review__{col}__{idx}")
            if review is None:
                # No stored decision yet -- leave completely untouched
                # rather than guess (the batch review table hasn't
                # been completed for this value).
                converted_values.append(raw)
                n_unresolved += 1
                continue

            action = review.get("action")
            if action == REVIEW_ACTION_CORRECT:
                corrected_raw = review.get("corrected_value")
                reparsed = phone_std.parse_phone_value(corrected_raw, default_region)
                if reparsed["is_valid"]:
                    formatted = reparsed.get(target_format.lower()) or reparsed["international"]
                    converted_values.append(formatted)
                    n_user_corrected += 1
                else:
                    # The client's own correction still isn't valid --
                    # keep it as typed rather than silently reject it,
                    # but still count it as unresolved for the report.
                    converted_values.append(corrected_raw)
                    n_unresolved += 1
            elif action == REVIEW_ACTION_KEEP:
                converted_values.append(raw)
                n_kept_original += 1
            elif action in (REVIEW_ACTION_MISSING, REVIEW_ACTION_DONT_KNOW):
                converted_values.append(np.nan)
                n_marked_missing += 1
            else:
                converted_values.append(raw)
                n_unresolved += 1

        df[col] = converted_values

        log_line = (
            f"Standardised '{col}' phone numbers to {target_format}: {n_reformatted} value(s) reformatted, "
            f"{n_already_target_format} value(s) already in that format"
        )
        if n_user_corrected:
            log_line += f", {n_user_corrected} value(s) corrected by client via review"
        if n_kept_original:
            log_line += f", {n_kept_original} value(s) kept as original per client's review decision"
        if n_marked_missing:
            log_line += f", {n_marked_missing} value(s) marked missing per client's review decision"
        if n_unresolved:
            log_line += f", {n_unresolved} value(s) still unresolved (pending batch review)"
        actions_log.append(log_line)

    return df, actions_log


if __name__ == "__main__":
    df = pd.DataFrame({
        "Phone": ["+44 20 7946 0958", "020 7946 0958", "44 055542231", "4455ABC231", "07911 123456", None],
    })
    profile = {"Phone": {"dtype": "text", "semantic_role": "categorical"}}

    print("=== BEFORE ===")
    print(df)

    answers = {
        "standardize_phone__Phone": True,
        "target_phone_format__Phone": "INTERNATIONAL",
        "default_phone_region__Phone": "GB",
    }

    print("\n=== find_unresolved_phone_values() (before any review decisions) ===")
    review_items = find_unresolved_phone_values(df, profile, answers)
    for item in review_items:
        print(item)

    # Simulate the client's decisions from the batch review table:
    # - "44 055542231" (the client's own spec example) -> Don't know -> missing
    # - "4455ABC231" -> Enter corrected value -> a real GB number
    answers["phone_review__Phone__2"] = {"action": "dont_know"}
    answers["phone_review__Phone__3"] = {"action": "correct", "corrected_value": "020 7946 0959"}

    converted_df, log = apply_phone_standardization(df, profile, answers)
    print("\n=== AFTER ===")
    print(converted_df)
    print("\n=== Actions log ===")
    for line in log:
        print(" -", line)
