"""
Module: Email Standardisation -- Apply Conversion + Batch Review
----------------------------------------------------------------------
Mirrors Module 54 (phone) exactly -- same two-function split
(find_unresolved_email_values for the batch review table, apply_email_
standardization for the real pipeline step), same "Don't know ->
missing" rule, same "only the SAFE category is ever auto-changed"
principle. See Module 54's docstring for the fuller reasoning.
"""

from importlib import import_module

import numpy as np
import pandas as pd

email_std = import_module("53_email_standardisation")

REVIEW_ACTION_CORRECT = "correct"
REVIEW_ACTION_KEEP = "keep"
REVIEW_ACTION_MISSING = "missing"
REVIEW_ACTION_DONT_KNOW = "dont_know"


def find_unresolved_email_values(df: pd.DataFrame, profile: dict, answers: dict) -> list:
    """
    Read-only scan for problem email values -- mirrors Module 54's
    find_unresolved_phone_values() exactly. Returns a list of dicts:
    column, row_index, original_value, problem. (No "expected_format"
    field here -- unlike phone numbers, there's no per-region example
    to show; the "reason" string from email_validator is already
    specific enough, e.g. "must have an @-sign".)
    """
    review_items = []
    detected = email_std.detect_email_columns(df)

    for col, info in detected.items():
        if col not in df.columns:
            continue
        if answers.get(f"confirm_email__{col}") is False:
            continue
        if answers.get(f"standardize_email__{col}") is False:
            continue

        casing_choice = answers.get(f"email_casing__{col}", "domain_only")
        lowercase_local = (casing_choice == "full_address")

        for idx, raw in df[col].items():
            if pd.isna(raw):
                continue
            if f"email_review__{col}__{idx}" in answers:
                continue
            parsed = email_std.parse_email_value(raw, lowercase_local_part=lowercase_local)
            if parsed["is_valid"]:
                continue
            review_items.append({
                "column": col,
                "row_index": idx,
                "original_value": raw,
                "problem": parsed["reason"],
            })

    return review_items


def apply_email_standardization(df: pd.DataFrame, profile: dict, answers: dict) -> tuple:
    """
    The real apply step, mirroring Module 54's apply_phone_
    standardization() exactly -- see that module's docstring for the
    fuller reasoning behind each branch.

    Returns (df, actions_log).
    """
    df = df.copy()
    actions_log = []

    detected = email_std.detect_email_columns(df)

    for col, info in detected.items():
        if col not in df.columns:
            continue

        confirm_answer = answers.get(f"confirm_email__{col}")
        if confirm_answer is False:
            continue

        standardize_answer = answers.get(f"standardize_email__{col}")
        if standardize_answer is False:
            actions_log.append(
                f"Left '{col}' as original text, per client's choice not to standardise emails"
            )
            continue

        casing_choice = answers.get(f"email_casing__{col}", "domain_only")
        lowercase_local = (casing_choice == "full_address")

        converted_values = []
        n_reformatted = 0
        n_already_standard = 0
        n_user_corrected = 0
        n_kept_original = 0
        n_marked_missing = 0
        n_unresolved = 0

        for idx, raw in df[col].items():
            if pd.isna(raw):
                converted_values.append(np.nan)
                continue

            parsed = email_std.parse_email_value(raw, lowercase_local_part=lowercase_local)

            if parsed["is_valid"]:
                normalized = parsed["normalized"]
                converted_values.append(normalized)
                if str(raw).strip() == normalized:
                    n_already_standard += 1
                else:
                    n_reformatted += 1
                continue

            review = answers.get(f"email_review__{col}__{idx}")
            if review is None:
                converted_values.append(raw)
                n_unresolved += 1
                continue

            action = review.get("action")
            if action == REVIEW_ACTION_CORRECT:
                corrected_raw = review.get("corrected_value")
                reparsed = email_std.parse_email_value(corrected_raw, lowercase_local_part=lowercase_local)
                if reparsed["is_valid"]:
                    converted_values.append(reparsed["normalized"])
                    n_user_corrected += 1
                else:
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
            f"Standardised '{col}' emails: {n_reformatted} value(s) reformatted, "
            f"{n_already_standard} value(s) already standard"
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
        "Email": [
            "JOHN.SMITH @ GMAIL.COM", "jane.doe@yahoo.com", "sara.khan@outlook.com",
            "tom.hill@company.co.uk", "john.gmail.com", "john smith@gmail.com",
        ],
    })
    profile = {"Email": {"dtype": "text", "semantic_role": "categorical"}}

    print("=== BEFORE ===")
    print(df)

    answers = {"standardize_email__Email": True, "email_casing__Email": "domain_only"}

    print("\n=== find_unresolved_email_values() ===")
    review_items = find_unresolved_email_values(df, profile, answers)
    for item in review_items:
        print(item)

    # "john.gmail.com" -> Don't know -> missing
    # "john smith@gmail.com" -> corrected
    answers["email_review__Email__4"] = {"action": "dont_know"}
    answers["email_review__Email__5"] = {"action": "correct", "corrected_value": "john.smith@gmail.com"}

    converted_df, log = apply_email_standardization(df, profile, answers)
    print("\n=== AFTER ===")
    print(converted_df)
    print("\n=== Actions log ===")
    for line in log:
        print(" -", line)
