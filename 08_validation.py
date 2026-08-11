"""
Module H: Validation
-----------------------
Checks data against known format/business rules: email format, phone
format, and numeric ranges. Same philosophy as every other module here:
flag only, never silently modify or drop a value.

IMPORTANT DISTINCTION from Module G (Outlier Detection):
  - Outlier detection asks "is this unusual COMPARED TO THE REST OF
    THIS DATASET?" (a statistical question).
  - Validation asks "does this violate a KNOWN, FIXED rule?" (e.g. age
    can never be negative or over ~120, regardless of what the rest of
    the data looks like).
  These can disagree -- a value can be statistically rare but still
  perfectly valid, or technically "normal-looking" but still invalid.

Also important: a MISSING value is not the same as an INVALID one. All
functions here return NaN (not True/False) for missing values, so the
final report can distinguish "wasn't provided" from "was provided but
doesn't meet the format/rule" from "valid".
"""

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Email validation
# ---------------------------------------------------------------------------

# A deliberately simple, practical pattern -- not a fully RFC-compliant
# email regex (those are notoriously complex and still imperfect). This
# catches the realistic problems we're likely to see: missing "@",
# missing domain, no dot in the domain part.
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_email_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Adds a "<column>_valid" column: True/False for present values, NaN
    for missing ones (a missing email isn't "invalid" -- that's a
    separate concern handled by Module M1).
    """
    df = df.copy()

    def check(value):
        if pd.isna(value):
            return np.nan
        return bool(EMAIL_PATTERN.match(str(value).strip()))

    df[f"{column}_valid"] = df[column].apply(check)
    return df


# ---------------------------------------------------------------------------
# 2. Phone validation
# ---------------------------------------------------------------------------


def validate_phone_column(
    df: pd.DataFrame, column: str, min_digits: int = 10, max_digits: int = 13
) -> pd.DataFrame:
    """
    Checks that a phone number has a plausible NUMBER OF DIGITS,
    regardless of formatting style (spaces, brackets, dashes are all
    fine -- we intentionally don't enforce one exact layout, since
    Module B/our own synthetic data generator legitimately produces
    several different valid-looking styles for the same real number).

    Defaults (10-13 digits) cover UK numbers in national format (11
    digits with a leading 0), international format with the country
    code replacing that leading 0 (12 digits), AND a quirk found by
    testing against our OWN synthetic data: Faker's en_GB generator
    sometimes writes "+44(0)..." -- combining the country code AND the
    redundant trunk zero together (13 digits), which isn't how a real
    international number would normally be written, but is harmless
    placeholder behaviour worth accommodating rather than treating as
    a validation failure. Adjust min/max per job for other regions.
    """
    df = df.copy()

    def check(value):
        if pd.isna(value):
            return np.nan
        digit_count = sum(ch.isdigit() for ch in str(value))
        return min_digits <= digit_count <= max_digits

    df[f"{column}_valid"] = df[column].apply(check)
    return df


# ---------------------------------------------------------------------------
# 3. Numeric range validation
# ---------------------------------------------------------------------------


def validate_numeric_range(
    df: pd.DataFrame, column: str, min_value=None, max_value=None
) -> pd.DataFrame:
    """
    Flags values outside a fixed, known-valid range (a HARD business
    rule, e.g. age must be 0-120) -- not a statistical judgement about
    what's "normal" for this particular dataset.

    Pass only min_value, only max_value, or both, depending on what
    makes sense for the column.
    """
    df = df.copy()

    def check(value):
        if pd.isna(value):
            return np.nan
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True

    df[f"{column}_valid"] = df[column].apply(check)
    return df


# ---------------------------------------------------------------------------
# 4. Generic pattern validation (extensibility)
# ---------------------------------------------------------------------------


def validate_pattern(df: pd.DataFrame, column: str, pattern: str) -> pd.DataFrame:
    """
    Validate a column against ANY regular expression -- e.g. a postcode
    format, a specific ID format, a product code format. Built
    generically so a new validation rule for a future job can be added
    by just supplying a new pattern, without writing a new function.
    """
    df = df.copy()
    compiled = re.compile(pattern)

    def check(value):
        if pd.isna(value):
            return np.nan
        return bool(compiled.match(str(value).strip()))

    df[f"{column}_valid"] = df[column].apply(check)
    return df


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")
    text_std = import_module("02_text_standardisation")

    # Colab-friendly hardcoded path, same as previous modules.
    filepath = "dirty_customer_data.xlsx"

    df = ingestion.load_data(filepath)
    df = text_std.trim_whitespace(df)
    df = text_std.standardise_casing(df, {"full_name": "title", "company": "title"})

    print("--- Email validation ---")
    df = validate_email_column(df, "email")
    print(df["email_valid"].value_counts(dropna=False))
    print(df.loc[df["email_valid"] == False, "email"].head(10))

    print("\n--- Phone validation ---")
    df = validate_phone_column(df, "phone")
    print(df["phone_valid"].value_counts(dropna=False))
    print(df.loc[df["phone_valid"] == False, "phone"].head(10))

    print("\n--- Age range validation (hard business rule: 0-120) ---")
    df = validate_numeric_range(df, "age", min_value=0, max_value=120)
    print(df["age_valid"].value_counts(dropna=False))
    print(df.loc[df["age_valid"] == False, "age"])

    print("\n--- Postcode format validation (example of generic pattern use) ---")
    # A reasonably standard UK postcode pattern -- illustrative use of
    # validate_pattern() for a format we didn't write a dedicated
    # function for.
    UK_POSTCODE_PATTERN = r"^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$"
    if "postcode" in df.columns:
        df = validate_pattern(df, "postcode", UK_POSTCODE_PATTERN)
        print(df["postcode_valid"].value_counts(dropna=False))
