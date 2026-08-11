"""
Module C: Duplicate Removal
----------------------------
Handles two kinds of duplicates:
  1. Exact duplicates -- every column matches, row-for-row identical.
  2. Subset duplicates -- only SOME columns match (e.g. same email, or
     same name + company), which is how we catch "near-duplicate"
     customers who were entered twice with small differences elsewhere
     (e.g. different casing, or a slightly different note field).

IMPORTANT PIPELINE ORDER NOTE:
This module should run AFTER Module B (text standardisation). A
near-duplicate like "cody willis" vs "CODY WILLIS" only becomes
detectable as an exact match once both have been standardised to the
same casing. If you run duplicate removal BEFORE text standardisation,
you'll miss these near-duplicates entirely.
"""

import sys

import pandas as pd


def get_duplicate_rows(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Return ONLY the rows involved in a duplicate (both/all copies),
    without removing anything. This is deliberately a separate,
    non-destructive function -- useful for a human to review exactly
    which rows will be affected before committing to removal, which
    fits the "manual quality check" step of the workflow.

    If `subset` is None, checks for exact duplicates across every
    column. If `subset` is a list of column names, checks duplicates
    based on just those columns (e.g. subset=["email"]).

    IMPORTANT: two rows are NEVER treated as duplicates just because
    they both happen to have a MISSING value in the subset column(s).
    By default, pandas treats NaN as equal to NaN -- which would mean
    two completely different customers who both left "email" blank
    get flagged as "duplicates" of each other. That's wrong, so we
    explicitly exclude rows with missing subset values from matching.
    """
    mask = df.duplicated(subset=subset, keep=False)

    if subset is not None:
        non_null_mask = df[subset].notna().all(axis=1)
        mask = mask & non_null_mask

    return df[mask]


def remove_exact_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Remove rows that are identical across EVERY column.

    `keep` controls which copy survives when duplicates are found:
      - "first": keep the first occurrence (default)
      - "last":  keep the last occurrence
      - False:   remove ALL copies, including the first (rarely what
                 you want -- included for completeness)
    """
    df = df.copy()
    return df.drop_duplicates(keep=keep).reset_index(drop=True)


def remove_subset_duplicates(
    df: pd.DataFrame, subset: list, keep: str = "first"
) -> pd.DataFrame:
    """
    Remove rows that match on a SUBSET of columns, even if other
    columns differ. This is what catches near-duplicate customers --
    e.g. same email address, but a slightly different name spelling
    or a different note field.

    `subset` is a list of column names to match on, e.g. ["email"] or
    ["full_name", "company"].

    Same safeguard as get_duplicate_rows(): rows with a MISSING value
    in the subset column(s) are never removed as "duplicates" of each
    other just because they share a blank. We handle this by splitting
    the data into "subset values all present" (where normal duplicate
    removal applies) and "subset value(s) missing" (left untouched),
    then combining the two back together.
    """
    df = df.copy()

    has_all_subset_values = df[subset].notna().all(axis=1)
    complete_rows = df[has_all_subset_values]
    incomplete_rows = df[~has_all_subset_values]

    deduped_complete = complete_rows.drop_duplicates(subset=subset, keep=keep)

    result = pd.concat([deduped_complete, incomplete_rows])
    return result.sort_index().reset_index(drop=True)


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")
    text_std = import_module("02_text_standardisation")

    # Colab-friendly hardcoded path, same as previous modules.
    filepath = "dirty_customer_data.xlsx"

    df = ingestion.load_data(filepath)
    print(f"Starting rows: {len(df)}")

    # Run text standardisation FIRST -- see the pipeline order note above.
    df = text_std.trim_whitespace(df)
    df = text_std.standardise_casing(df, {"full_name": "title", "company": "title"})

    # --- Exact duplicates ---
    exact_dupes = get_duplicate_rows(df)
    print(f"\nExact duplicate rows found (both copies counted): {len(exact_dupes)}")

    df_no_exact = remove_exact_duplicates(df)
    print(f"Rows after removing exact duplicates: {len(df_no_exact)}")

    # --- Subset duplicates: same email address ---
    email_dupes = get_duplicate_rows(df_no_exact, subset=["email"])
    print(f"\nRows sharing the same email (both copies counted): {len(email_dupes)}")
    print(email_dupes[["full_name", "email", "company"]].head(10))

    df_final = remove_subset_duplicates(df_no_exact, subset=["email"])
    print(f"\nRows after removing email-based subset duplicates: {len(df_final)}")

    # --- Small illustrative example ---
    # Our synthetic dataset's near-duplicates happened to become FULL exact
    # duplicates once text was standardised (Module B), so the email-subset
    # check above had nothing extra to catch. To prove subset-matching
    # genuinely works on PARTIAL matches (the real-world case it's built
    # for), here's a tiny toy example where only the email matches but
    # other fields genuinely differ:
    print("\n--- Illustrative example: partial-match duplicates ---")
    toy = pd.DataFrame(
        {
            "full_name": ["Jane Doe", "J. Doe", "Amit Shah"],
            "email": ["jane@example.com", "jane@example.com", "amit@example.com"],
            "notes": ["Called Monday", "Follow up needed", "New lead"],
        }
    )
    print("Before:")
    print(toy)
    print("\nAfter removing subset duplicates on ['email']:")
    print(remove_subset_duplicates(toy, subset=["email"]))
