"""
Module: Manual Value Corrections
-----------------------------------
Lets the client correct a SPECIFIC value they've spotted while
browsing the "Explore unique values per column" section of the app
(e.g. a typo like "Scotlant" that should read "Scotland") -- applies a
direct, EXACT-value replacement across every cell in that column
matching it precisely, never a substring or fuzzy match, so a
correction can never accidentally touch a different value that merely
looks similar.

This is entirely OPT-IN and client-driven: nothing here ever guesses
or fuzzy-matches a correction on its own -- every replacement comes
from an explicit "replace X with Y" instruction the client typed
themselves while looking directly at that value.
"""

import pandas as pd


def apply_value_corrections(df: pd.DataFrame, value_corrections: dict) -> tuple:
    """
    `value_corrections`: {column_name: {original_value: corrected_value}}
    -- exactly as the client typed it in the "Explore unique values"
    section (app.py builds this dict; nothing here needs to know how).

    Only ever touches a column if it still exists in `df` at this
    point in the pipeline -- fails safe (silently skips) rather than
    raising, since this runs very early, before any column renames
    from the other standardisation features, so every column should
    still be under its original name here regardless.

    Matches an EXACT value only (df[col] == original_value) -- never
    case-insensitive, never a substring match. If the client also
    wants a differently-cased variant fixed, that's a distinct value
    with its own distinct correction, exactly as it appeared in the
    "Explore unique values" dropdown they picked it from.

    Returns (df, actions_log).
    """
    df = df.copy()
    actions_log = []

    if not value_corrections:
        return df, actions_log

    for col, corrections in value_corrections.items():
        if col not in df.columns or not corrections:
            continue

        for original_value, corrected_value in corrections.items():
            if not corrected_value or corrected_value == original_value:
                continue  # nothing typed, or typed back the same thing -- no-op

            mask = df[col] == original_value
            n_matches = int(mask.sum())
            if n_matches == 0:
                continue

            df.loc[mask, col] = corrected_value
            actions_log.append(
                f"Replaced {n_matches} occurrence(s) of {original_value!r} with "
                f"{corrected_value!r} in '{col}', per client's manual correction"
            )

    return df, actions_log


if __name__ == "__main__":
    df = pd.DataFrame({
        "Country": ["Scotlant", "England", "Scotlant", "Wales", "USA"],
        "City": ["Londres", "Manchester", "Londres", "Cardiff", "Chicago"],
    })
    print("=== BEFORE ===")
    print(df)

    value_corrections = {
        "Country": {"Scotlant": "Scotland"},
        "City": {"Londres": "London", "Chicago": ""},  # blank correction -- should be a no-op
    }
    out, log = apply_value_corrections(df, value_corrections)

    print("\n=== AFTER ===")
    print(out)
    print("\n=== Actions log ===")
    for line in log:
        print(" -", line)
