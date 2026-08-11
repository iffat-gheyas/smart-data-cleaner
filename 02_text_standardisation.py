"""
Module B: Text Standardisation
--------------------------------
Fixes two common data-entry problems in text columns:
  1. Leading/trailing whitespace (e.g. "  John Smith  ")
  2. Inconsistent capitalisation (e.g. "JOHN SMITH", "john smith", "John Smith"
     all referring to the same kind of value)

Both functions work on a COPY of the DataFrame and return it -- they never
modify the original DataFrame in place. This matters once we chain modules
together in main.py: each step takes the previous step's output and hands
back a new result, so nothing gets silently mutated behind your back.
"""

import sys

import pandas as pd


def trim_whitespace(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from text columns.

    If `columns` is not given, we automatically apply this to every
    column that pandas has typed as "object" (pandas' generic label for
    text columns). This is a sensible default: trimming whitespace from
    a text column is essentially always safe and desired, so we don't
    need the user to list every text column by hand.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=["object", "string"]).columns

    for col in columns:
        # .str.strip() only works on actual strings. Any missing values
        # (NaN) in the column are left untouched -- NaN isn't a string,
        # so trying to strip it would raise an error if we weren't careful.
        # Using df[col].str.strip() (the pandas string accessor) handles
        # this correctly: it skips NaN automatically and only strips
        # actual string values.
        df[col] = df[col].str.strip()

    return df


def standardise_casing(df: pd.DataFrame, column_case_map: dict) -> pd.DataFrame:
    """
    Apply a chosen capitalisation style to specific columns.

    `column_case_map` is a dictionary like:
        {"full_name": "title", "company": "title"}

    We ask the CALLER to specify which columns get which case style,
    rather than guessing -- capitalisation preference genuinely depends
    on what the column represents (a person's name usually wants Title
    Case, but a product code might need to stay UPPER). Guessing here
    risks silently making the wrong choice, similar to why we don't
    auto-guess ambiguous dates elsewhere in this pipeline.
    """
    df = df.copy()

    case_functions = {
        "title": lambda s: s.str.title(),
        "upper": lambda s: s.str.upper(),
        "lower": lambda s: s.str.lower(),
    }

    for col, style in column_case_map.items():
        if style not in case_functions:
            raise ValueError(
                f"Unknown case style '{style}' for column '{col}'. "
                f"Choose from: {list(case_functions.keys())}"
            )
        df[col] = case_functions[style](df[col])

    return df


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")

    # Colab-friendly: hardcode the path rather than relying on sys.argv,
    # same adjustment as Module A.
    filepath = "dirty_customer_data.xlsx"

    df = ingestion.load_data(filepath)

    print("Before standardisation (full_name sample):")
    print(df["full_name"].head(10).tolist())

    df = trim_whitespace(df)
    df = standardise_casing(df, {"full_name": "title", "company": "title"})

    print("\nAfter standardisation (full_name sample):")
    print(df["full_name"].head(10).tolist())
