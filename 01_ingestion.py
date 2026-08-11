"""
Module A: Ingestion & Diagnostic Summary
-----------------------------------------
Reads a CSV or Excel file into a pandas DataFrame and produces a plain
diagnostic summary of its current state (shape, column types, missing
data, duplicates) BEFORE any cleaning happens.

This summary becomes the "before" half of the final Cleaning Report
(Module F), so we deliberately return it as a dictionary of values,
not just printed text -- that way it can be reused later without
re-analysing the data.
"""

import sys

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Read a .csv, .xls, .xlsx file, OR a Google Sheets URL into a DataFrame.

    We pick the reader based on what `filepath` looks like: a Google
    Sheets URL is routed to load_google_sheet(); everything else is
    handled by pandas' normal file readers, based on the file extension.
    This means main.py's INPUT_FILE setting can be a local file OR a
    Google Sheets link, with no other code needing to change.
    """
    if filepath.startswith("https://docs.google.com/spreadsheets"):
        return load_google_sheet(filepath)

    lower = filepath.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif lower.endswith((".xls", ".xlsx")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported file type: {filepath}. "
            "Expected .csv, .xls, .xlsx, or a Google Sheets URL"
        )
    return df


def load_google_sheet(sheet_url_or_key: str, worksheet_name: str = None) -> pd.DataFrame:
    """
    Read a Google Sheet into a DataFrame.

    IMPORTANT: this uses Google Colab's built-in authentication, tied to
    YOUR OWN Google account. This means:
      - It only works when run inside an actual Google Colab notebook
        (the `from google.colab import auth` import will fail anywhere
        else, including a plain local Python script).
      - The target sheet must already be accessible to your Google
        account -- either you own it, or the client has shared it with
        you (e.g. via "Share" -> your email, Viewer or Editor access).
      - This is a DEVELOPMENT/TESTING setup, not what the future hosted
        client-facing tool will use. A hosted app needs a Google Cloud
        service account (its own credentials, with the sheet shared to
        the service account's email) instead of relying on an
        interactive personal login -- that's separate work for later,
        when we build the Streamlit version.

    `sheet_url_or_key` accepts either the full Google Sheets URL, or
    just the sheet's key (the long ID string in the URL).
    """
    from google.colab import auth

    auth.authenticate_user()

    import gspread
    from google.auth import default

    creds, _ = default()
    gc = gspread.authorize(creds)

    if sheet_url_or_key.startswith("http"):
        sh = gc.open_by_url(sheet_url_or_key)
    else:
        sh = gc.open_by_key(sheet_url_or_key)

    worksheet = sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1
    records = worksheet.get_all_records()
    return pd.DataFrame(records)


def save_to_google_sheet(
    df: pd.DataFrame, sheet_url_or_key: str, worksheet_name: str = None
) -> None:
    """
    Write a DataFrame back into a Google Sheet, replacing its current
    contents. Same Colab-account authentication caveat as
    load_google_sheet() above applies here too.

    Requires the gspread_dataframe package (pip install gspread_dataframe
    -- not preinstalled in Colab by default, unlike gspread itself).
    """
    from google.colab import auth

    auth.authenticate_user()

    import gspread
    from google.auth import default
    from gspread_dataframe import set_with_dataframe

    creds, _ = default()
    gc = gspread.authorize(creds)

    if sheet_url_or_key.startswith("http"):
        sh = gc.open_by_url(sheet_url_or_key)
    else:
        sh = gc.open_by_key(sheet_url_or_key)

    worksheet = sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1
    worksheet.clear()
    set_with_dataframe(worksheet, df)


def diagnostic_summary(df: pd.DataFrame) -> dict:
    """
    Analyse the DataFrame's current state and return the findings as
    a dictionary. Kept separate from load_data() because you'll often
    want to run diagnostics again mid-pipeline (e.g. "before" AND
    "after" cleaning) without re-reading the file from disk each time.
    """
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "column_dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isna().sum().to_dict(),
        "missing_percent_per_column": (df.isna().mean() * 100).round(2).to_dict(),
        "fully_empty_rows": int(df.isna().all(axis=1).sum()),
        "fully_empty_columns": [col for col in df.columns if df[col].isna().all()],
        "duplicate_rows_exact": int(df.duplicated(keep=False).sum()),
    }


def print_summary(summary: dict) -> None:
    """
    Display a diagnostic summary in a readable way.

    This is intentionally a separate function from diagnostic_summary()
    -- the SUMMARY DATA and the way we DISPLAY it are two different
    concerns. Later, when we build the final report (Module F), we'll
    reuse the same dictionary but format it differently (e.g. as a
    section in an Excel report), without needing to re-run any analysis.
    """
    print(f"Rows: {summary['n_rows']}")
    print(f"Columns: {summary['n_columns']}")
    print(f"Fully empty rows: {summary['fully_empty_rows']}")
    print(f"Fully empty columns: {summary['fully_empty_columns']}")
    print(f"Exact duplicate rows (both copies counted): {summary['duplicate_rows_exact']}")
    print("\nMissing values per column:")
    for col, pct in summary["missing_percent_per_column"].items():
        count = summary["missing_per_column"][col]
        print(f"  {col}: {count} missing ({pct}%)")


if __name__ == "__main__":
    # Lets you test this module completely on its own, e.g.:
    #   python 01_ingestion.py dirty_customer_data.xlsx
    # If no filename is given, it defaults to our synthetic test file.
    filepath = sys.argv[1] if len(sys.argv) > 1 else "dirty_customer_data.xlsx"
    df = load_data(filepath)
    summary = diagnostic_summary(df)
    print_summary(summary)
