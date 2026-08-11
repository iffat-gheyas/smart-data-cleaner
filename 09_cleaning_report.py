"""
Module F: Cleaning Report
---------------------------
Produces a "before vs after" summary of everything the pipeline did --
the client-facing proof that real, meaningful cleaning happened.

Reuses Module A's diagnostic_summary() for both the "before" and
"after" snapshots, so the exact same analysis logic runs on both ends
and the comparison is directly apples-to-apples.

DESIGN: this module doesn't know HOW any of the cleaning happened -- it
just takes (1) a diagnostic summary from the START of the pipeline,
(2) a diagnostic summary from the END, and (3) a plain list of short
text log entries describing what each step did. Collecting that log is
main.py's job as it calls each module -- this keeps the report fully
decoupled from every other module.
"""

import pandas as pd


def build_comparison(before_summary: dict, after_summary: dict) -> dict:
    """
    Compare two diagnostic summaries (from Module A's diagnostic_summary())
    and compute the headline before/after numbers.

    Includes both a RAW missing-value count AND an AVERAGE missing-value
    PERCENTAGE across columns. The raw count alone can be misleading
    once column splitting has run (Module E) -- splitting one column
    into three multiplies how many cells COUNT a single missing value
    across, inflating the raw total even though nothing about data
    quality actually got worse. The percentage-based figure stays
    meaningful regardless of how many columns exist.
    """
    avg_missing_pct_before = sum(before_summary["missing_percent_per_column"].values()) / len(
        before_summary["missing_percent_per_column"]
    )
    avg_missing_pct_after = sum(after_summary["missing_percent_per_column"].values()) / len(
        after_summary["missing_percent_per_column"]
    )

    return {
        "rows_before": before_summary["n_rows"],
        "rows_after": after_summary["n_rows"],
        "rows_removed": before_summary["n_rows"] - after_summary["n_rows"],
        "columns_before": before_summary["n_columns"],
        "columns_after": after_summary["n_columns"],
        "duplicate_rows_before": before_summary["duplicate_rows_exact"],
        "duplicate_rows_after": after_summary["duplicate_rows_exact"],
        "total_missing_before": sum(before_summary["missing_per_column"].values()),
        "total_missing_after": sum(after_summary["missing_per_column"].values()),
        "avg_missing_percent_before": round(avg_missing_pct_before, 2),
        "avg_missing_percent_after": round(avg_missing_pct_after, 2),
    }


def print_cleaning_report(before_summary: dict, after_summary: dict, actions_log: list) -> dict:
    """
    Print a readable report to the console/notebook. Returns the
    comparison dict too, in case the caller wants to reuse the numbers
    (e.g. to display them in a Streamlit app later).
    """
    comparison = build_comparison(before_summary, after_summary)

    print("=" * 60)
    print("DATA CLEANING REPORT")
    print("=" * 60)

    print(
        f"\nRows:    {comparison['rows_before']} -> {comparison['rows_after']} "
        f"({comparison['rows_removed']} removed)"
    )
    print(f"Columns: {comparison['columns_before']} -> {comparison['columns_after']}")
    print(
        f"Duplicate rows: {comparison['duplicate_rows_before']} -> "
        f"{comparison['duplicate_rows_after']}"
    )
    print(
        f"Total missing values (raw count): {comparison['total_missing_before']} -> "
        f"{comparison['total_missing_after']} "
        f"(raw count rises after column splitting -- see % below for the real picture)"
    )
    print(
        f"Average missing % per column: {comparison['avg_missing_percent_before']}% -> "
        f"{comparison['avg_missing_percent_after']}%"
    )

    print("\nActions performed:")
    for entry in actions_log:
        print(f"  - {entry}")

    print("=" * 60)

    return comparison


def _split_off_outlier_flag_columns(df: pd.DataFrame) -> tuple:
    """
    Separates Module G's diagnostic outlier-flag columns (e.g.
    "salary_outlier_iqr", "multivariate_outlier") OUT of the actual
    cleaned data, returning (main_df, flags_df).

    These columns are genuinely useful QA information -- exactly which
    rows got flagged, in which column -- but they don't belong mixed
    into the "Cleaned Data" sheet itself. Added after real-user
    feedback: a client opening the "Cleaned Data" sheet shouldn't have
    to see raw True/False diagnostic columns sitting next to their
    actual data, especially now that outlier capping is opt-in (see
    Module K) -- a column can be flagged as an outlier WITHOUT ever
    having its value touched, so seeing "True" next to an unmodified
    value could easily read as a bug rather than the intended
    "detected, not changed" behaviour.

    Returns (df, None) if there are no such columns to split off, so
    the caller can skip creating an empty extra sheet.
    """
    flag_cols = [
        col for col in df.columns
        if col.endswith("_outlier_iqr") or col == "multivariate_outlier"
    ]
    if not flag_cols:
        return df, None

    flags_df = df[flag_cols].copy()
    main_df = df.drop(columns=flag_cols)
    return main_df, flags_df


def _build_outlier_flags_sheet(flags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns the raw True/False flag columns into a compact "just the
    flagged rows" sheet, with a "Cleaned Data Row" column that points
    directly at the matching row in the "Cleaned Data" sheet.

    Only rows with AT LEAST ONE flag set to True are kept -- for a
    large file, a full column of mostly-False values would be far
    less useful than a short list of exactly the rows worth a second
    look. +2 (not +1) on the row number accounts for the "Cleaned
    Data" sheet's own header row, so "Cleaned Data Row 34" means
    literally open that sheet and go to row 34.
    """
    flagged_rows = flags_df[flags_df.any(axis=1)].copy()
    flagged_rows.insert(0, "Cleaned Data Row", flagged_rows.index + 2)
    return flagged_rows.reset_index(drop=True)


def _split_off_validation_flag_columns(df: pd.DataFrame) -> tuple:
    """
    Separates Module H's validation flag columns (e.g. "Phone_valid",
    "Email_valid", "Customer Address: Postcode_valid") OUT of the
    actual cleaned data, returning (main_df, flags_df) -- mirrors
    _split_off_outlier_flag_columns() immediately above, same
    reasoning and same client-reported issue: a client opening the
    "Cleaned Data" sheet shouldn't have to see a raw True/False
    diagnostic column sitting next to their actual phone number or
    email address -- that's genuinely useful QA information, but it
    belongs in its own sheet, not mixed into the data itself.

    Matches by suffix ("_valid") rather than a fixed list of column
    names, since which columns exist depends on which of Phone/Email/
    Address the client's file actually has -- same general approach
    the outlier-flag split already uses ("_outlier_iqr").

    Returns (df, None) if there are no such columns, so the caller can
    skip creating an empty extra sheet.
    """
    flag_cols = [col for col in df.columns if col.endswith("_valid")]
    if not flag_cols:
        return df, None

    flags_df = df[flag_cols].copy()
    main_df = df.drop(columns=flag_cols)
    return main_df, flags_df


def _build_validation_flags_sheet(flags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns the raw True/False validity columns into a compact "just the
    problem rows" sheet, with the same "Cleaned Data Row" pointer
    convention as _build_outlier_flags_sheet().

    OPPOSITE POLARITY from the outlier-flags sheet, worth being
    explicit about: there, True means "flagged, worth a look" -- here,
    True means "valid, perfectly ordinary" and FALSE is the
    interesting case. So this keeps only rows where AT LEAST ONE
    validity column is False, not True.
    """
    flagged_rows = flags_df[(flags_df == False).any(axis=1)].copy()
    flagged_rows.insert(0, "Cleaned Data Row", flagged_rows.index + 2)
    return flagged_rows.reset_index(drop=True)


def export_cleaning_report_to_excel(
    before_summary: dict,
    after_summary: dict,
    actions_log: list,
    output_path: str,
    cleaned_df: pd.DataFrame = None,
) -> str:
    """
    Write the cleaning report as sheets in an Excel workbook.

    If `cleaned_df` is given, it's included as a "Cleaned Data" sheet
    in the SAME workbook -- so the client receives one file containing
    both their cleaned data and the proof of what was done to it,
    rather than two separate deliverables.
    """
    comparison = build_comparison(before_summary, after_summary)

    summary_df = pd.DataFrame(
        [
            {"Metric": "Rows (before)", "Value": comparison["rows_before"]},
            {"Metric": "Rows (after)", "Value": comparison["rows_after"]},
            {"Metric": "Rows removed", "Value": comparison["rows_removed"]},
            {"Metric": "Columns (before)", "Value": comparison["columns_before"]},
            {"Metric": "Columns (after)", "Value": comparison["columns_after"]},
            {
                "Metric": "Duplicate rows (before)",
                "Value": comparison["duplicate_rows_before"],
            },
            {
                "Metric": "Duplicate rows (after)",
                "Value": comparison["duplicate_rows_after"],
            },
            {
                "Metric": "Total missing values (raw count, before)",
                "Value": comparison["total_missing_before"],
            },
            {
                "Metric": "Total missing values (raw count, after)",
                "Value": comparison["total_missing_after"],
            },
            {
                "Metric": "Average missing % per column (before)",
                "Value": comparison["avg_missing_percent_before"],
            },
            {
                "Metric": "Average missing % per column (after)",
                "Value": comparison["avg_missing_percent_after"],
            },
        ]
    )

    actions_df = pd.DataFrame({"Action performed": actions_log})

    # engine="xlsxwriter", NOT "openpyxl" -- confirmed via direct XML
    # inspection that pandas 3.0.2's openpyxl writer produces an invalid
    # empty <c t="inlineStr"></c> cell (missing the <is> child the OOXML
    # schema requires) for ANY missing value in a text/object column,
    # which triggers Excel's "problem with content, needs repair" dialog
    # on open. Reproduces identically regardless of the NA sentinel used
    # (None / pd.NA / np.nan / even a literal empty string) -- it's a bug
    # in pandas' own openpyxl-writer layer for this version, not fixable
    # by changing what value represents "missing." xlsxwriter doesn't
    # have the same bug (verified with a direct XML comparison).
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        outlier_flags_df = None
        validation_flags_df = None
        if cleaned_df is not None:
            main_df, outlier_flags_df = _split_off_outlier_flag_columns(cleaned_df)
            # Chained on the ALREADY outlier-stripped main_df -- both
            # splits are independent (different column suffixes), so
            # order between them doesn't matter, but doing it this way
            # means "Cleaned Data" only ever needs one round of column
            # removal to reason about.
            main_df, validation_flags_df = _split_off_validation_flag_columns(main_df)
            main_df.to_excel(writer, sheet_name="Cleaned Data", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        actions_df.to_excel(writer, sheet_name="Actions Log", index=False)
        if outlier_flags_df is not None:
            _build_outlier_flags_sheet(outlier_flags_df).to_excel(
                writer, sheet_name="Outlier Flags", index=False
            )
        if validation_flags_df is not None:
            _build_validation_flags_sheet(validation_flags_df).to_excel(
                writer, sheet_name="Validation Flags", index=False
            )

    return output_path


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")
    text_std = import_module("02_text_standardisation")
    dedup = import_module("03_duplicate_removal")
    datetime_std = import_module("04_datetime_standardisation")
    splitting = import_module("05_column_splitting")
    missing_val = import_module("06_missing_value_single_imputation")
    outliers = import_module("07_outlier_detection")
    validation = import_module("08_validation")

    # Colab-friendly hardcoded path, same as previous modules.
    filepath = "dirty_customer_data.xlsx"

    actions_log = []

    # --- Load and snapshot BEFORE state ---
    df = ingestion.load_data(filepath)
    before_summary = ingestion.diagnostic_summary(df)

    # --- Text standardisation ---
    df = text_std.trim_whitespace(df)
    df = text_std.standardise_casing(df, {"full_name": "title", "company": "title"})
    actions_log.append("Trimmed whitespace and standardised text casing")

    # --- Duplicate removal (after text standardisation, per Module C's lesson) ---
    rows_before = len(df)
    df = dedup.remove_exact_duplicates(df)
    df = dedup.remove_subset_duplicates(df, subset=["email"])
    actions_log.append(f"Removed {rows_before - len(df)} duplicate rows (exact + email-based)")

    # --- Date/time standardisation ---
    df, fail_signup = datetime_std.standardise_date_column(df, "signup_date", dayfirst=True)
    df, fail_last_order = datetime_std.standardise_date_column(df, "last_order_date", dayfirst=True)
    df, fail_ts = datetime_std.standardise_date_column(
        df, "transaction_timestamp", dayfirst=True, output_format="%Y-%m-%d %H:%M:%S"
    )
    actions_log.append(
        f"Standardised date/time formats ({fail_signup + fail_last_order + fail_ts} unparseable values flagged)"
    )

    # --- Outlier detection + capping BEFORE missing-value imputation (Module G's lesson) ---
    df = outliers.detect_outliers_iqr(df, "age")
    df = outliers.detect_outliers_iqr(df, "order_amount")
    n_age_outliers = df["age_outlier_iqr"].sum()
    n_amount_outliers = df["order_amount_outlier_iqr"].sum()
    df = outliers.cap_outliers_iqr(df, "age")
    df = outliers.cap_outliers_iqr(df, "order_amount")
    actions_log.append(
        f"Detected and capped {n_age_outliers} age outliers and {n_amount_outliers} order_amount outliers"
    )

    # --- Missing value imputation (numeric columns are now outlier-free) ---
    n_age_missing = df["age"].isna().sum()
    n_amount_missing = df["order_amount"].isna().sum()
    df = missing_val.knn_impute_numeric(df, columns=["age", "order_amount"], k=5)
    df = missing_val.knn_impute_categorical(
        df, categorical_column="country", numeric_feature_columns=["age", "order_amount"], k=5
    )
    df = missing_val.simple_impute(df, "notes", strategy="constant", constant_value="No notes provided")
    actions_log.append(
        f"Filled {n_age_missing} missing age and {n_amount_missing} missing order_amount values via KNN; "
        f"filled missing country via categorical KNN; filled missing notes with a placeholder"
    )

    # --- Column splitting (after date standardisation, so datetime split is unambiguous) ---
    df = splitting.split_full_name(df)
    df = splitting.split_address_column(df)
    df = splitting.split_datetime_column(df, "signup_date", include_time=False)
    df = splitting.split_datetime_column(df, "last_order_date", include_time=False)
    df = splitting.split_datetime_column(df, "transaction_timestamp", include_time=True)
    actions_log.append("Split full_name, address, and datetime columns into separate fields")

    # --- Validation (postcode is only available now that splitting has run) ---
    df = validation.validate_email_column(df, "email")
    df = validation.validate_phone_column(df, "phone")
    n_invalid_email = (df["email_valid"] == False).sum()
    n_invalid_phone = (df["phone_valid"] == False).sum()
    actions_log.append(
        f"Flagged {n_invalid_email} invalid email(s) and {n_invalid_phone} invalid phone number(s)"
    )

    # --- Snapshot AFTER state and build the report ---
    after_summary = ingestion.diagnostic_summary(df)

    print_cleaning_report(before_summary, after_summary, actions_log)

    export_cleaning_report_to_excel(
        before_summary, after_summary, actions_log, "cleaned_output_with_report.xlsx", cleaned_df=df
    )
    print("\nSaved: cleaned_output_with_report.xlsx (Cleaned Data + Summary + Actions Log sheets)")
