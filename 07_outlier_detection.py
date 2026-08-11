"""
Module G: Outlier Detection
------------------------------
Two different detection methods, because they catch different problems:

  1. IQR (univariate) -- flags a value that's extreme for ITS OWN column
     alone (e.g. age = 250). Simple, interpretable, easy to explain to
     a non-technical client.

  2. Isolation Forest (multivariate) -- flags a ROW that's unusual as a
     whole combination of values, even if no single column looks
     extreme by itself. More powerful, harder to explain in plain
     English.

PHILOSOPHY (same as the rest of this pipeline): detection NEVER
automatically deletes or changes data. Outliers are sometimes genuine,
real data -- a human should decide what to do with a flagged value, not
have it silently removed. Handling functions (capping / removing) are
provided separately below, and are always an explicit, opt-in step.

DEPENDENCY: Isolation Forest cannot run on columns with missing values.
Run Module M1 (missing value handling) on the relevant numeric columns
BEFORE using detect_outliers_isolation_forest().
"""

import pandas as pd
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# 1. Detection -- IQR (univariate)
# ---------------------------------------------------------------------------


def detect_outliers_iqr(
    df: pd.DataFrame, column: str, multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Flag values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].

    Adds a new boolean column "<column>_outlier_iqr" -- the original
    column is never modified. multiplier=1.5 is the standard convention
    (roughly equivalent to what a boxplot draws its whiskers at); a
    higher multiplier flags fewer, more extreme values.
    """
    df = df.copy()

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    df[f"{column}_outlier_iqr"] = (df[column] < lower_bound) | (
        df[column] > upper_bound
    )

    return df


# ---------------------------------------------------------------------------
# 2. Detection -- Isolation Forest (multivariate)
# ---------------------------------------------------------------------------


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: list,
    contamination="auto",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Flag rows that look unusual as a COMBINATION of the given numeric
    columns, using Isolation Forest.

    `contamination` is the expected proportion of outliers in the data
    (e.g. 0.05 for "roughly 5% of rows are unusual"). "auto" lets
    scikit-learn estimate this itself -- a reasonable default when you
    don't have a specific expectation, similar in spirit to choosing a
    k for KNN: a starting point you can tune per job.

    Adds a new boolean column "multivariate_outlier". Requires the
    given columns to have NO missing values -- run Module M1's missing
    value handling on them first.
    """
    df = df.copy()

    feature_matrix = df[columns]
    if feature_matrix.isna().any().any():
        raise ValueError(
            "columns must have no missing values before running Isolation "
            "Forest. Run missing value imputation (Module M1) on them first."
        )

    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(feature_matrix)  # -1 = outlier, 1 = normal

    df["multivariate_outlier"] = predictions == -1
    return df


# ---------------------------------------------------------------------------
# 3. Handling (opt-in, explicit, never automatic)
# ---------------------------------------------------------------------------


def cap_outliers_iqr(
    df: pd.DataFrame, column: str, multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Winsorize: cap values at the IQR bounds instead of removing rows.

    A value of -999 becomes the lower bound; a value of 50000 becomes
    the upper bound. This keeps the row (and all its other columns)
    rather than discarding it outright -- often preferable when you
    suspect the value is a genuine data-entry error but the rest of the
    row is still useful.
    """
    df = df.copy()

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def remove_flagged_rows(df: pd.DataFrame, flag_column: str) -> pd.DataFrame:
    """
    Remove rows where `flag_column` is True.

    Deliberately requires the caller to name a specific flag column
    (e.g. "age_outlier_iqr" or "multivariate_outlier") rather than
    having any implicit "remove all outliers" behaviour -- removal is
    always a conscious, explicit choice per column/method.
    """
    df = df.copy()
    return df[~df[flag_column]].reset_index(drop=True)


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")
    text_std = import_module("02_text_standardisation")
    missing_val = import_module("06_missing_value_single_imputation")

    # Colab-friendly hardcoded path, same as previous modules.
    filepath = "dirty_customer_data.xlsx"

    df = ingestion.load_data(filepath)
    df = text_std.trim_whitespace(df)
    df = text_std.standardise_casing(df, {"full_name": "title", "company": "title"})

    # Isolation Forest needs complete numeric columns -- fill gaps first.
    df = missing_val.knn_impute_numeric(df, columns=["age", "order_amount"], k=5)

    print("--- IQR outlier detection ---")
    df = detect_outliers_iqr(df, "age")
    df = detect_outliers_iqr(df, "order_amount")
    print(f"age outliers flagged: {df['age_outlier_iqr'].sum()}")
    print(f"order_amount outliers flagged: {df['order_amount_outlier_iqr'].sum()}")
    print(df.loc[df["age_outlier_iqr"], ["age", "age_outlier_iqr"]])
    print(df.loc[df["order_amount_outlier_iqr"], ["order_amount", "order_amount_outlier_iqr"]])

    print("\n--- Isolation Forest (multivariate) outlier detection ---")
    df = detect_outliers_isolation_forest(df, columns=["age", "order_amount"])
    print(f"Multivariate outliers flagged: {df['multivariate_outlier'].sum()}")
    print(df.loc[df["multivariate_outlier"], ["age", "order_amount"]])

    print("\n--- Example handling: capping age at IQR bounds ---")
    df_capped = cap_outliers_iqr(df, "age")
    print(df_capped.loc[df["age_outlier_iqr"], "age"])
