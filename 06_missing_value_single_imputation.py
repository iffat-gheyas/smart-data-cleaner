"""
Module M1: Missing Value Handling -- Single Imputation (Core, v1.0)
---------------------------------------------------------------------
A reusable, extensible toolkit for handling missing values. Designed to
be imported not just by this cleaning pipeline, but by future forecasting,
ML, and CRM-analytics projects too -- so every function here takes a
DataFrame in, returns a DataFrame out, with no hidden dependency on the
rest of this specific pipeline.

Strategies supported (per the design spec):
  - Detect and report missing values
  - Leave missing values unchanged
  - Delete rows containing missing values
  - Delete columns containing missing values
  - Replace with a user-defined constant
  - Mean / Median / Mode imputation
  - Forward fill / Backward fill
  - KNN-based imputation (numeric AND categorical)

Module M2 (multiple imputation / MICE) is a SEPARATE module, deferred to
v1.5+ -- see project spec for why these are split.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Detection / reporting
# ---------------------------------------------------------------------------


def missing_value_report(df: pd.DataFrame) -> dict:
    """
    Report missing values per column: count and percentage.

    Kept as a self-contained function here (rather than only relying on
    Module A's diagnostic summary) so this module can be imported and
    used independently in other future projects without needing the
    rest of this specific pipeline.
    """
    return {
        "missing_per_column": df.isna().sum().to_dict(),
        "missing_percent_per_column": (df.isna().mean() * 100).round(2).to_dict(),
    }


# ---------------------------------------------------------------------------
# 2. Row / column deletion
# ---------------------------------------------------------------------------


def drop_rows_with_missing(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Drop rows that have a missing value.

    If `columns` is given, only checks those columns (e.g. drop a row
    only if `email` is missing, ignoring gaps elsewhere). If `columns`
    is None, drops a row if ANY column has a missing value.
    """
    df = df.copy()
    if columns is None:
        return df.dropna(axis=0).reset_index(drop=True)
    return df.dropna(axis=0, subset=columns).reset_index(drop=True)


def drop_columns_with_missing(
    df: pd.DataFrame, threshold_percent: float = 0.0
) -> pd.DataFrame:
    """
    Drop entire columns that have missing values.

    `threshold_percent` lets you be selective: 0.0 (default) drops any
    column with even a single missing value. Setting it higher (e.g.
    50.0) only drops columns that are AT LEAST that percentage missing
    -- useful for a column like our synthetic `unused_column`, which is
    100% empty and clearly not salvageable, versus a column that's only
    5% missing and worth keeping/imputing instead.
    """
    df = df.copy()
    missing_pct = df.isna().mean() * 100
    cols_to_drop = missing_pct[missing_pct > threshold_percent].index.tolist()
    return df.drop(columns=cols_to_drop)


# ---------------------------------------------------------------------------
# 3. Simple per-column fill strategies
# ---------------------------------------------------------------------------
# Deliberately built as a REGISTRY (a dictionary mapping a strategy name
# to a function) rather than a long if/elif chain. This is what makes the
# module "extensible" per the design spec: adding a brand-new strategy
# later means adding one new entry here, not rewriting a big function.

SIMPLE_FILL_STRATEGIES = {
    "mean": lambda s: s.fillna(s.mean()),
    "median": lambda s: s.fillna(s.median()),
    "mode": lambda s: s.fillna(s.mode().iloc[0]) if not s.mode().empty else s,
    "ffill": lambda s: s.ffill(),
    "bfill": lambda s: s.bfill(),
}


def simple_impute(
    df: pd.DataFrame, column: str, strategy: str, constant_value=None
) -> pd.DataFrame:
    """
    Apply one of the simple fill strategies to a single column.

    strategy options: "leave", "constant", "mean", "median", "mode",
    "ffill", "bfill". For "constant", also pass constant_value.
    """
    df = df.copy()

    if strategy == "leave":
        return df

    if strategy == "constant":
        df[column] = df[column].fillna(constant_value)
        return df

    if strategy not in SIMPLE_FILL_STRATEGIES:
        valid = list(SIMPLE_FILL_STRATEGIES.keys()) + ["constant", "leave"]
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {valid}")

    df[column] = SIMPLE_FILL_STRATEGIES[strategy](df[column])
    return df


# ---------------------------------------------------------------------------
# 4. KNN imputation -- numeric
# ---------------------------------------------------------------------------


def knn_impute_numeric(df: pd.DataFrame, columns: list, k: int = 5) -> pd.DataFrame:
    """
    Impute missing values across several NUMERIC columns at once using
    K-Nearest Neighbours.

    Important detail: features are SCALED before computing distances,
    then un-scaled afterwards. Without this, a column with naturally
    large numbers (e.g. order_amount, 10-500) would dominate the
    "nearest neighbour" calculation over a column with small numbers
    (e.g. age, 18-75) -- purely because of scale, not because it's
    actually more similar/relevant. This would silently bias the
    imputed values.

    KNN imputation is applied to all `columns` TOGETHER (not one at a
    time), because "nearest neighbour" is a joint concept across all
    the features being compared -- it wouldn't make sense to compute
    it column-by-column.
    """
    df = df.copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])

    imputer = KNNImputer(n_neighbors=k)
    imputed_scaled = imputer.fit_transform(scaled)

    # Undo the scaling to get values back in the original units.
    imputed = scaler.inverse_transform(imputed_scaled)

    df[columns] = imputed
    return df


# ---------------------------------------------------------------------------
# 5. KNN imputation -- categorical
# ---------------------------------------------------------------------------


def knn_impute_categorical(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_feature_columns: list,
    k: int = 5,
) -> pd.DataFrame:
    """
    Impute a missing CATEGORICAL value using the mode (most frequent
    value) among its k nearest neighbours, where "nearest" is measured
    using the numeric feature columns.

    Approach:
      1. Find each row's k nearest neighbours based on numeric features.
      2. Look at those neighbours' values in `categorical_column`.
      3. Fill the missing value with the most frequent category among
         them (excluding any neighbours that are ALSO missing that
         category, and excluding the row itself).

    REQUIREMENT: `numeric_feature_columns` must already be complete (no
    missing values) -- run knn_impute_numeric() on them first, or use
    another strategy to fill them, so that neighbour distances are
    well-defined for every row.

    Fallback: if none of a row's k neighbours have a known category
    (rare, but possible with a small k or an unusual row), we fall back
    to the overall most frequent category in the column, and report how
    often this fallback was needed -- so you can decide whether to
    increase k for that job.
    """
    df = df.copy()

    feature_matrix = df[numeric_feature_columns]
    if feature_matrix.isna().any().any():
        raise ValueError(
            "numeric_feature_columns must have no missing values before "
            "running knn_impute_categorical(). Run knn_impute_numeric() "
            "(or another fill strategy) on them first."
        )

    missing_mask = df[categorical_column].isna()
    if missing_mask.sum() == 0:
        return df

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # +1 neighbour requested because a row is always its own nearest
    # neighbour (distance 0) -- we exclude itself when reading results.
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df)))
    nn.fit(scaled_features)
    _, neighbour_positions = nn.kneighbors(scaled_features)

    global_mode = df[categorical_column].mode()
    global_mode = global_mode.iloc[0] if not global_mode.empty else np.nan

    filled_values = df[categorical_column].copy()
    fallback_count = 0

    for row_position in np.where(missing_mask)[0]:
        row_index = df.index[row_position]
        neighbour_row_positions = [
            p for p in neighbour_positions[row_position] if p != row_position
        ][:k]
        neighbour_indices = df.index[neighbour_row_positions]

        neighbour_values = df.loc[neighbour_indices, categorical_column].dropna()

        if neighbour_values.empty:
            filled_values.loc[row_index] = global_mode
            fallback_count += 1
        else:
            filled_values.loc[row_index] = neighbour_values.mode().iloc[0]

    df[categorical_column] = filled_values

    if fallback_count > 0:
        print(
            f"  Note: {fallback_count} row(s) had no neighbours with a known "
            f"'{categorical_column}' value -- filled using the overall most "
            f"common category instead. Consider increasing k if this seems high."
        )

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

    print("--- Missing value report (before) ---")
    report = missing_value_report(df)
    for col, pct in report["missing_percent_per_column"].items():
        print(f"  {col}: {report['missing_per_column'][col]} missing ({pct}%)")

    print("\n--- Drop columns that are entirely (100%) missing ---")
    df_test = drop_columns_with_missing(df, threshold_percent=90)
    print(f"Columns before: {len(df.columns)}, after: {len(df_test.columns)}")
    print(f"Dropped: {set(df.columns) - set(df_test.columns)}")

    print("\n--- Simple imputation examples ---")
    df_simple = simple_impute(df, "age", strategy="mean")
    df_simple = simple_impute(df_simple, "order_amount", strategy="median")
    df_simple = simple_impute(df_simple, "notes", strategy="constant", constant_value="No notes provided")
    print("age missing after mean-fill:", df_simple["age"].isna().sum())
    print("order_amount missing after median-fill:", df_simple["order_amount"].isna().sum())
    print("notes missing after constant-fill:", df_simple["notes"].isna().sum())

    print("\n--- KNN numeric imputation ---")
    df_knn = knn_impute_numeric(df, columns=["age", "order_amount"], k=5)
    print("age missing after KNN:", df_knn["age"].isna().sum())
    print("order_amount missing after KNN:", df_knn["order_amount"].isna().sum())
    print(df_knn[["age", "order_amount"]].head(10))

    print("\n--- KNN categorical imputation (country, using age/order_amount as features) ---")
    df_knn_cat = knn_impute_categorical(
        df_knn, categorical_column="country", numeric_feature_columns=["age", "order_amount"], k=5
    )
    print("country missing after categorical KNN:", df_knn_cat["country"].isna().sum())
