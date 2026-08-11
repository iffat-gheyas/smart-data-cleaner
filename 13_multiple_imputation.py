"""
Module M2: Missing Value Handling -- Multiple Imputation (Phase 2 -- new)
----------------------------------------------------------------------------
Sprint 1's Module M1 (06_missing_value_single_imputation.py) explicitly
deferred this to its own module: "Module M2 (multiple imputation / MICE)
is a SEPARATE module, deferred to v1.5+". This is that module.

TWO OUTPUT SHAPES, ONE UNDERLYING COMPUTATION:
The spec (Section 5) describes two legitimate ways to use Multiple
Imputation's `m` posterior draws:
  - CONSOLIDATED (Sections 4.2, 4.3, 6, 7): one completed dataset,
    where every missing cell gets a single representative value (the
    mean/median of its m draws) PLUS companion columns disclosing the
    uncertainty behind that value (_was_imputed, _imputation_sd,
    _imputation_lower_95/_upper_95 for numeric; _top_probability,
    _categorical_uncertainty, per-category probabilities for
    categorical). Good for client delivery, dashboards, demos.
  - M COMPLETE DATASETS (Section 5.1): m separate, fully complete
    datasets, each containing ONE specific plausible draw per missing
    cell (no averaging, no uncertainty columns -- a single draw isn't
    "uncertain", it just IS one specific value). Good for formal
    statistical inference: pooling via Rubin's rules, or training one
    model per dataset and combining the results downstream.

Computing the m draws is the expensive part (m fits of an
IterativeImputer, or m rounds of neighbour sampling per missing row).
This module deliberately SEPARATES "compute the draws" from "do
something with the draws", so requesting both output shapes for the
same column never recomputes the expensive part twice -- see
_compute_numeric_mi_draws / _compute_categorical_mi_draws below, and
how both multiple_impute_* (consolidated) and generate_*_mi_datasets
(M complete datasets) accept an already-computed `draws` argument.

WHAT "MULTIPLE" MEANS HERE, AND WHY IT'S DIFFERENT FROM MODULE M1:
Every strategy in Module M1 (mean, median, mode, KNN) produces ONE
single best-guess value per missing cell and treats it as if it were
real data from then on. Multiple Imputation instead generates SEVERAL
independently-plausible values per missing cell -- what happens next
with those values (consolidate with disclosed uncertainty, or keep
them apart as m separate datasets) is a choice, not a fixed answer.

NUMERIC columns: uses scikit-learn's IterativeImputer with a Bayesian
Ridge estimator and sample_posterior=True -- the standard MICE
(Multivariate Imputation by Chained Equations) approach: each missing
value is estimated from a regression on every OTHER column, refit
iteratively, and sample_posterior=True means each of the `m` runs
draws a genuinely different plausible value from that model's
posterior rather than always returning the same point estimate.

CATEGORICAL columns: literal variance isn't defined for nominal
labels, so this uses stochastic nearest-neighbour categorical
imputation (spec Section 4.3): for each missing row, draw `m` samples
from its k-nearest-neighbours' category distribution.

Every function here follows the same DataFrame-in/DataFrame-out
convention as the rest of the project, and NEVER modifies a value that
wasn't actually missing to begin with.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 -- required to unlock IterativeImputer below
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Numeric -- shared draw computation
# ---------------------------------------------------------------------------


def compute_numeric_mi_draws(df: pd.DataFrame, columns: list, m: int = 10, random_state: int = 42) -> np.ndarray:
    """
    Runs the actual MICE fitting -- m independent IterativeImputer fits
    over `columns` jointly, each with sample_posterior=True so every
    run draws genuinely different plausible values from the model's
    posterior. Returns the raw draws as a (m, n_rows, n_columns) array.

    This is the expensive step, deliberately separated out so it only
    ever runs ONCE per (df, columns, m) combination, regardless of
    whether the caller wants the consolidated view, the M-complete-
    datasets view, or both.
    """
    feature_matrix = df[columns].to_numpy(dtype=float)
    draws = np.empty((m, feature_matrix.shape[0], feature_matrix.shape[1]))

    for i in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            random_state=random_state + i,
            max_iter=10,
        )
        draws[i] = imputer.fit_transform(feature_matrix)

    return draws


# ---------------------------------------------------------------------------
# 2. Numeric -- consolidated output
# ---------------------------------------------------------------------------


def multiple_impute_numeric(
    df: pd.DataFrame,
    columns: list,
    target_columns: list = None,
    m: int = 10,
    consolidation: str = "mean",
    random_state: int = 42,
    draws: np.ndarray = None,
) -> pd.DataFrame:
    """
    Multiple-imputes NUMERIC columns using MICE, consolidated to one
    output value per cell plus full uncertainty columns.

    `columns` is the FULL set of numeric columns used as JOINT
    features for the model (conditioning each estimate on every OTHER
    available numeric column, not just its own values).
    `target_columns` (a subset of `columns`; defaults to all of
    `columns`) is which of those columns actually get their imputed
    value + uncertainty columns written back to `df`.

    `consolidation` is "mean" or "median" of the m draws -- which one
    becomes the actual value written into the column.

    `draws`: pass in an already-computed array from
    compute_numeric_mi_draws() to skip recomputing it (used when the
    caller also needs generate_numeric_mi_datasets() for the same
    columns -- see that function).

    Adds "<col>_was_imputed", "<col>_imputation_sd",
    "<col>_imputation_lower_95" / "_upper_95" (empirical 2.5th/97.5th
    percentiles across the m draws, not a normal approximation) for
    every column in target_columns.
    """
    if consolidation not in ("mean", "median"):
        raise ValueError(f"consolidation must be 'mean' or 'median', got {consolidation!r}")

    df = df.copy()
    if target_columns is None:
        target_columns = columns
    if draws is None:
        draws = compute_numeric_mi_draws(df, columns, m=m, random_state=random_state)

    original_missing_masks = {col: df[col].isna().to_numpy() for col in target_columns}

    consolidated = draws.mean(axis=0) if consolidation == "mean" else np.median(draws, axis=0)
    std_draws = draws.std(axis=0)
    lower_draws = np.percentile(draws, 2.5, axis=0)
    upper_draws = np.percentile(draws, 97.5, axis=0)

    for col in target_columns:
        j = columns.index(col)
        mask = original_missing_masks[col]
        original_values = df[col].to_numpy(dtype=float)

        was_imputed_col = f"{col}_was_imputed"
        sd_col = f"{col}_imputation_sd"
        lower_col = f"{col}_imputation_lower_95"
        upper_col = f"{col}_imputation_upper_95"

        df[was_imputed_col] = mask
        df[sd_col] = 0.0
        df[lower_col] = original_values
        df[upper_col] = original_values

        df.loc[mask, col] = consolidated[mask, j]
        df.loc[mask, sd_col] = np.round(std_draws[mask, j], 2)
        df.loc[mask, lower_col] = np.round(lower_draws[mask, j], 2)
        df.loc[mask, upper_col] = np.round(upper_draws[mask, j], 2)

    return df


# ---------------------------------------------------------------------------
# 3. Numeric -- M complete datasets
# ---------------------------------------------------------------------------


def generate_numeric_mi_datasets(
    df: pd.DataFrame,
    columns: list,
    target_columns: list = None,
    m: int = 10,
    random_state: int = 42,
    draws: np.ndarray = None,
) -> list:
    """
    Returns a list of `m` DataFrames -- one per posterior draw -- each
    with target_columns' originally-missing cells filled using THAT
    SPECIFIC draw (no averaging, no uncertainty columns). Every
    dataset has identical observed values; only the originally-missing
    cells in `target_columns` differ between them, exactly as one
    draw's worth of plausible values would.

    `draws`: pass in an already-computed array to reuse it (e.g. if
    multiple_impute_numeric() was already called for the same columns
    with the same m/random_state, its draws can be reused here instead
    of refitting the model from scratch).
    """
    if target_columns is None:
        target_columns = columns
    if draws is None:
        draws = compute_numeric_mi_draws(df, columns, m=m, random_state=random_state)

    original_missing_masks = {col: df[col].isna().to_numpy() for col in target_columns}

    datasets = []
    for i in range(draws.shape[0]):
        variant = df.copy()
        for col in target_columns:
            j = columns.index(col)
            mask = original_missing_masks[col]
            variant.loc[mask, col] = draws[i][mask, j]
        datasets.append(variant)
    return datasets


# ---------------------------------------------------------------------------
# 4. Categorical -- shared draw computation
# ---------------------------------------------------------------------------


def compute_categorical_mi_draws(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_feature_columns: list,
    k: int = 5,
    m: int = 10,
    random_state: int = 42,
) -> dict:
    """
    For every row missing `categorical_column`, finds its k nearest
    neighbours (by `numeric_feature_columns`, which must already be
    complete) and draws m independent samples from those neighbours'
    category distribution. Returns {row_index: [m sampled categories]}
    -- the shared expensive computation, reused by both
    multiple_impute_categorical() (consolidated) and
    generate_categorical_mi_datasets() (M complete datasets).
    """
    rng = np.random.default_rng(random_state)

    feature_matrix = df[numeric_feature_columns]
    if feature_matrix.isna().any().any():
        raise ValueError(
            "numeric_feature_columns must have no missing values before "
            "running compute_categorical_mi_draws(). Run numeric imputation "
            "on them first."
        )

    missing_mask = df[categorical_column].isna()
    if missing_mask.sum() == 0:
        return {}

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df)))
    nn.fit(scaled_features)
    _, neighbour_positions = nn.kneighbors(scaled_features)

    global_value_counts = df[categorical_column].value_counts()
    global_categories = global_value_counts.index.tolist()
    global_probs = (global_value_counts / global_value_counts.sum()).tolist()

    per_row_draws = {}
    for row_position in np.where(missing_mask)[0]:
        row_index = df.index[row_position]
        neighbour_row_positions = [
            p for p in neighbour_positions[row_position] if p != row_position
        ][:k]
        neighbour_indices = df.index[neighbour_row_positions]
        neighbour_values = df.loc[neighbour_indices, categorical_column].dropna()

        if neighbour_values.empty:
            categories, probs = global_categories, global_probs
        else:
            value_counts = neighbour_values.value_counts()
            categories = value_counts.index.tolist()
            probs = (value_counts / value_counts.sum()).tolist()

        per_row_draws[row_index] = rng.choice(categories, size=m, p=probs)

    return per_row_draws


# ---------------------------------------------------------------------------
# 5. Categorical -- consolidated output
# ---------------------------------------------------------------------------

MAX_CATEGORIES_FOR_FULL_PROBABILITY_COLUMNS = 10


def _add_categorical_probability_columns(
    df: pd.DataFrame, categorical_column: str, missing_mask: pd.Series, probability_records: dict
) -> pd.DataFrame:
    """
    Adds the probability-disclosure columns for one categorical
    column's consolidated Multiple Imputation result.
    `probability_records` is {row_index: {category: probability}}.

    LOW cardinality (<= MAX_CATEGORIES_FOR_FULL_PROBABILITY_COLUMNS
    distinct categories): one "<column>_prob__<category>" column per
    category. HIGH cardinality: "<column>_2nd_choice" /
    "_2nd_probability" / "_3rd_choice" / "_3rd_probability" instead.
    """
    df = df.copy()
    all_categories = sorted({cat for probs in probability_records.values() for cat in probs.keys()})

    if len(all_categories) <= MAX_CATEGORIES_FOR_FULL_PROBABILITY_COLUMNS:
        for category in all_categories:
            col_name = f"{categorical_column}_prob__{category}"
            values = pd.Series(0.0, index=df.index)
            observed_mask = ~missing_mask
            values.loc[observed_mask & (df[categorical_column] == category)] = 1.0
            for row_index, probs in probability_records.items():
                values.loc[row_index] = probs.get(category, 0.0)
            df[col_name] = values
    else:
        second_choice = pd.Series(pd.NA, index=df.index, dtype="object")
        second_prob = pd.Series(np.nan, index=df.index)
        third_choice = pd.Series(pd.NA, index=df.index, dtype="object")
        third_prob = pd.Series(np.nan, index=df.index)
        for row_index, probs in probability_records.items():
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            if len(ranked) > 1:
                second_choice.loc[row_index], second_prob.loc[row_index] = ranked[1][0], round(ranked[1][1], 2)
            if len(ranked) > 2:
                third_choice.loc[row_index], third_prob.loc[row_index] = ranked[2][0], round(ranked[2][1], 2)
        df[f"{categorical_column}_2nd_choice"] = second_choice
        df[f"{categorical_column}_2nd_probability"] = second_prob
        df[f"{categorical_column}_3rd_choice"] = third_choice
        df[f"{categorical_column}_3rd_probability"] = third_prob

    return df


def multiple_impute_categorical(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_feature_columns: list,
    k: int = 5,
    m: int = 10,
    random_state: int = 42,
    per_row_draws: dict = None,
) -> pd.DataFrame:
    """
    Multiple-imputes ONE categorical column using stochastic
    nearest-neighbour categorical imputation, consolidated to one
    output value per cell plus probability-disclosure columns.

    `per_row_draws`: pass in an already-computed dict from
    compute_categorical_mi_draws() to skip recomputing it.

    Writes the HIGHEST-probability category as the actual imputed
    value -- even when that probability is low, per the spec's
    explicit rule that low confidence is disclosed through the
    probability columns, not by leaving the cell empty.
    """
    df = df.copy()

    missing_mask = df[categorical_column].isna()

    was_imputed_col = f"{categorical_column}_was_imputed"
    top_prob_col = f"{categorical_column}_top_probability"
    uncertainty_col = f"{categorical_column}_categorical_uncertainty"

    df[was_imputed_col] = missing_mask
    df[top_prob_col] = 1.0
    df[uncertainty_col] = 0.0

    if missing_mask.sum() == 0:
        return _add_categorical_probability_columns(df, categorical_column, missing_mask, {})

    if per_row_draws is None:
        per_row_draws = compute_categorical_mi_draws(
            df, categorical_column, numeric_feature_columns, k=k, m=m, random_state=random_state
        )

    filled_values = df[categorical_column].copy()
    probability_records = {}

    for row_index, draws in per_row_draws.items():
        draw_counts = pd.Series(draws).value_counts()
        estimated_probs = (draw_counts / len(draws)).to_dict()

        top_category = draw_counts.idxmax()
        top_probability = draw_counts.max() / len(draws)

        filled_values.loc[row_index] = top_category
        df.loc[row_index, top_prob_col] = round(top_probability, 2)
        df.loc[row_index, uncertainty_col] = round(1 - top_probability, 2)
        probability_records[row_index] = estimated_probs

    df[categorical_column] = filled_values
    df = _add_categorical_probability_columns(df, categorical_column, missing_mask, probability_records)
    return df


# ---------------------------------------------------------------------------
# 6. Categorical -- M complete datasets
# ---------------------------------------------------------------------------


def generate_categorical_mi_datasets(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_feature_columns: list,
    k: int = 5,
    m: int = 10,
    random_state: int = 42,
    per_row_draws: dict = None,
) -> list:
    """
    Returns a list of `m` DataFrames, each with `categorical_column`'s
    originally-missing cells filled using ONE specific draw's sampled
    category (no majority vote, no probability columns). Observed
    values are identical across all m datasets.
    """
    if per_row_draws is None:
        per_row_draws = compute_categorical_mi_draws(
            df, categorical_column, numeric_feature_columns, k=k, m=m, random_state=random_state
        )

    datasets = []
    for i in range(m):
        variant = df.copy()
        for row_index, draws in per_row_draws.items():
            variant.loc[row_index, categorical_column] = draws[i]
        datasets.append(variant)
    return datasets


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")
    text_std = import_module("02_text_standardisation")

    filepath = "dirty_customer_data.xlsx"

    df = ingestion.load_data(filepath)
    df = text_std.trim_whitespace(df)
    df = text_std.standardise_casing(df, {"full_name": "title", "company": "title"})

    print("--- Consolidated: numeric (age, order_amount) ---")
    df_mi = multiple_impute_numeric(df, columns=["age", "order_amount"], m=10)
    print(df_mi[[
        "age", "age_was_imputed", "age_imputation_sd", "age_imputation_lower_95", "age_imputation_upper_95"
    ]].head(10))

    print("\n--- M complete datasets: numeric (age, order_amount), m=5 ---")
    datasets = generate_numeric_mi_datasets(df, columns=["age", "order_amount"], m=5)
    print(f"Generated {len(datasets)} datasets. First dataset's age column (first 10 rows):")
    print(datasets[0]["age"].head(10))
    print("Second dataset's age column (should differ where originally missing):")
    print(datasets[1]["age"].head(10))
