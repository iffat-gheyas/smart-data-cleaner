"""
Module AR: Unified Cross-Column Ranking
------------------------------------------
Per the client's explicit request: ONE method for the whole dataset,
selected by aggregating performance across every column needing
imputation -- rather than each column independently picking its own
best method (the earlier, more accurate-per-column design).

FAIRNESS ACROSS COLUMNS: comparing raw errors from a column scaled
0-400 (e.g. Precipitation) against one scaled 7-20 (e.g. Specific
Humidity) would let whichever column has the largest numbers dominate
the aggregate decision. Every column is standardized (Module AI/35)
before backtesting, so every method's error comes out already in a
common, comparable unit (standard deviations from that column's own
trend) -- no further scale normalization needed.

EQUAL WEIGHT PER COLUMN, NOT PER TEST POINT: a column that happens to
have more masked test points shouldn't silently count for more in the
"overall" verdict. Each column's OWN mean error is computed first;
those column-level means are then averaged together, unweighted by
how many points each column happened to contribute.

CONTEXTUAL/NEURAL METHODS ALREADY STANDARDIZE INTERNALLY (Modules
AJ/36, AL/38 detrend+standardize the target and every predictor as
part of their own pipeline) -- feeding them an already-standardized
series would double-standardize, which is close to a no-op but not
worth the confusion. They're evaluated and applied directly, exactly
as built; only the simpler methods (interpolation, seasonal, cross-
series regression) are explicitly wrapped via Module AQ/43.
"""

from importlib import import_module

import numpy as np
import pandas as pd
import re

backtest_mod = import_module("27_time_series_backtesting")
fold_transform_mod = import_module("35_time_series_fold_transforms")
wrapper_mod = import_module("43_time_series_standardized_wrapper")
apply_mod = import_module("28_time_series_apply_imputation")

# Methods that already do their own internal detrend/standardize as
# part of their pipeline -- evaluated/applied directly, never wrapped
# again by Module AQ/43 (which would double-standardize).
def _is_already_internally_standardized(method_name: str) -> bool:
    return method_name.startswith("contextual regression") or method_name.startswith("neural")



CROSS_SERIES_PATTERN = re.compile(r"^cross-series regression \(via .+\)$")
GENERIC_CROSS_SERIES_NAME = "cross-series regression"


def _normalize_method_name(method_name: str) -> str:
    """
    Cross-series regression's real candidate name includes WHICH
    predictor was best for that specific column (e.g. "cross-series
    regression (via Temperature)") -- a different target column will
    almost always find a different best predictor, so this exact name
    essentially never qualifies as a candidate for more than one
    column. Confirmed directly as a real problem: it could "win" the
    unified ranking based on a sample of just ONE column, since no
    other column ever offers that exact same named candidate. For
    ranking/aggregation purposes, this collapses to one generic
    "cross-series regression" identity; each column still uses
    whichever predictor is genuinely best for IT when evaluated or
    applied -- only the AGGREGATE label is generic, not the underlying
    per-column computation.
    """
    if CROSS_SERIES_PATTERN.match(method_name):
        return GENERIC_CROSS_SERIES_NAME
    return method_name


def compute_standardized_errors_for_column(
    series: pd.Series, method_name: str, cycle_length: int = None, other_series_dict: dict = None,
    mask_positions: list = None, standardization_method: str = "zscore",
) -> dict:
    """
    Standardizes `series` (and every column in other_series_dict,
    each independently using its OWN trend/scale), masks a handful of
    currently-observed points, runs `method_name` DIRECTLY on the
    standardized data, and returns the per-point ABSOLUTE errors --
    already in standardized units, ready to aggregate fairly across
    columns with Module AR's own logic below.

    method_name may be the generic "cross-series regression" -- this
    column's own best-correlated predictor is resolved automatically
    (see _normalize_method_name()'s docstring for why the generic name
    is used at the ranking layer).

    Returns {"n_tested", "n_successful", "errors": np.ndarray} --
    errors is empty if the method couldn't be tested at all.
    """
    fold_transform = fold_transform_mod.fit_fold_transform(series, standardization_method)
    standardized_series = fold_transform_mod.transform(series, fold_transform)

    standardized_other = None
    if other_series_dict:
        standardized_other = {}
        for col, other_series in other_series_dict.items():
            other_ft = fold_transform_mod.fit_fold_transform(other_series, standardization_method)
            standardized_other[col] = fold_transform_mod.transform(other_series, other_ft)

    if mask_positions is None:
        mask_positions = backtest_mod.select_mask_positions(standardized_series, random_seed=42 if cycle_length else 42)
    if not mask_positions:
        return {"n_tested": 0, "n_successful": 0, "errors": np.array([])}

    true_values = standardized_series.loc[mask_positions].copy()
    masked_series = standardized_series.copy()
    masked_series.loc[mask_positions] = np.nan

    all_methods, _ = backtest_mod.build_candidate_methods(masked_series, cycle_length, standardized_other)

    if method_name == GENERIC_CROSS_SERIES_NAME:
        resolved_name = next((m for m in all_methods if CROSS_SERIES_PATTERN.match(m)), None)
        if resolved_name is None:
            return {"n_tested": len(mask_positions), "n_successful": 0, "errors": np.array([])}
    else:
        resolved_name = method_name

    if resolved_name not in all_methods:
        return {"n_tested": len(mask_positions), "n_successful": 0, "errors": np.array([])}

    imputed, was_imputed, could_not_fill = all_methods[resolved_name](masked_series)
    tested_positions = [p for p in mask_positions if was_imputed.get(p, False)]
    if not tested_positions:
        return {"n_tested": len(mask_positions), "n_successful": 0, "errors": np.array([])}

    errors = (imputed.loc[tested_positions] - true_values.loc[tested_positions]).abs().to_numpy(dtype=float)
    return {"n_tested": len(mask_positions), "n_successful": len(tested_positions), "errors": errors}


def run_unified_single_value_backtest(
    columns_data: dict, cycle_lengths: dict = None, other_series_dicts: dict = None,
) -> pd.DataFrame:
    """
    columns_data: {column_name: pd.Series}, every column needing
    imputation.
    cycle_lengths: {column_name: cycle_length or None}.
    other_series_dicts: {column_name: {other_col: pd.Series}} -- the
    OTHER columns available as predictors for each target (typically
    every other column in columns_data plus any non-imputed columns).

    Returns ONE results_df -- every candidate method that could be
    tested on at least one column, with its standardized MAE averaged
    EQUALLY across every column it was tested on (not weighted by how
    many test points each column happened to contribute), sorted
    best-to-worst.
    """
    cycle_lengths = cycle_lengths or {}
    other_series_dicts = other_series_dicts or {}

    candidate_names = set()
    for col, series in columns_data.items():
        methods_dict, _ = backtest_mod.build_candidate_methods(
            series, cycle_lengths.get(col), other_series_dicts.get(col)
        )
        candidate_names.update(_normalize_method_name(m) for m in methods_dict)

    rows = []
    for method_name in sorted(candidate_names):
        per_column_means = []
        columns_tested = []
        n_tested_total = 0
        n_successful_total = 0

        for col, series in columns_data.items():
            result = compute_standardized_errors_for_column(
                series, method_name, cycle_lengths.get(col), other_series_dicts.get(col)
            )
            n_tested_total += result["n_tested"]
            n_successful_total += result["n_successful"]
            if result["n_successful"] > 0:
                per_column_means.append(float(np.mean(result["errors"])))
                columns_tested.append(col)

        overall_mae = float(np.mean(per_column_means)) if per_column_means else np.nan

        rows.append(
            {
                "method": method_name,
                "n_columns_tested": len(columns_tested),
                "columns_tested": ", ".join(columns_tested),
                "n_tested": n_tested_total,
                "n_successful": n_successful_total,
                "n_failed": n_tested_total - n_successful_total,
                "standardized_mae": overall_mae,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df["_sort_key"] = results_df["standardized_mae"].fillna(np.inf)
    results_df = results_df.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    return results_df


def recommend_unified_method(results_df: pd.DataFrame) -> str:
    """The rank-1 method's name -- results_df is already sorted best-first."""
    if results_df.empty or results_df.iloc[0]["n_successful"] == 0:
        return None
    return results_df.iloc[0]["method"]


def apply_unified_method_to_column(
    df: pd.DataFrame, value_column: str, method_name: str, cycle_length: int = None,
    other_series_dict: dict = None, bounds: dict = None,
) -> tuple:
    """
    Applies the ONE chosen (overall-best) method to `value_column`'s
    REAL gaps, on that column's OWN original scale. Contextual/neural
    methods (which already internally standardize) go through Module
    AB/28's existing apply_chosen_imputation unchanged; every other
    method is applied via Module AQ/43's standardize-wrap, matching
    exactly how it was evaluated during ranking -- consistency between
    "what was ranked" and "what gets applied" matters here, since some
    methods (PCHIP, cubic spline) genuinely behave differently at
    different scales, unlike provably scale-invariant linear
    interpolation.

    Returns (df_with_imputation, report) -- same shape as Module AB/28's own functions.
    """
    if _is_already_internally_standardized(method_name):
        result_df, report = apply_mod.apply_chosen_imputation(df, value_column, method_name, cycle_length, other_series_dict)
        if bounds:
            bounds_mod = import_module("42_time_series_plausibility_bounds")
            was_imputed_mask = result_df[f"{value_column}_was_imputed"] == 1
            result_df[value_column] = bounds_mod.clip_only_imputed_values(
                result_df[value_column], was_imputed_mask, bounds.get("lower"), bounds.get("upper")
            )
        return result_df, report

    # For everything else, apply via the standardize-wrap directly,
    # then build the same output shape apply_chosen_imputation produces.
    series = df.set_index("Time_Step")[value_column]
    all_methods, _ = backtest_mod.build_candidate_methods(series, cycle_length, other_series_dict)

    if method_name == GENERIC_CROSS_SERIES_NAME:
        # Resolve to whichever predictor is genuinely best for THIS
        # column's real data -- not a name carried over from ranking
        # time (which may have used a different column's predictor).
        resolved_name = next((m for m in all_methods if CROSS_SERIES_PATTERN.match(m)), None)
        if resolved_name is None:
            raise ValueError(f"Cross-series regression has no usable predictor for column '{value_column}'.")
    else:
        resolved_name = method_name

    if resolved_name not in all_methods:
        raise ValueError(f"'{method_name}' is not a valid candidate for column '{value_column}'.")

    raw_method_func = all_methods[resolved_name]
    imputed, was_imputed, could_not_fill = wrapper_mod.standardize_wrap_method(series, raw_method_func)

    if bounds:
        bounds_mod = import_module("42_time_series_plausibility_bounds")
        imputed = bounds_mod.clip_only_imputed_values(imputed, was_imputed, bounds.get("lower"), bounds.get("upper"))

    df = df.copy()
    imputed_aligned = imputed.reindex(df["Time_Step"]).to_numpy()
    was_imputed_aligned = was_imputed.reindex(df["Time_Step"]).fillna(False).to_numpy()
    could_not_fill_aligned = could_not_fill.reindex(df["Time_Step"]).fillna(False).to_numpy()

    df[value_column] = imputed_aligned
    df[f"{value_column}_was_imputed"] = was_imputed_aligned.astype(int)
    df[f"{value_column}_imputation_method"] = np.where(was_imputed_aligned, resolved_name, None)

    report = {
        "method": resolved_name,
        "n_imputed": int(was_imputed_aligned.sum()),
        "n_could_not_fill": int(could_not_fill_aligned.sum()),
        "time_steps_could_not_fill": df.loc[could_not_fill_aligned, "Time_Step"].tolist(),
    }
    return df, report


eval_mod = import_module("39_time_series_mi_evaluation_metrics")


def evaluate_unified_mi_method_for_column(
    series: pd.Series, method_name: str, mask_positions: list, cycle_length: int = None,
    other_series_dict: dict = None, m: int = 10, random_seed: int = 42, standardization_method: str = "zscore",
) -> dict:
    """
    Same leave-one-out MI evaluation as Module AO/41's evaluate_mi_method
    (never using a masked point's own error to build its own interval),
    but operating on the STANDARDIZED series -- so interval accuracy and
    MAAPE-based accuracy are computed in a common, cross-column-
    comparable unit, exactly like Module AR's single-value ranking above.

    Returns {"n_successful", "combined_score"} or None if untestable.
    """
    fold_transform = fold_transform_mod.fit_fold_transform(series, standardization_method)
    standardized_series = fold_transform_mod.transform(series, fold_transform)

    standardized_other = None
    if other_series_dict:
        standardized_other = {}
        for col, other_series in other_series_dict.items():
            other_ft = fold_transform_mod.fit_fold_transform(other_series, standardization_method)
            standardized_other[col] = fold_transform_mod.transform(other_series, other_ft)

    true_values = standardized_series.loc[mask_positions].copy()
    masked_series = standardized_series.copy()
    masked_series.loc[mask_positions] = np.nan

    all_methods, _ = backtest_mod.build_candidate_methods(masked_series, cycle_length, standardized_other)

    if method_name == GENERIC_CROSS_SERIES_NAME:
        resolved_name = next((m for m in all_methods if CROSS_SERIES_PATTERN.match(m)), None)
    else:
        resolved_name = method_name
    if resolved_name is None or resolved_name not in all_methods:
        return None

    imputed, was_imputed, could_not_fill = all_methods[resolved_name](masked_series)
    tested_positions = [p for p in mask_positions if was_imputed.get(p, False)]
    if not tested_positions:
        return None

    all_residuals = (imputed.loc[tested_positions] - true_values.loc[tested_positions]).to_numpy(dtype=float)

    rng = np.random.RandomState(random_seed)
    lowers, uppers, medians = [], [], []
    for i, pos in enumerate(tested_positions):
        loo_residuals = np.delete(all_residuals, i)
        if len(loo_residuals) == 0:
            loo_residuals = all_residuals
        point_est = imputed.loc[pos]
        draws = point_est + rng.choice(loo_residuals, size=m, replace=True)
        lowers.append(np.percentile(draws, 2.5))
        uppers.append(np.percentile(draws, 97.5))
        medians.append(np.median(draws))

    true_at_tested = true_values.loc[tested_positions].to_numpy(dtype=float)
    interval_accuracy_pct = eval_mod.compute_interval_coverage_pct(true_at_tested, lowers, uppers)
    maape_value = eval_mod.compute_maape(true_at_tested, medians)
    maape_accuracy_pct = eval_mod.maape_to_accuracy_pct(maape_value)
    combined_score = eval_mod.compute_combined_score(interval_accuracy_pct, maape_accuracy_pct)

    return {"n_successful": len(tested_positions), "combined_score": combined_score}


def run_unified_mi_backtest(
    columns_data: dict, cycle_lengths: dict = None, other_series_dicts: dict = None, m: int = 10, random_seed: int = 42,
) -> pd.DataFrame:
    """
    Multiple Imputation's equivalent of run_unified_single_value_backtest()
    above -- aggregates each method's combined_score (interval accuracy
    x2 + MAAPE-based accuracy x1, Module AM/39) across every column,
    EQUAL WEIGHT PER COLUMN, into ONE overall ranking, sorted best-to-
    worst (higher combined_score = better).
    """
    cycle_lengths = cycle_lengths or {}
    other_series_dicts = other_series_dicts or {}

    candidate_names = set()
    for col, series in columns_data.items():
        methods_dict, _ = backtest_mod.build_candidate_methods(series, cycle_lengths.get(col), other_series_dicts.get(col))
        candidate_names.update(_normalize_method_name(m) for m in methods_dict)

    rows = []
    for method_name in sorted(candidate_names):
        per_column_scores = []
        columns_tested = []
        n_successful_total = 0

        for col, series in columns_data.items():
            mask_positions = backtest_mod.select_mask_positions(series, random_seed=42 if cycle_lengths.get(col) else 42)
            if not mask_positions:
                continue
            result = evaluate_unified_mi_method_for_column(
                series, method_name, mask_positions, cycle_lengths.get(col), other_series_dicts.get(col), m, random_seed
            )
            if result:
                per_column_scores.append(result["combined_score"])
                columns_tested.append(col)
                n_successful_total += result["n_successful"]

        overall_score = float(np.mean(per_column_scores)) if per_column_scores else np.nan

        rows.append(
            {
                "method": method_name,
                "n_columns_tested": len(columns_tested),
                "columns_tested": ", ".join(columns_tested),
                "n_successful": n_successful_total,
                "combined_score": overall_score,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df["_sort_key"] = results_df["combined_score"].fillna(-np.inf)
    results_df = results_df.sort_values("_sort_key", ascending=False).drop(columns="_sort_key").reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    return results_df


def recommend_unified_mi_method(results_df: pd.DataFrame) -> str:
    """The rank-1 method's name -- results_df is already sorted best-first by combined_score."""
    if results_df.empty or results_df.iloc[0]["n_successful"] == 0:
        return None
    return results_df.iloc[0]["method"]


def generate_unified_mi_draws_for_column(
    df: pd.DataFrame, value_column: str, method_name: str, cycle_length: int = None,
    other_series_dict: dict = None, m: int = 10, random_seed: int = 42, standardization_method: str = "zscore",
) -> list:
    """
    Generates M draws for `method_name` on `value_column`'s REAL gaps,
    operating in STANDARDIZED space throughout (consistent with how
    the unified ranking evaluated this method -- residual distribution,
    point estimate, and bootstrap addition all happen in standardized
    units), then inverse-transforms the FINAL draws back to this
    column's own original scale. Returns a list of M pd.Series, same
    shape Module AN/40's generalized draws already produce.
    """
    series = df.set_index("Time_Step")[value_column]
    fold_transform = fold_transform_mod.fit_fold_transform(series, standardization_method)
    standardized_series = fold_transform_mod.transform(series, fold_transform)

    standardized_other = None
    if other_series_dict:
        standardized_other = {}
        for col, other_series in other_series_dict.items():
            other_ft = fold_transform_mod.fit_fold_transform(other_series, standardization_method)
            standardized_other[col] = fold_transform_mod.transform(other_series, other_ft)

    mask_positions = backtest_mod.select_mask_positions(standardized_series, random_seed=42 if cycle_length else 42)
    if not mask_positions:
        raise ValueError(f"Not enough observed data in '{value_column}' to backtest '{method_name}' reliably.")

    true_values = standardized_series.loc[mask_positions].copy()
    masked_series = standardized_series.copy()
    masked_series.loc[mask_positions] = np.nan

    all_methods_masked, _ = backtest_mod.build_candidate_methods(masked_series, cycle_length, standardized_other)
    resolved_name = (
        next((m for m in all_methods_masked if CROSS_SERIES_PATTERN.match(m)), None)
        if method_name == GENERIC_CROSS_SERIES_NAME else method_name
    )
    if resolved_name is None or resolved_name not in all_methods_masked:
        raise ValueError(f"'{method_name}' is not a valid candidate for column '{value_column}'.")

    imputed, was_imputed, _ = all_methods_masked[resolved_name](masked_series)
    tested_positions = [p for p in mask_positions if was_imputed.get(p, False)]
    if not tested_positions:
        raise ValueError(f"Could not compute a residual distribution for '{resolved_name}' on '{value_column}'.")
    residuals = (imputed.loc[tested_positions] - true_values.loc[tested_positions]).to_numpy(dtype=float)

    all_methods_real, _ = backtest_mod.build_candidate_methods(standardized_series, cycle_length, standardized_other)
    resolved_real_name = (
        next((m for m in all_methods_real if CROSS_SERIES_PATTERN.match(m)), None)
        if method_name == GENERIC_CROSS_SERIES_NAME else method_name
    )
    if resolved_real_name is None or resolved_real_name not in all_methods_real:
        raise ValueError(f"'{method_name}' cannot be applied to column '{value_column}'.")
    point_estimate_standardized, _, _ = all_methods_real[resolved_real_name](standardized_series)

    was_missing = series.isna()
    rng = np.random.RandomState(random_seed)
    draws = []
    for _ in range(m):
        draw_standardized = standardized_series.copy()
        for idx in series.index[was_missing]:
            point_value = point_estimate_standardized.loc[idx]
            if pd.isna(point_value):
                continue
            draw_standardized.loc[idx] = point_value + rng.choice(residuals)
        draw_original_scale = fold_transform_mod.inverse_transform(draw_standardized, fold_transform)
        draws.append(draw_original_scale)

    return draws


def apply_unified_mi_to_column(
    df: pd.DataFrame, value_column: str, method_name: str, cycle_length: int = None,
    other_series_dict: dict = None, m: int = 10, random_seed: int = 42, output_mode: str = "consolidated",
    bounds: dict = None,
) -> dict:
    """
    Generates unified (standardized-space) draws and consolidates them
    via Module 29's own consolidate_draws() -- reused as-is, producing
    the same "<col>_was_imputed"/"_imputation_sd"/"_imputation_lower_95"/
    "_imputation_upper_95" columns and/or M-datasets regardless of
    which method or column generated the draws.
    """
    multi_imp_mod = import_module("29_time_series_multiple_imputation")

    draws = generate_unified_mi_draws_for_column(df, value_column, method_name, cycle_length, other_series_dict, m, random_seed)

    if bounds:
        bounds_mod = import_module("42_time_series_plausibility_bounds")
        was_missing = df.set_index("Time_Step")[value_column].isna()
        draws = [bounds_mod.clip_only_imputed_values(d, was_missing, bounds.get("lower"), bounds.get("upper")) for d in draws]

    result = multi_imp_mod.consolidate_draws(df, value_column, draws, output_mode)
    was_missing = df.set_index("Time_Step")[value_column].isna()
    result["n_imputed"] = int(was_missing.sum())
    result["m"] = m
    result["method_used"] = method_name
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: fairness -- two columns with VASTLY different scales but similar relative noise")
    print("must contribute comparably to the aggregate ranking, not have the larger-scale column dominate")
    print("=" * 60)
    n = 40
    trend = np.linspace(10, 20, n)
    small_scale = pd.Series(trend + np.random.RandomState(1).normal(0, 0.5, n), index=pd.Index(range(n), name="Time_Step"))
    large_scale = pd.Series((trend + np.random.RandomState(1).normal(0, 0.5, n)) * 100, index=pd.Index(range(n), name="Time_Step"))  # same relative noise, 100x scale
    small_scale.loc[[10, 25]] = None
    large_scale.loc[[10, 25]] = None

    results = run_unified_single_value_backtest({"Small": small_scale, "Large": large_scale}, cycle_lengths={"Small": None, "Large": None})
    print(results[["rank", "method", "n_columns_tested", "standardized_mae"]].head(6).to_string(index=False))
    print("(Since both columns have the same RELATIVE noise pattern, standardized MAE should be similar for a given")
    print(" method regardless of which column's raw scale is bigger -- confirming fairness.)")

    print("\n" + "=" * 60)
    print("TEST 2: apply the chosen unified method to a column's REAL gaps, correct original scale")
    print("=" * 60)
    recommended = recommend_unified_method(results)
    print("Recommended overall method:", recommended)
    df_large = pd.DataFrame({"Time_Step": range(n), "Large": large_scale.to_numpy()})
    result_df, report = apply_unified_method_to_column(df_large, "Large", recommended, cycle_length=None)
    print("Imputed values (should be on the ORIGINAL large scale, ~1000-2000 range, not standardized ~0-1):")
    print(result_df.loc[[10, 25], ["Time_Step", "Large"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST 3: multivariate case -- contextual regression candidates also participate")
    print("=" * 60)
    rainfall_df = pd.read_csv("Rainfall_multivariate_test.csv")
    rainfall_df["Time_Step"] = range(len(rainfall_df))
    indexed = rainfall_df.set_index("Time_Step")
    columns_data = {
        "Specific Humidity": indexed["Specific Humidity"],
        "Temperature": indexed["Temperature"],
        "Precipitation": indexed["Precipitation"],
    }
    other_dicts = {
        col: {other: indexed[other] for other in columns_data if other != col}
        for col in columns_data
    }
    cycle_lengths = {col: 12 for col in columns_data}
    results_mv = run_unified_single_value_backtest(columns_data, cycle_lengths, other_dicts)
    print(results_mv[["rank", "method", "n_columns_tested", "standardized_mae"]].head(8).to_string(index=False))
    print()
    print("Overall recommended method for ALL THREE columns:", recommend_unified_method(results_mv))
