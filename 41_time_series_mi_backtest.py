"""
Module AO: Multiple Imputation Backtest & Ranking
------------------------------------------------------
Extends the single-value backtest's philosophy (Module AA/27: mask
currently-observed points, test every candidate, rank by evidence) to
Multiple Imputation -- but scored differently, since MI's whole point
is producing a calibrated UNCERTAINTY interval, not just one number.

Per the client's specification, each method is scored on two measures:
  1. INTERVAL ACCURACY: % of masked test points whose true value falls
     within that method's own constructed 95% interval.
  2. MAAPE-BASED ACCURACY: using the median of the method's draws as
     its point estimate (Module AM/39).
Combined via a weighted average -- interval accuracy counts TWICE as
heavily as MAAPE-based accuracy (Module AM's default weights).

LEAVE-ONE-OUT TO AVOID CIRCULARITY: naively bootstrapping each masked
point's own interval from a residual pool that INCLUDES that same
point's own error would make the interval look artificially well-
calibrated -- it's partly built from the very answer being checked
against. Instead, for each masked point, its interval is built from
every OTHER masked point's residual only, never its own -- a genuine
held-out check, at the cost of a slightly smaller residual pool per
point (workable given typical mask counts of 5-8 points; degenerates
gracefully, not incorrectly, when only 1 point succeeded at all).
"""

from importlib import import_module

import numpy as np
import pandas as pd

backtest_mod = import_module("27_time_series_backtesting")
eval_mod = import_module("39_time_series_mi_evaluation_metrics")


def evaluate_mi_method(
    series: pd.Series, method_name: str, mask_positions: list, cycle_length: int = None,
    other_series_dict: dict = None, m: int = 10, random_seed: int = 42,
) -> dict:
    """
    Leave-one-out evaluation of one method's Multiple-Imputation
    quality at the given masked positions. Returns a dict with
    n_tested, n_successful, n_failed, interval_accuracy_pct,
    maape_accuracy_pct, combined_score -- NaN scores if the method
    couldn't be tested at all (0 successful positions).
    """
    true_values = series.loc[mask_positions].copy()
    masked_series = series.copy()
    masked_series.loc[mask_positions] = np.nan

    all_methods, _ = backtest_mod.build_candidate_methods(masked_series, cycle_length, other_series_dict)
    if method_name not in all_methods:
        return {
            "n_tested": len(mask_positions), "n_successful": 0, "n_failed": len(mask_positions),
            "interval_accuracy_pct": np.nan, "maape_accuracy_pct": np.nan, "combined_score": np.nan,
        }

    imputed, was_imputed, could_not_fill = all_methods[method_name](masked_series)
    tested_positions = [p for p in mask_positions if was_imputed.get(p, False)]
    n_failed = len(mask_positions) - len(tested_positions)

    if not tested_positions:
        return {
            "n_tested": len(mask_positions), "n_successful": 0, "n_failed": n_failed,
            "interval_accuracy_pct": np.nan, "maape_accuracy_pct": np.nan, "combined_score": np.nan,
        }

    all_residuals = (imputed.loc[tested_positions] - true_values.loc[tested_positions]).to_numpy(dtype=float)

    rng = np.random.RandomState(random_seed)
    lowers, uppers, medians = [], [], []
    for i, pos in enumerate(tested_positions):
        loo_residuals = np.delete(all_residuals, i)  # every OTHER masked point's residual -- never this point's own
        if len(loo_residuals) == 0:
            # Only one point succeeded at all -- can't truly leave-one-out
            # with n=1. Falls back to using its own residual rather than
            # crashing; the resulting single-point "accuracy" is honestly
            # thin/noisy either way, disclosed via a very small n_successful
            # rather than hidden.
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

    return {
        "n_tested": len(mask_positions),
        "n_successful": len(tested_positions),
        "n_failed": n_failed,
        "interval_accuracy_pct": interval_accuracy_pct,
        "maape_accuracy_pct": maape_accuracy_pct,
        "combined_score": combined_score,
    }


def run_mi_backtest(
    series: pd.Series, cycle_length: int = None, other_series_dict: dict = None,
    m: int = 10, mask_positions: list = None, random_seed: int = 42,
) -> tuple:
    """
    Returns (results_df, mask_positions_used). results_df has one row
    per candidate method (the SAME set Module AA/27's single-value
    backtest tests, via the shared build_candidate_methods()), sorted
    BEST-TO-WORST by combined_score DESCENDING (higher = better, since
    these are accuracy percentages, unlike MAE where lower is better).
    """
    if mask_positions is None:
        mask_positions = backtest_mod.select_mask_positions(series, random_seed=42 if cycle_length else 42)
    if not mask_positions:
        return pd.DataFrame(), []

    masked_series = series.copy()
    masked_series.loc[mask_positions] = np.nan
    all_methods, ineligible_neural_rows = backtest_mod.build_candidate_methods(masked_series, cycle_length, other_series_dict)

    rows = []
    for ineligible_row in ineligible_neural_rows:
        rows.append(
            {
                "method": ineligible_row["method"],
                "n_tested": len(mask_positions),
                "n_successful": 0,
                "n_failed": len(mask_positions),
                "interval_accuracy_pct": np.nan,
                "maape_accuracy_pct": np.nan,
                "combined_score": np.nan,
                "note": ineligible_row["note"],
            }
        )

    for method_name in all_methods:
        result = evaluate_mi_method(series, method_name, mask_positions, cycle_length, other_series_dict, m, random_seed)
        rows.append({"method": method_name, "note": "", **result})

    results_df = pd.DataFrame(rows)
    results_df["_sort_key"] = results_df["combined_score"].fillna(-np.inf)
    results_df = results_df.sort_values("_sort_key", ascending=False).drop(columns="_sort_key").reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    return results_df, mask_positions


def recommend_mi_method(results_df: pd.DataFrame) -> str:
    """The rank-1 method's name -- results_df is already sorted best-first by combined_score."""
    if results_df.empty or results_df.iloc[0]["n_successful"] == 0:
        return None
    return results_df.iloc[0]["method"]


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: full MI ranking on the shampoo dataset -- must produce a sensible, complete ranking")
    print("=" * 60)
    sales = [266.0, 145.9, 183.1, 119.3, 180.3, 168.5, 231.8, 224.5, 192.8, 122.9, 336.5, 185.9,
             194.3, 149.5, 210.1, 273.3, 191.4, 287.0, 226.0, 303.6, 289.9, 421.6, 264.5, 342.3,
             339.7, 440.4, 315.9, 439.3, 401.3, 437.4, 575.5, 407.6, 682.0, 475.3, 581.3, 646.9]
    series = pd.Series(sales, index=pd.Index(range(len(sales)), name="Time_Step"))
    series.loc[[8, 13]] = None  # genuine gaps, untouched by masking

    results, mask_positions = run_mi_backtest(series, cycle_length=12, m=15)
    print("Masked test positions:", mask_positions)
    print()
    print(results[["rank", "method", "n_successful", "interval_accuracy_pct", "maape_accuracy_pct", "combined_score"]].to_string(index=False))
    print()
    print("Recommended MI method:", recommend_mi_method(results))

    print("\n" + "=" * 60)
    print("TEST 2: confirm combined_score is EXACTLY the weighted average for the top-ranked row (hand check)")
    print("=" * 60)
    top_row = results.iloc[0]
    expected = (2 * top_row["interval_accuracy_pct"] + 1 * top_row["maape_accuracy_pct"]) / 3
    print(f"Reported combined_score: {top_row['combined_score']:.4f}")
    print(f"Hand-computed (2*interval + 1*maape)/3: {expected:.4f}")
    print("Exact match:", np.isclose(top_row["combined_score"], expected))

    print("\n" + "=" * 60)
    print("TEST 3: ranking is genuinely descending by combined_score (best first)")
    print("=" * 60)
    scores = results["combined_score"].fillna(-np.inf).tolist()
    print("Monotonically non-increasing:", all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))

    print("\n" + "=" * 60)
    print("TEST 4: multivariate case -- cross-series/contextual candidates also get MI-ranked")
    print("=" * 60)
    rainfall_df = pd.read_csv("Rainfall_multivariate_test.csv")
    rainfall_df["Time_Step"] = range(len(rainfall_df))
    other_series = {
        "Specific Humidity": rainfall_df.set_index("Time_Step")["Specific Humidity"],
        "Temperature": rainfall_df.set_index("Time_Step")["Temperature"],
    }
    precip_series = rainfall_df.set_index("Time_Step")["Precipitation"]
    results_mv, _ = run_mi_backtest(precip_series, cycle_length=12, other_series_dict=other_series, m=15)
    print(results_mv[["rank", "method", "n_successful", "interval_accuracy_pct", "maape_accuracy_pct", "combined_score"]].head(8).to_string(index=False))

    print("\n" + "=" * 60)
    print("TEST 5: original series never mutated")
    print("=" * 60)
    before = series.copy()
    run_mi_backtest(series, cycle_length=12, m=15)
    print("Series unchanged after running MI backtest:", series.equals(before))
