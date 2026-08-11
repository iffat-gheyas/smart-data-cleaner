"""
Module AN: Generalized Multiple-Imputation Draws
----------------------------------------------------
Extends uncertainty quantification to EVERY single-imputation method
(interpolation, seasonal, cross-series regression, contextual
regression/tree models, neural) -- not just the one seasonal-trend
residual bootstrap method built earlier -- by reusing each method's
OWN historical prediction errors (measured via the exact same backtest
masking already used for ranking, Module AA/27) as its residual/
uncertainty distribution.

APPROACH:
  1. Mask a handful of currently-observed points (Module AA's own
     masking), run the chosen method against the masked series, and
     compute its (predicted - true) residuals at those points -- this
     is the method's own empirical error distribution on THIS dataset.
  2. Compute the method's deterministic point estimate at the
     GENUINELY missing positions (reusing Module AB/28's
     apply_chosen_imputation, which already knows how to apply any
     named method -- including the dynamically-labeled cross-series/
     contextual/neural ones -- to real gaps).
  3. For each of M draws: point_estimate + a bootstrap sample from the
     residual distribution from step 1.

This generalizes the exact idea the seasonal-bootstrap method already
used (residual bootstrap around a point estimate), applied uniformly
to any method's own residuals rather than one specific decomposition --
every method gets a principled way to express uncertainty, not only
the one built first.

SAME DISCLOSED SCOPE LIMITATION as the original seasonal-bootstrap
method: one GLOBAL residual pool per method (not position-specific),
shared across every missing position that method fills -- a
defensible simplification given how few residual observations a
typical backtest produces (often just 5-8 masked points), not silently
pretending otherwise.
"""

from importlib import import_module

import numpy as np
import pandas as pd

backtest_mod = import_module("27_time_series_backtesting")
apply_mod = import_module("28_time_series_apply_imputation")
bounds_mod = import_module("42_time_series_plausibility_bounds")


def compute_method_residuals(
    series: pd.Series, method_name: str, cycle_length: int = None, other_series_dict: dict = None, mask_positions: list = None
) -> np.ndarray:
    """
    Returns the array of (predicted - true) residuals for `method_name`
    at a set of temporarily-masked, CURRENTLY-OBSERVED points -- reusing
    the exact same masking Module AA/27's ranking backtest uses (via
    its now-shared build_candidate_methods()), so this can never drift
    out of sync with what the ranking table itself tested.

    Returns an empty array if the method couldn't be tested at all
    (e.g. too few observed points to mask at all, the method isn't a
    valid candidate for this dataset, or it genuinely failed at every
    masked position) -- callers should treat an empty array as "no
    usable residual distribution", not silently proceed with it.
    """
    if mask_positions is None:
        mask_positions = backtest_mod.select_mask_positions(series, random_seed=42 if cycle_length else 42)
    if not mask_positions:
        return np.array([])

    true_values = series.loc[mask_positions].copy()
    masked_series = series.copy()
    masked_series.loc[mask_positions] = np.nan

    all_methods, _ = backtest_mod.build_candidate_methods(masked_series, cycle_length, other_series_dict)
    if method_name not in all_methods:
        return np.array([])

    imputed, was_imputed, could_not_fill = all_methods[method_name](masked_series)
    tested_positions = [p for p in mask_positions if was_imputed.get(p, False)]
    if not tested_positions:
        return np.array([])

    return (imputed.loc[tested_positions] - true_values.loc[tested_positions]).to_numpy(dtype=float)


def generate_generalized_mi_draws(
    df: pd.DataFrame, value_column: str, method_name: str, cycle_length: int = None,
    other_series_dict: dict = None, m: int = 10, random_seed: int = 42, bounds: dict = None,
) -> list:
    """
    Returns a list of M pd.Series (indexed by Time_Step), one complete
    draw each -- the SAME "M complete datasets" shape the existing
    seasonal-bootstrap Multiple Imputation method already produces, so
    downstream consolidation/output code doesn't need to change to
    support this.

    bounds: optional {"lower": float, "upper": float} (Module AP/42) --
    applied to EACH draw's imputed positions before it's returned, so
    a physically-implausible bootstrap sample (e.g. a large negative
    residual pushing an already-near-zero point estimate below zero)
    never reaches the final output. Genuinely observed values are
    never touched, consistent with every other use of bounds here.

    Raises ValueError if a residual distribution couldn't be computed
    at all (e.g. genuinely too little data to backtest this method
    reliably) -- the caller should try a different method rather than
    silently proceeding with a meaningless (empty) uncertainty spread.

    NEVER touches genuinely-observed values -- only positions that are
    actually missing get drawn for, in every draw, matching every
    other imputation method in this pipeline.
    """
    series = df.set_index("Time_Step")[value_column]

    residuals = compute_method_residuals(series, method_name, cycle_length, other_series_dict)
    if len(residuals) == 0:
        raise ValueError(
            f"Could not compute a residual distribution for '{method_name}' on '{value_column}' -- "
            f"not enough currently-observed data to backtest this method reliably."
        )

    point_estimate_df, _ = apply_mod.apply_chosen_imputation(df, value_column, method_name, cycle_length, other_series_dict)
    point_estimate_series = point_estimate_df.set_index("Time_Step")[value_column]

    was_missing = series.isna()
    rng = np.random.RandomState(random_seed)
    draws = []
    for _ in range(m):
        draw_series = series.copy()
        for idx in series.index[was_missing]:
            point_value = point_estimate_series.loc[idx]
            if pd.isna(point_value):
                # The point estimate itself couldn't fill this position
                # (even Module AB/28's own fallback cascade came up
                # empty) -- leave it genuinely missing in every draw,
                # consistent with every other honesty convention here,
                # rather than fabricate a number with no real basis.
                continue
            draw_series.loc[idx] = point_value + rng.choice(residuals)
        if bounds:
            draw_series = bounds_mod.clip_only_imputed_values(draw_series, was_missing, bounds.get("lower"), bounds.get("upper"))
        draws.append(draw_series)

    return draws


def apply_generalized_mi(
    df: pd.DataFrame, value_column: str, method_name: str, cycle_length: int = None,
    other_series_dict: dict = None, m: int = 10, random_seed: int = 42, output_mode: str = "consolidated",
    bounds: dict = None,
) -> dict:
    """
    Generates M draws for `method_name` (any single-value method, via
    generate_generalized_mi_draws() above) and consolidates them using
    Module 29's own consolidate_draws() -- reused as-is, since the
    draws are already in the exact shape that function expects,
    producing the SAME "<col>_was_imputed"/"_imputation_sd"/
    "_imputation_lower_95"/"_imputation_upper_95" columns and/or
    M-datasets list regardless of which method generated the draws.

    bounds: optional {"lower": float, "upper": float} (Module AP/42),
    applied to each draw before consolidation -- so the consolidated
    median/SD/CI and every M-dataset variant all correctly reflect the
    same plausibility constraint, not just the point estimate.
    """
    multi_imp_mod = import_module("29_time_series_multiple_imputation")

    draws = generate_generalized_mi_draws(df, value_column, method_name, cycle_length, other_series_dict, m, random_seed, bounds)
    result = multi_imp_mod.consolidate_draws(df, value_column, draws, output_mode)

    was_missing = df.set_index("Time_Step")[value_column].isna()
    result["n_imputed"] = int(was_missing.sum())
    result["m"] = m
    result["method_used"] = method_name
    return result


def run_mi_workflow(
    df: pd.DataFrame, value_column: str, cycle_length: int = None, other_series_dict: dict = None,
    m: int = 10, output_mode: str = "consolidated", override_method: str = None, random_seed: int = 42,
    bounds: dict = None,
) -> dict:
    """
    THE top-level function app.py calls for Multiple Imputation --
    directly parallel to Module AB/28's run_single_imputation_workflow():
    runs the MI backtest/ranking (Module AO/41), applies the winning
    (or overridden) method's generalized draws to the real missing
    values, and returns everything the UI needs -- the ranking table
    (interval accuracy, MAAPE-based accuracy, combined score), which
    method was used, and the completed dataframe.

    bounds: optional {"lower": float, "upper": float} (Module AP/42),
    passed straight through to apply_generalized_mi().
    """
    mi_backtest_mod = import_module("41_time_series_mi_backtest")

    series = df.set_index("Time_Step")[value_column]
    backtest_results, mask_positions = mi_backtest_mod.run_mi_backtest(
        series, cycle_length=cycle_length, other_series_dict=other_series_dict, m=m, random_seed=random_seed
    )

    recommended = mi_backtest_mod.recommend_mi_method(backtest_results) if not backtest_results.empty else "linear"
    method_used = override_method if override_method else recommended
    was_overridden = override_method is not None and override_method != recommended

    mi_result = apply_generalized_mi(
        df, value_column, method_used, cycle_length, other_series_dict, m, random_seed, output_mode, bounds
    )

    return {
        "backtest_results": backtest_results,
        "recommended_method": recommended,
        "method_used": method_used,
        "was_overridden": was_overridden,
        **mi_result,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: residual computation for a simple, well-understood method (linear)")
    print("=" * 60)
    sales = [266.0, 145.9, 183.1, 119.3, 180.3, 168.5, 231.8, 224.5, 192.8, 122.9, 336.5, 185.9,
             194.3, 149.5, 210.1, 273.3, 191.4, 287.0, 226.0, 303.6, 289.9, 421.6, 264.5, 342.3,
             339.7, 440.4, 315.9, 439.3, 401.3, 437.4, 575.5, 407.6, 682.0, 475.3, 581.3, 646.9]
    df = pd.DataFrame({"Time_Step": range(len(sales)), "Sales": sales})
    df.loc[[8, 13], "Sales"] = None  # genuine gaps, matching the earlier shampoo test

    series = df.set_index("Time_Step")["Sales"]
    residuals = compute_method_residuals(series, "linear", cycle_length=12)
    print("Residuals for 'linear':", residuals.round(1).tolist())
    print("Number of residuals (should match number of masked test points):", len(residuals))

    print("\n" + "=" * 60)
    print("TEST 2: M draws for 'linear' -- point estimates should all be IDENTICAL (deterministic method), spread should vary")
    print("=" * 60)
    draws = generate_generalized_mi_draws(df, "Sales", "linear", cycle_length=12, m=10, random_seed=42)
    for step in [8, 13]:
        values_across_draws = [d.loc[step] for d in draws]
        print(f"Time_Step {step}: {[round(v, 1) for v in values_across_draws]}")
        print(f"  Genuinely varying across draws: {len(set(round(v, 1) for v in values_across_draws)) > 1}")

    print("\n" + "=" * 60)
    print("TEST 3: observed values NEVER touched across any draw")
    print("=" * 60)
    for step in [0, 1, 5, 20]:
        values_across_draws = [d.loc[step] for d in draws]
        print(f"Time_Step {step}: all draws identical to original ({series.loc[step]}): {all(v == series.loc[step] for v in values_across_draws)}")

    print("\n" + "=" * 60)
    print("TEST 4: reproducibility -- same seed gives same draws")
    print("=" * 60)
    draws_again = generate_generalized_mi_draws(df, "Sales", "linear", cycle_length=12, m=10, random_seed=42)
    same = all(np.allclose(d1.dropna(), d2.dropna()) for d1, d2 in zip(draws, draws_again))
    print("Same seed reproduces identical draws:", same)

    print("\n" + "=" * 60)
    print("TEST 5: works for a genuinely different method family too (contextual regression, multivariate)")
    print("=" * 60)
    rainfall_df = pd.read_csv("Rainfall_multivariate_test.csv")
    rainfall_df["Time_Step"] = range(len(rainfall_df))
    other_series = {
        "Specific Humidity": rainfall_df.set_index("Time_Step")["Specific Humidity"],
        "Temperature": rainfall_df.set_index("Time_Step")["Temperature"],
    }
    draws_ctx = generate_generalized_mi_draws(
        rainfall_df, "Precipitation", "contextual regression (random_forest)",
        cycle_length=12, other_series_dict=other_series, m=10, random_seed=42,
    )
    missing_steps = rainfall_df.index[rainfall_df["Precipitation"].isna()].tolist()
    for step in missing_steps:
        values_across_draws = [d.loc[step] for d in draws_ctx]
        print(f"Time_Step {step}: {[round(v, 1) for v in values_across_draws]}")

    print("\n" + "=" * 60)
    print("TEST 6: unknown/invalid method name -- must raise clearly, not silently produce garbage")
    print("=" * 60)
    try:
        generate_generalized_mi_draws(df, "Sales", "not_a_real_method", cycle_length=12)
        print("BUG: did not raise")
    except ValueError as e:
        print("Correctly raised:", e)
