"""
Module AA: Imputation Backtesting & Method Recommendation
----------------------------------------------------------------
The scoped-down version of ChatGPT's backtesting proposal: masks a
handful of CURRENTLY-OBSERVED values (never the genuinely-missing
ones), tests every single-imputation method from Module Z against
them, ranks by MAE, and recommends the best -- while showing the full
comparison table so the client can make an informed override, not
just trust a silent pick.

DELIBERATELY LEFT OUT, per the scope discussion: causal/forecasting-
style validation (rolling-origin, expanding windows) -- this feature
fills gaps in already-collected historical data, where values exist
on both sides of every real gap, so retrospective validation is the
only mode that actually matches what this does. Also left out: Kalman/
state-space, Gaussian process, and Bayesian methods as candidates --
those remain out of scope for the same reason they were cut from
Module Y's question list (real risk of overfitting a 36-observation
series, not just extra work).

Also implements the detrend-then-seasonalize fix: the naive seasonal
method from Module Z (raw median of same-phase values) performed WORST
of all four methods on the real test data, specifically because it
ignores trend -- confirmed directly by testing. impute_seasonal_detrended()
here fits a simple linear trend first, computes seasonal medians on the
RESIDUALS, then adds the trend back.
"""

import numpy as np
import pandas as pd

from importlib import import_module

single_imp = import_module("26_time_series_single_imputation")
cross_series_mod = import_module("33_time_series_cross_series_imputation")
contextual_models_mod = import_module("36_time_series_contextual_models")
neural_eligibility_mod = import_module("37_time_series_neural_eligibility")
neural_mod = import_module("38_time_series_neural_imputation")


def impute_seasonal_detrended(series: pd.Series, cycle_length: int) -> tuple:
    """
    Thin alias -- the actual detrend-then-seasonalize implementation
    now lives in Module Z (26) as the canonical impute_seasonal(),
    since it is a strict improvement over the naive version, not a
    separate option a client would ever want the worse version
    instead of. Kept as a named alias here so this module's own
    __main__ test block reads clearly about what it is comparing.
    """
    return single_imp.impute_seasonal(series, cycle_length)


BACKTEST_METHODS = {
    "linear": single_imp.impute_linear,
    "pchip": single_imp.impute_pchip,
    "cubic_spline": single_imp.impute_cubic_spline,
    # seasonal handled separately below -- needs cycle_length, others don't
}


def select_mask_positions(
    series: pd.Series, mask_fraction: float = 0.15, min_mask: int = 3, max_mask: int = 8, random_seed: int = 42
) -> list:
    """
    Picks a handful of CURRENTLY-OBSERVED (non-null) positions to
    temporarily hide for testing. Never touches genuinely-missing
    values -- those aren't eligible at all, they're excluded from the
    candidate pool before any random selection happens.

    Count is derived from mask_fraction of the OBSERVED count, clamped
    to [min_mask, max_mask], and further capped so at least 4
    observed points always remain untouched afterward (Cubic Spline's
    minimum requirement) -- conservative on purpose, per the "don't
    over-mask a short dataset" principle, even though we dropped the
    rest of that framework.
    """
    observed_positions = series.index[series.notna()].tolist()
    n_observed = len(observed_positions)

    n_mask = int(round(n_observed * mask_fraction))
    n_mask = max(min_mask, min(max_mask, n_mask))
    n_mask = min(n_mask, max(0, n_observed - 4))  # always leave >=4 real points behind

    if n_mask <= 0:
        return []

    rng = np.random.RandomState(random_seed)
    return sorted(rng.choice(observed_positions, size=n_mask, replace=False).tolist())


def build_candidate_methods(masked_series: pd.Series, cycle_length: int = None, other_series_dict: dict = None) -> tuple:
    """
    Builds the {method_name: callable} dict used by run_backtest(),
    extracted into its own function so anything that needs to run a
    SPECIFIC named method (e.g. Module AN/40's generalized draw
    generation) uses the EXACT same construction -- guaranteeing the
    set of methods available to rank and the set available to
    actually draw uncertainty from can never drift out of sync.

    Returns (all_methods, ineligible_neural_rows) -- the second is a
    list of "not eligible" placeholder result-rows for any neural
    candidate skipped due to insufficient data, used only by
    run_backtest() itself when building its results table.
    """
    all_methods = dict(BACKTEST_METHODS)
    all_methods["pchip (detrended)"] = single_imp.impute_pchip_detrended
    ineligible_neural_rows = []

    if cycle_length and cycle_length > 1:
        all_methods["seasonal (detrended)"] = lambda s: single_imp.impute_seasonal(s, cycle_length)
        all_methods["linear (deseasonalized)"] = lambda s: single_imp.impute_linear_deseasonalized(s, cycle_length)
        all_methods["pchip (deseasonalized)"] = lambda s: single_imp.impute_pchip_deseasonalized(s, cycle_length)
        all_methods["pchip (detrended + deseasonalized)"] = lambda s: single_imp.impute_pchip_fully_decomposed(s, cycle_length)
        all_methods["cubic_spline (deseasonalized)"] = lambda s: single_imp.impute_cubic_spline_deseasonalized(s, cycle_length)

    if other_series_dict:
        # Correlation is computed on the MASKED series -- the same
        # information the method itself will actually have to work
        # with when applied for real, not the unmasked truth.
        corr_result = cross_series_mod.find_best_correlated_series(masked_series, other_series_dict)
        if corr_result["predictor"] is not None:
            cross_series_predictor_name = corr_result["predictor"]
            predictor_series = other_series_dict[cross_series_predictor_name]
            all_methods[f"cross-series regression (via {cross_series_predictor_name})"] = (
                lambda s, _pred=predictor_series: cross_series_mod.impute_cross_series_regression(s, _pred)
            )

        # Contextual regression/tree-based models -- lag/lead features
        # of the target AND every other series, plus missingness
        # indicators (Modules AH/34, AI/35, AJ/36). Each candidate
        # declines gracefully (reports could_not_fill for everything)
        # rather than crashing if there isn't enough training data,
        # so including all six here is safe even on small datasets --
        # the backtest's own ranking naturally sinks any that can't
        # produce reliable results.
        for model_name in contextual_models_mod.TRADITIONAL_MODEL_CANDIDATES:
            all_methods[f"contextual regression ({model_name})"] = (
                lambda s, _model=model_name: contextual_models_mod.fit_and_predict_contextual_model(
                    s, other_series_dict, _model
                )
            )

        # Neural candidates (1D CNN, Bidirectional LSTM) -- gated by
        # Module AK/37's eligibility check. Checked HERE (not just
        # relying on the neural functions' own internal gate) so an
        # ineligible dataset skips the comparatively expensive attempt
        # entirely, rather than quietly declining deep inside the call.
        n_observed = int(masked_series.notna().sum())
        n_variables = 1 + len(other_series_dict)
        window_length = 7  # matches Module AL/38's default max_lag=3, max_lead=3
        missing_pct = 100.0 * masked_series.isna().sum() / len(masked_series) if len(masked_series) else 100.0
        eligibility = neural_eligibility_mod.check_neural_eligibility(n_observed, n_variables, window_length, missing_pct)
        if eligibility["eligible"]:
            all_methods["neural (1D CNN)"] = lambda s: neural_mod.fit_and_predict_cnn(s, other_series_dict)
            all_methods["neural (Bidirectional LSTM)"] = lambda s: neural_mod.fit_and_predict_bilstm(s, other_series_dict)
        else:
            # Shown explicitly in the results table, per the spec --
            # "mark the neural candidate as 'Not eligible...'" -- rather
            # than just silently omitting them, which would leave the
            # client wondering why CNN/LSTM never even appear.
            for neural_label in ("neural (1D CNN)", "neural (Bidirectional LSTM)"):
                ineligible_neural_rows.append(
                    {
                        "method": neural_label,
                        "n_tested": 0,
                        "n_successful": 0,
                        "n_failed": 0,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "note": eligibility["reason"],
                    }
                )

    return all_methods, ineligible_neural_rows


def run_backtest(series: pd.Series, cycle_length: int = None, mask_positions: list = None, other_series_dict: dict = None) -> tuple:
    """
    Returns (results_df, mask_positions_used). results_df has one row
    per method: n_tested, n_successful, n_failed, mae, rmse -- sorted
    best-to-worst by MAE (ascending -- lower error is better; methods
    with zero successful fills are sorted to the bottom regardless of
    a technically-undefined MAE).

    other_series_dict: optional {column_name: pd.Series}, other value
    columns from the SAME multivariate dataset, indexed the same way
    as `series`. When provided, Module AG/33's cross-series regression
    is added as an ADDITIONAL candidate -- using whichever other
    series correlates most strongly with THIS one -- competing
    on evidence against every univariate method, never assumed to be
    better just because it exists. Omitted entirely (None) for
    genuinely univariate data, where it wouldn't apply at all.

    The series passed in is NEVER mutated -- masking happens on a
    local copy per method, and the original values are exactly what
    "true_values" is compared against, so nothing about the caller's
    data is altered by running this.
    """
    if mask_positions is None:
        mask_positions = select_mask_positions(series, random_seed=42 if cycle_length else 42)
    if not mask_positions:
        return pd.DataFrame(), []

    true_values = series.loc[mask_positions].copy()
    masked_series = series.copy()
    masked_series.loc[mask_positions] = np.nan

    all_methods, ineligible_neural_rows = build_candidate_methods(masked_series, cycle_length, other_series_dict)
    rows = []
    for row in ineligible_neural_rows:
        row = dict(row)
        row["n_tested"] = len(mask_positions)
        row["n_failed"] = len(mask_positions)
        rows.append(row)

    for method_name, func in all_methods.items():
        imputed, was_imputed, could_not_fill = func(masked_series)
        tested_positions = [p for p in mask_positions if was_imputed.get(p, False)]
        failed_positions = [p for p in mask_positions if not was_imputed.get(p, False)]

        if tested_positions:
            errors = (imputed.loc[tested_positions] - true_values.loc[tested_positions]).abs()
            mae = errors.mean()
            rmse = np.sqrt((errors ** 2).mean())
        else:
            mae = np.nan
            rmse = np.nan

        rows.append(
            {
                "method": method_name,
                "n_tested": len(mask_positions),
                "n_successful": len(tested_positions),
                "n_failed": len(failed_positions),
                "mae": mae,
                "rmse": rmse,
                "note": "",
            }
        )

    results_df = pd.DataFrame(rows)
    # Best-to-worst: successful methods sorted by ascending MAE first,
    # any method that failed on every test point pushed to the bottom.
    results_df["_sort_key"] = results_df["mae"].fillna(np.inf)
    results_df = results_df.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    return results_df, mask_positions


def recommend_method(results_df: pd.DataFrame) -> str:
    """The rank-1 method's name -- just the top row, since results_df is already sorted best-first."""
    if results_df.empty or results_df.iloc[0]["n_successful"] == 0:
        return None
    return results_df.iloc[0]["method"]


if __name__ == "__main__":
    import importlib.util
    import sys

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    split_mod = load("split_mod", "16_time_series_split_validate.py")
    sort_mod = load("sort_mod", "17_time_series_sort.py")
    gaps_mod = load("gaps_mod", "19_time_series_gaps_and_timestep.py")
    val_mod = load("val_mod", "21_time_series_value_validation.py")

    component_meanings = [
        "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
        "Month",
    ]

    print("=" * 60)
    print("TEST 1: detrended seasonal vs naive seasonal, on the KNOWN true values")
    print("   (1-09 true=192.8, 2-02 true=149.5)")
    print("=" * 60)
    df = pd.read_csv("shampoo_missing_reconstructed.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(df, "Month", "-", component_meanings)
    sorted_df, _ = sort_mod.sort_time_series(split_df, split_report["component_columns"])
    resolved_df = sorted_df.copy()
    resolved_df["_period_duplicate_count"] = 1
    complete_df, _ = gaps_mod.build_complete_sequence(resolved_df, split_report["component_columns"], "Month", ["Sales"])
    freq_info = gaps_mod.detect_original_frequency(split_report["component_columns"])
    after_ph, _ = val_mod.apply_confirmed_placeholders(complete_df, {})
    resolved_vals, excl, manual, _ = val_mod.apply_non_numeric_resolutions(after_ph, {})
    final_df, _ = val_mod.finalize_numeric_conversion(resolved_vals, ["Sales"], excl, manual)

    series = final_df.set_index("Time_Step")["Sales"]
    imputed_naive, _, _ = single_imp.impute_seasonal_naive(series, freq_info["cycle_length"])
    imputed_detrended, _, _ = impute_seasonal_detrended(series, freq_info["cycle_length"])

    missing_steps = series.index[series.isna()].tolist()
    for step in missing_steps:
        true_val = 192.8 if step == missing_steps[0] else 149.5
        naive_err = abs(imputed_naive.loc[step] - true_val)
        detrend_err = abs(imputed_detrended.loc[step] - true_val)
        print(f"Time_Step {step}: naive={imputed_naive.loc[step]:.1f} (err {naive_err:.1f})   "
              f"detrended={imputed_detrended.loc[step]:.1f} (err {detrend_err:.1f})   "
              f"{'IMPROVED' if detrend_err < naive_err else 'worse'}")

    print("\n" + "=" * 60)
    print("TEST 2: full backtest -- mask observed values, test all methods, rank")
    print("=" * 60)
    results_df, mask_positions = run_backtest(series, cycle_length=freq_info["cycle_length"])
    print("Masked positions (currently-observed values temporarily hidden):", mask_positions)
    print()
    print(results_df.to_string(index=False))
    print()
    print("Recommended method:", recommend_method(results_df))

    print("\n" + "=" * 60)
    print("TEST 3: confirm the original series is genuinely untouched after backtesting")
    print("=" * 60)
    print("Original non-null count before:", series.notna().sum())
    _, _ = run_backtest(series, cycle_length=freq_info["cycle_length"])
    print("Original non-null count after running backtest again:", series.notna().sum(), "(must be unchanged)")
    print("Any of the masked positions permanently altered?", series.loc[mask_positions].isna().any(), "(should be False)")

    print("\n" + "=" * 60)
    print("TEST 4: confirm the two GENUINELY missing values were never eligible for masking")
    print("=" * 60)
    print("Real missing Time_Steps:", missing_steps)
    print("Any real-missing step present in mask_positions?", any(s in mask_positions for s in missing_steps), "(should be False)")
