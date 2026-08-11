"""
Module AC: Time Series Multiple Imputation
------------------------------------------------
The one multiple-imputation method scoped for Phase B: a seasonal-
trend residual bootstrap. Reuses the exact same trend+seasonal
decomposition as Module Z's impute_seasonal() (proven, by direct
testing, to actually help on real data), but instead of taking the
single MEDIAN residual at each missing value's phase, draws M
bootstrap SAMPLES from the residual pool -- producing M genuinely
different plausible completions rather than one best guess.

SCOPE DECISION, disclosed rather than silently assumed: bootstraps
from the GLOBAL residual pool (all residuals across every phase),
not phase-specific pools. With ~34 observations across 12 phases,
most phases only have 2-3 residuals of their own -- nowhere near
enough to bootstrap a meaningful distribution FROM. Pooling globally
assumes residual variance is roughly similar across phases, a common
and defensible simplifying assumption for a first version, but a real
one worth knowing about rather than presenting this as more
sophisticated than it is.

OUTPUT CONVENTION matches the cross-sectional pipeline's Multiple
Imputation exactly, for consistency across the whole product:
  - Consolidated: one dataset, imputed value = median of the M draws,
    plus _was_imputed / _imputation_sd / _imputation_lower_95 /
    _imputation_upper_95 (empirical, from the actual draws -- not a
    normal-distribution approximation, same principle as Module 13).
  - M complete datasets: no averaging, m fully separate completed
    series, one plausible draw per missing cell.
  - Both: produces both from the same run.

NEVER touches genuinely-observed values -- only positions that are
actually missing get drawn for, in every draw.
"""

import numpy as np
import pandas as pd


def _fit_trend_and_residuals(series: pd.Series, cycle_length: int) -> tuple:
    """
    Same decomposition as Module Z's impute_seasonal(): OLS trend,
    then residuals from that trend at every OBSERVED point. Returns
    (trend_at, seasonal_component, global_residuals) -- global_residuals
    is the full pool of residuals (all phases combined) that the
    bootstrap draws from.
    """
    observed = series.dropna()
    if len(observed) < 2:
        raise ValueError(
            "Need at least 2 observed values to fit a trend for multiple imputation -- "
            "this series doesn't have enough data."
        )

    slope, intercept = np.polyfit(observed.index.to_numpy(dtype=float), observed.to_numpy(dtype=float), deg=1)
    trend_at = lambda x: slope * x + intercept  # noqa: E731

    residuals = observed - trend_at(observed.index.to_numpy(dtype=float))
    phase_of_index = observed.index.to_numpy(dtype=float) % cycle_length
    seasonal_component = {}
    for phase in range(cycle_length):
        phase_values = residuals[phase_of_index == phase]
        seasonal_component[phase] = phase_values.median() if len(phase_values) > 0 else 0.0

    # Residuals AFTER removing the seasonal component too -- this is
    # the genuine "noise" pool the bootstrap draws from, distinct from
    # the seasonal signal already captured in seasonal_component.
    deseasonalized_residuals = residuals - pd.Series(
        [seasonal_component[int(p)] for p in phase_of_index], index=residuals.index
    )

    return trend_at, seasonal_component, deseasonalized_residuals


def generate_bootstrap_draws(series: pd.Series, cycle_length: int, m: int = 10, random_seed: int = 42) -> list:
    """
    Returns a list of M pd.Series, each a complete version of `series`
    with every genuinely-missing position filled by a DIFFERENT
    bootstrap draw: trend + seasonal component + a resampled residual.
    Observed values are identical and untouched across all M draws --
    only the missing positions actually vary between draws.
    """
    trend_at, seasonal_component, residual_pool = _fit_trend_and_residuals(series, cycle_length)
    missing_positions = series.index[series.isna()].tolist()

    rng = np.random.RandomState(random_seed)
    draws = []
    for draw_i in range(m):
        draw_series = series.copy()
        for idx in missing_positions:
            phase = int(idx) % cycle_length
            sampled_residual = rng.choice(residual_pool.to_numpy())
            draw_series.loc[idx] = trend_at(idx) + seasonal_component[phase] + sampled_residual
        draws.append(draw_series)

    return draws


def consolidate_draws(df: pd.DataFrame, value_column: str, draws: list, output_mode: str = "consolidated") -> dict:
    """
    Given a LIST of M draws (pd.Series, indexed by Time_Step -- the
    same shape both generate_bootstrap_draws() above and Module AN/40's
    generate_generalized_mi_draws() produce), builds the consolidated
    (median + SD + empirical 95% CI) and/or M-datasets output. Shared
    logic, independent of which method actually generated the draws --
    extracted here so the new generalized-method Multiple Imputation
    flow (any single-value method, not just this module's own seasonal-
    trend bootstrap) doesn't need to duplicate this consolidation code.

    Returns a dict with whichever of these output_mode calls for:
      "consolidated_df": one dataset, "<col>_was_imputed" (0/1),
        "<col>_imputation_sd", "<col>_imputation_lower_95",
        "<col>_imputation_upper_95" (empirical 2.5th/97.5th percentile
        of the M draws, not a normal-distribution approximation)
      "m_datasets": list of m complete dataframes, one draw each
    """
    if output_mode not in ("consolidated", "multiple_datasets", "both"):
        raise ValueError("output_mode must be 'consolidated', 'multiple_datasets', or 'both'")

    series = df.set_index("Time_Step")[value_column]
    was_missing = series.isna()
    draws_matrix = pd.concat([d.rename(f"draw_{i}") for i, d in enumerate(draws)], axis=1)

    result = {}

    if output_mode in ("consolidated", "both"):
        consolidated = df.copy()
        median_values = draws_matrix.median(axis=1)
        sd_values = draws_matrix.std(axis=1)
        lower_95 = draws_matrix.quantile(0.025, axis=1)
        upper_95 = draws_matrix.quantile(0.975, axis=1)

        consolidated[value_column] = median_values.reindex(consolidated["Time_Step"]).to_numpy()
        consolidated[f"{value_column}_was_imputed"] = was_missing.reindex(consolidated["Time_Step"]).fillna(False).astype(int).to_numpy()
        consolidated[f"{value_column}_imputation_sd"] = np.where(
            consolidated[f"{value_column}_was_imputed"] == 1,
            sd_values.reindex(consolidated["Time_Step"]).to_numpy(),
            np.nan,
        )
        consolidated[f"{value_column}_imputation_lower_95"] = np.where(
            consolidated[f"{value_column}_was_imputed"] == 1,
            lower_95.reindex(consolidated["Time_Step"]).to_numpy(),
            np.nan,
        )
        consolidated[f"{value_column}_imputation_upper_95"] = np.where(
            consolidated[f"{value_column}_was_imputed"] == 1,
            upper_95.reindex(consolidated["Time_Step"]).to_numpy(),
            np.nan,
        )
        result["consolidated_df"] = consolidated

    if output_mode in ("multiple_datasets", "both"):
        m_datasets = []
        for i, draw_series in enumerate(draws):
            variant = df.copy()
            variant[value_column] = draw_series.reindex(variant["Time_Step"]).to_numpy()
            m_datasets.append(variant)
        result["m_datasets"] = m_datasets

    return result


def multiple_impute_time_series(
    df: pd.DataFrame, value_column: str, cycle_length: int, m: int = 10, random_seed: int = 42, output_mode: str = "consolidated"
) -> dict:
    """
    output_mode: "consolidated", "multiple_datasets", or "both" --
    same three choices as the cross-sectional pipeline's Multiple
    Imputation, for consistency.

    Returns a dict with whichever of these the output_mode calls for:
      "consolidated_df": one dataset, "<col>_was_imputed" (0/1),
        "<col>_imputation_sd", "<col>_imputation_lower_95",
        "<col>_imputation_upper_95" (empirical 2.5th/97.5th percentile
        of the M draws, not a normal-distribution approximation)
      "m_datasets": list of m complete dataframes, one draw each
      "n_imputed": how many genuinely-missing cells were filled
    """
    series = df.set_index("Time_Step")[value_column]
    was_missing = series.isna()
    n_imputed = int(was_missing.sum())

    draws = generate_bootstrap_draws(series, cycle_length, m=m, random_seed=random_seed)
    result = consolidate_draws(df, value_column, draws, output_mode)
    result["n_imputed"] = n_imputed
    result["m"] = m

    return result


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

    print("=" * 60)
    print("TEST 1: M draws are genuinely different from each other (real stochasticity)")
    print("=" * 60)
    series = final_df.set_index("Time_Step")["Sales"]
    draws = generate_bootstrap_draws(series, freq_info["cycle_length"], m=10, random_seed=42)
    missing_steps = series.index[series.isna()].tolist()
    for step in missing_steps:
        values_across_draws = [d.loc[step] for d in draws]
        print(f"Time_Step {step}: {[round(v,1) for v in values_across_draws]}")
        print(f"  all identical? {len(set(round(v,1) for v in values_across_draws)) == 1} (should be False)")

    print("\n" + "=" * 60)
    print("TEST 2: observed values are IDENTICAL and untouched across every draw")
    print("=" * 60)
    observed_steps = series.index[series.notna()].tolist()[:5]
    for step in observed_steps:
        values_across_draws = [d.loc[step] for d in draws]
        print(f"Time_Step {step}: all draws = {values_across_draws[0]}, all identical: {len(set(values_across_draws)) == 1}")

    print("\n" + "=" * 60)
    print("TEST 3: reproducibility -- same seed gives same draws")
    print("=" * 60)
    draws_again = generate_bootstrap_draws(series, freq_info["cycle_length"], m=10, random_seed=42)
    identical = all((d1 == d2).all() or (d1.isna() == d2.isna()).all() for d1, d2 in zip(draws, draws_again))
    same_values = all(np.allclose(d1.dropna(), d2.dropna()) for d1, d2 in zip(draws, draws_again))
    print("Same seed reproduces identical draws:", same_values)

    print("\n" + "=" * 60)
    print("TEST 4: consolidated output -- check structure and plausibility vs known true values")
    print("   (1-09 true=192.8, 2-02 true=149.5)")
    print("=" * 60)
    result = multiple_impute_time_series(final_df, "Sales", freq_info["cycle_length"], m=20, random_seed=42, output_mode="consolidated")
    print("n_imputed:", result["n_imputed"], " m:", result["m"])
    consolidated = result["consolidated_df"]
    imputed_rows = consolidated[consolidated["Sales_was_imputed"] == 1][
        ["Time_Step", "Sales", "Sales_imputation_sd", "Sales_imputation_lower_95", "Sales_imputation_upper_95"]
    ]
    print(imputed_rows.to_string(index=False))
    for _, row in imputed_rows.iterrows():
        true_val = 192.8 if row["Time_Step"] == 9 else 149.5
        within_interval = row["Sales_imputation_lower_95"] <= true_val <= row["Sales_imputation_upper_95"]
        print(f"  Time_Step {int(row['Time_Step'])}: true={true_val} within 95% interval: {within_interval}")

    print("\n" + "=" * 60)
    print("TEST 5: M-datasets output mode")
    print("=" * 60)
    result_m = multiple_impute_time_series(final_df, "Sales", freq_info["cycle_length"], m=5, random_seed=42, output_mode="multiple_datasets")
    print("Number of datasets:", len(result_m["m_datasets"]))
    for i, ds in enumerate(result_m["m_datasets"]):
        vals = ds.loc[ds["Time_Step"].isin(missing_steps), ["Time_Step", "Sales"]]
        print(f"Dataset {i}: {vals['Sales'].tolist()}")

    print("\n" + "=" * 60)
    print("TEST 6: 'both' output mode returns everything")
    print("=" * 60)
    result_both = multiple_impute_time_series(final_df, "Sales", freq_info["cycle_length"], m=5, random_seed=42, output_mode="both")
    print("Has consolidated_df:", "consolidated_df" in result_both)
    print("Has m_datasets:", "m_datasets" in result_both, " count:", len(result_both.get("m_datasets", [])))
