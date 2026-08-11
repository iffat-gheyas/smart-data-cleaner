"""
Module AB: Apply Chosen Imputation
----------------------------------------
The connector between Module AA's backtest recommendation (or a
client override) and actually filling the REAL missing values --
deliberately kept separate from backtesting itself, since backtesting
operates on artificially-masked COPIES of observed data while this
operates on the genuinely missing cells, and conflating the two paths
would risk exactly the kind of accidental data-mutation bug the
backtesting module goes out of its way to avoid.

Also provides run_single_imputation_workflow(), the single top-level
function app.py's UI will actually call: runs the backtest, applies
the winning (or overridden) method to the real gaps, and returns
everything the UI needs to display -- the ranking table, which method
was used, and the completed dataframe.
"""

import re

import numpy as np
import pandas as pd

from importlib import import_module

single_imp = import_module("26_time_series_single_imputation")
backtest_mod = import_module("27_time_series_backtesting")
cross_series_mod = import_module("33_time_series_cross_series_imputation")
contextual_models_mod = import_module("36_time_series_contextual_models")
neural_mod = import_module("38_time_series_neural_imputation")
bounds_mod = import_module("42_time_series_plausibility_bounds")

# Maps a backtest results_df "method" label back to apply_single_imputation's
# expected method keyword -- the backtest table shows "seasonal (detrended)"
# for clarity, but Module Z's apply_single_imputation just expects "seasonal".
METHOD_LABEL_TO_KEY = {
    "linear": "linear",
    "pchip": "pchip",
    "cubic_spline": "cubic_spline",
    "seasonal (detrended)": "seasonal",
    "pchip (detrended)": "pchip_detrended",
    "linear (deseasonalized)": "linear_deseasonalized",
    "pchip (deseasonalized)": "pchip_deseasonalized",
    "pchip (detrended + deseasonalized)": "pchip_fully_decomposed",
    "cubic_spline (deseasonalized)": "cubic_spline_deseasonalized",
}

CROSS_SERIES_LABEL_PATTERN = re.compile(r"^cross-series regression \(via (.+)\)$")
CONTEXTUAL_MODEL_LABEL_PATTERN = re.compile(r"^contextual regression \((.+)\)$")
NEURAL_LABEL_TO_ARCHITECTURE = {
    "neural (1D CNN)": "cnn",
    "neural (Bidirectional LSTM)": "lstm",
}


def apply_chosen_imputation(
    df: pd.DataFrame, value_column: str, chosen_method_label: str, cycle_length: int = None, other_series_dict: dict = None
) -> tuple:
    """
    chosen_method_label: one of the labels shown in the backtest
    results table (e.g. "seasonal (detrended)", or the dynamically-
    named "cross-series regression (via <predictor column>)"), or a
    raw method key ("linear") -- all are accepted so this works
    whether the caller passes through the exact table value or a
    hand-picked override.

    other_series_dict: required ONLY when chosen_method_label is a
    cross-series method -- {column_name: pd.Series indexed by
    Time_Step}, the other value columns from the same multivariate
    dataset.

    Returns (df_with_imputation, report) -- identical shape whether
    the method came from Module Z (26) or the cross-series path here.
    """
    cross_series_match = CROSS_SERIES_LABEL_PATTERN.match(chosen_method_label)
    if cross_series_match:
        predictor_name = cross_series_match.group(1)
        if not other_series_dict or predictor_name not in other_series_dict:
            raise ValueError(
                f"'{chosen_method_label}' needs '{predictor_name}' as a predictor series, but it "
                f"wasn't provided to apply_chosen_imputation()."
            )
        df = df.copy()
        series = df.set_index("Time_Step")[value_column]
        predictor_series = other_series_dict[predictor_name]
        imputed, was_imputed, could_not_fill = cross_series_mod.impute_cross_series_regression(series, predictor_series)

        imputed_aligned = imputed.reindex(df["Time_Step"]).to_numpy()
        was_imputed_aligned = was_imputed.reindex(df["Time_Step"]).fillna(False).to_numpy()
        could_not_fill_aligned = could_not_fill.reindex(df["Time_Step"]).fillna(False).to_numpy()

        df[value_column] = imputed_aligned
        df[f"{value_column}_was_imputed"] = was_imputed_aligned.astype(int)
        df[f"{value_column}_imputation_method"] = np.where(was_imputed_aligned, chosen_method_label, None)

        report = {
            "method": chosen_method_label,
            "n_imputed": int(was_imputed_aligned.sum()),
            "n_could_not_fill": int(could_not_fill_aligned.sum()),
            "time_steps_could_not_fill": df.loc[could_not_fill_aligned, "Time_Step"].tolist(),
        }
        return df, report

    contextual_match = CONTEXTUAL_MODEL_LABEL_PATTERN.match(chosen_method_label)
    if contextual_match:
        model_name = contextual_match.group(1)
        if not other_series_dict:
            raise ValueError(
                f"'{chosen_method_label}' needs the other value columns' series, but none were "
                f"provided to apply_chosen_imputation()."
            )
        df = df.copy()
        series = df.set_index("Time_Step")[value_column]
        imputed, was_imputed, could_not_fill = contextual_models_mod.fit_and_predict_contextual_model(
            series, other_series_dict, model_name
        )

        imputed_aligned = imputed.reindex(df["Time_Step"]).to_numpy()
        was_imputed_aligned = was_imputed.reindex(df["Time_Step"]).fillna(False).to_numpy()
        could_not_fill_aligned = could_not_fill.reindex(df["Time_Step"]).fillna(False).to_numpy()

        df[value_column] = imputed_aligned
        df[f"{value_column}_was_imputed"] = was_imputed_aligned.astype(int)
        df[f"{value_column}_imputation_method"] = np.where(was_imputed_aligned, chosen_method_label, None)

        report = {
            "method": chosen_method_label,
            "n_imputed": int(was_imputed_aligned.sum()),
            "n_could_not_fill": int(could_not_fill_aligned.sum()),
            "time_steps_could_not_fill": df.loc[could_not_fill_aligned, "Time_Step"].tolist(),
        }
        return df, report

    if chosen_method_label in NEURAL_LABEL_TO_ARCHITECTURE:
        if not other_series_dict:
            raise ValueError(
                f"'{chosen_method_label}' needs the other value columns' series, but none were "
                f"provided to apply_chosen_imputation()."
            )
        df = df.copy()
        series = df.set_index("Time_Step")[value_column]
        architecture = NEURAL_LABEL_TO_ARCHITECTURE[chosen_method_label]
        fit_fn = neural_mod.fit_and_predict_cnn if architecture == "cnn" else neural_mod.fit_and_predict_bilstm
        imputed, was_imputed, could_not_fill = fit_fn(series, other_series_dict)

        imputed_aligned = imputed.reindex(df["Time_Step"]).to_numpy()
        was_imputed_aligned = was_imputed.reindex(df["Time_Step"]).fillna(False).to_numpy()
        could_not_fill_aligned = could_not_fill.reindex(df["Time_Step"]).fillna(False).to_numpy()

        df[value_column] = imputed_aligned
        df[f"{value_column}_was_imputed"] = was_imputed_aligned.astype(int)
        df[f"{value_column}_imputation_method"] = np.where(was_imputed_aligned, chosen_method_label, None)

        report = {
            "method": chosen_method_label,
            "n_imputed": int(was_imputed_aligned.sum()),
            "n_could_not_fill": int(could_not_fill_aligned.sum()),
            "time_steps_could_not_fill": df.loc[could_not_fill_aligned, "Time_Step"].tolist(),
        }
        return df, report

    method_key = METHOD_LABEL_TO_KEY.get(chosen_method_label, chosen_method_label)
    return single_imp.apply_single_imputation(df, value_column, method_key, cycle_length)


def run_single_imputation_workflow(
    df: pd.DataFrame, value_column: str, cycle_length: int = None, override_method: str = None,
    other_series_dict: dict = None, bounds: dict = None,
) -> dict:
    """
    THE top-level function app.py's UI will call for single imputation.

    1. Runs the backtest (Module AA) on whatever's currently observed.
    2. Determines which method to actually apply: the client's
       override if given, otherwise the backtest's own top-ranked
       recommendation.
    3. Applies that method to the REAL missing values (this module).

    other_series_dict: optional {column_name: pd.Series indexed by
    Time_Step} -- the OTHER value columns from the same multivariate
    dataset. When provided, cross-series regression (Module AG/33) is
    included as a candidate in both the backtest and the actual
    application step, competing on evidence against every univariate
    method. Omitted for genuinely univariate data.

    bounds: optional {"lower": float, "upper": float} (either key may
    be omitted or None) -- Module AP/42's plausibility bounds, applied
    to every IMPUTED value (primary method's fills AND any fallback
    cascade fills) in ONE pass at the very end, never touching
    genuinely observed values even if one happens to fall outside the
    confirmed range.

    Returns a dict with everything the UI needs to show:
      "backtest_results": the full ranking table (client sees ALL
        methods' performance, not just the winner -- per the explicit
        request that led to building this module)
      "recommended_method": the backtest's own top pick
      "method_used": what was ACTUALLY applied (may differ from the
        recommendation if the client overrode it)
      "was_overridden": bool
      "df": the dataframe with the real gaps filled
      "imputation_report": Module Z's own per-column report (n_imputed,
        any that still couldn't be filled, etc.)

    If NOTHING is currently observed enough to backtest at all (an
    edge case, not expected in practice but worth not crashing on),
    backtest_results comes back empty and method_used falls back to
    "linear" as the most broadly-applicable default.
    """
    series = df.set_index("Time_Step")[value_column]
    backtest_results, mask_positions = backtest_mod.run_backtest(series, cycle_length=cycle_length, other_series_dict=other_series_dict)

    recommended = backtest_mod.recommend_method(backtest_results) if not backtest_results.empty else "linear"
    method_used = override_method if override_method else recommended
    was_overridden = override_method is not None and override_method != recommended

    result_df, imputation_report = apply_chosen_imputation(df, value_column, method_used, cycle_length, other_series_dict)

    # --- Cascading fallback: even the BEST method can genuinely fail
    # to fill some specific rows (e.g. cross-series regression when its
    # chosen predictor is ALSO missing at that exact row) -- confirmed
    # directly as a real, reachable case, not hypothetical. Rather than
    # leave those rows silently blank in the final output, try each
    # remaining backtest candidate in ranked order for just those rows,
    # and fall back to forward/backward-fill as an absolute last resort
    # if every candidate fails. Every fallback fill is clearly labeled
    # in "<col>_imputation_method" with which method actually produced
    # it -- never silently attributed to the original chosen method.
    if imputation_report["n_could_not_fill"] > 0:
        remaining = list(imputation_report["time_steps_could_not_fill"])
        fallback_chain = (
            [m for m in backtest_results["method"].tolist() if m != method_used] if not backtest_results.empty else []
        )
        fallback_detail = {}

        for fallback_method in fallback_chain:
            if not remaining:
                break
            try:
                fallback_df, fallback_report = apply_chosen_imputation(
                    result_df, value_column, fallback_method, cycle_length, other_series_dict
                )
            except Exception:
                continue  # this candidate couldn't even run here -- move to the next one, don't propagate the error

            fallback_indexed = fallback_df.set_index("Time_Step")
            newly_filled = [t for t in remaining if t not in fallback_report["time_steps_could_not_fill"]]
            for t in newly_filled:
                result_df.loc[result_df["Time_Step"] == t, value_column] = fallback_indexed.loc[t, value_column]
                result_df.loc[result_df["Time_Step"] == t, f"{value_column}_was_imputed"] = 1
                result_df.loc[result_df["Time_Step"] == t, f"{value_column}_imputation_method"] = f"{fallback_method} (fallback)"
                fallback_detail[t] = fallback_method
            remaining = [t for t in remaining if t not in newly_filled]

        if remaining:
            # Absolute last resort -- guaranteed to succeed as long as
            # ANY value exists anywhere in the column, since it doesn't
            # need a fitted model or a predictor, just neighboring
            # values in the same column.
            series_now = result_df.set_index("Time_Step")[value_column]
            last_resort_filled = series_now.ffill().bfill()
            for t in remaining:
                result_df.loc[result_df["Time_Step"] == t, value_column] = last_resort_filled.loc[t]
                result_df.loc[result_df["Time_Step"] == t, f"{value_column}_was_imputed"] = 1
                result_df.loc[result_df["Time_Step"] == t, f"{value_column}_imputation_method"] = "forward/backward-fill (last resort)"
                fallback_detail[t] = "forward/backward-fill (last resort)"
            remaining = []

        imputation_report["n_could_not_fill"] = len(remaining)
        imputation_report["time_steps_could_not_fill"] = remaining
        imputation_report["fallback_methods_used"] = fallback_detail

    if bounds:
        was_imputed_mask = result_df[f"{value_column}_was_imputed"] == 1
        result_df[value_column] = bounds_mod.clip_only_imputed_values(
            result_df[value_column], was_imputed_mask, bounds.get("lower"), bounds.get("upper")
        )

    return {
        "backtest_results": backtest_results,
        "backtest_mask_positions": mask_positions,
        "recommended_method": recommended,
        "method_used": method_used,
        "was_overridden": was_overridden,
        "df": result_df,
        "imputation_report": imputation_report,
    }


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
    print("TEST 1: end-to-end workflow -- backtest recommends, then fills the REAL gaps")
    print("=" * 60)
    result = run_single_imputation_workflow(final_df, "Sales", cycle_length=freq_info["cycle_length"])
    print("Backtest results:")
    print(result["backtest_results"].to_string(index=False))
    print()
    print("Recommended:", result["recommended_method"])
    print("Method used:", result["method_used"], " (overridden:", result["was_overridden"], ")")
    print()
    print("Imputation report:", result["imputation_report"])
    print()
    imputed_df = result["df"]
    real_gaps = imputed_df[imputed_df["Sales_was_imputed"] == 1][["Time_Step", "Sales", "Sales_imputation_method"]]
    print("The two REAL gaps, now filled:")
    print(real_gaps.to_string(index=False))
    for _, row in real_gaps.iterrows():
        true_val = 192.8 if row["Time_Step"] == 9 else 149.5
        print(f"  Time_Step {int(row['Time_Step'])}: filled={row['Sales']:.1f}  true={true_val}  abs_error={abs(row['Sales']-true_val):.1f}")

    print("\n" + "=" * 60)
    print("TEST 2: client override -- force 'linear' even though it wasn't the recommendation")
    print("=" * 60)
    result_override = run_single_imputation_workflow(final_df, "Sales", cycle_length=freq_info["cycle_length"], override_method="linear")
    print("Recommended:", result_override["recommended_method"], " Method used:", result_override["method_used"], " Overridden:", result_override["was_overridden"])
    real_gaps2 = result_override["df"][result_override["df"]["Sales_was_imputed"] == 1][["Time_Step", "Sales", "Sales_imputation_method"]]
    print(real_gaps2.to_string(index=False))
