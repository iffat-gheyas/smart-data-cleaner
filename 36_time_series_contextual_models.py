"""
Module AJ: Contextual Regression & Tree-Based Imputation Models
--------------------------------------------------------------------------
Ties Module AH (lag/lead feature engineering) and Module AI (fold-aware
detrending/standardization) together into actual working imputation
candidates -- the traditional-model half of the new contextual family
(Elastic Net, Ridge, Lasso, Random Forest, Extra Trees, Gradient
Boosting). CNN/LSTM come later, once eligibility gating exists.

PIPELINE ORDER (per the spec, Section 11's leakage-prevention steps,
scoped to RETROSPECTIVE mode for now -- see note below):
  1. Fit a trend + standardizer for the TARGET and every OTHER series,
     each using that series' OWN observed values (Module AI).
  2. Transform (detrend + standardize) target and other series.
  3. Build lag/lead features FROM THE TRANSFORMED series (Module AH) --
     equivalent to detrending each lag/lead column individually, just
     far simpler, since a lag of a detrended series is exactly the
     detrended value at that shifted time.
  4. Zero-fill missing feature values for model input (Module AH),
     keeping the missingness indicators as separate signal.
  5. Fit the chosen model using rows where the target is genuinely
     observed (the "training" rows).
  6. Predict at rows where the target is missing.
  7. Inverse-transform the prediction back to the target's original
     scale (Module AI) -- inverse-standardize, then add the trend back.

SCOPE NOTE: this module implements RETROSPECTIVE mode only. Forecast-
safe (causal) mode needs a per-row-refit trend/standardizer (Module
AI's fit_causal_transform, called once per target row using only data
up to that point) -- a real, larger extension, deliberately deferred
until there's an actual forward-looking-forecast use case that needs
it, rather than built speculatively now. This product's current use
(cleaning a historical file for analysis) is retrospective by nature.

ELIGIBILITY: a lightweight row-count guard (not the full neural-
network eligibility gate that comes later, but the same spirit) --
declines gracefully rather than fitting a model on too little data to
produce a trustworthy result.
"""

from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

contextual_features_mod = import_module("34_time_series_contextual_features")
fold_transform_mod = import_module("35_time_series_fold_transforms")

MIN_TRAINING_ROWS = 10  # below this, decline rather than fit an unreliable model

# CV-tuned linear models rather than a fixed alpha=1.0 -- confirmed
# directly (Test 1 below) that a fixed alpha=1.0 was too aggressive for
# this data, noticeably underperforming Ridge/Random Forest. Each of
# these picks its own regularization strength via cross-validation
# STRICTLY within the training fold already being used to fit it --
# never touching the actual held-out backtest validation rows -- so
# this is more principled than guessing one fixed alpha across every
# dataset this product will ever see. cv=3 (not sklearn's default 5)
# to stay usable on the small training sets typical of this product.
TRADITIONAL_MODEL_CANDIDATES = {
    "elastic_net": lambda: ElasticNetCV(l1_ratio=0.5, cv=3, max_iter=5000, random_state=42),
    "ridge": lambda: RidgeCV(alphas=np.logspace(-3, 3, 13), cv=3),
    "lasso": lambda: LassoCV(cv=3, max_iter=5000, random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=200, max_depth=5, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
}


def fit_and_predict_contextual_model(
    target_series: pd.Series,
    other_series_dict: dict,
    model_name: str,
    max_lag: int = 3,
    max_lead: int = 3,
    other_max_lag: int = 0,
    other_max_lead: int = 0,
    standardization_method: str = "zscore",
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> tuple:
    """
    target_series: the column to impute, indexed by Time_Step, with
    genuine gaps (or artificially masked positions, when called from
    the backtest) as NaN.
    other_series_dict: {column_name: pd.Series}, the other value
    columns from the same multivariate dataset, same index.
    model_name: a key from TRADITIONAL_MODEL_CANDIDATES.

    Returns (imputed_series, was_imputed, could_not_fill) -- the same
    three-part interface as every other imputation method in this
    pipeline, so this drops into the existing backtest/apply machinery
    without special-casing beyond knowing it needs other_series_dict
    (same as the cross-series method already does).

    Declines gracefully (could_not_fill = every missing row) rather
    than fitting an unreliable model when fewer than min_training_rows
    genuinely-observed target rows exist.
    """
    if model_name not in TRADITIONAL_MODEL_CANDIDATES:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(TRADITIONAL_MODEL_CANDIDATES.keys())}.")

    target_col = "__target__"
    other_cols = list(other_series_dict.keys())
    combined = pd.DataFrame({target_col: target_series, **other_series_dict})

    training_mask = target_series.notna()
    if training_mask.sum() < min_training_rows:
        was_imputed = pd.Series(False, index=target_series.index)
        could_not_fill = target_series.isna()
        return target_series.copy(), was_imputed, could_not_fill

    # --- Step 1-2: fit + apply a trend/standardizer PER SERIES, each
    # using that series' own observed values ---
    fold_transforms = {}
    transformed = pd.DataFrame(index=combined.index)
    for col in combined.columns:
        ft = fold_transform_mod.fit_fold_transform(combined[col], standardization_method)
        fold_transforms[col] = ft
        transformed[col] = fold_transform_mod.transform(combined[col], ft)

    # --- Step 3: lag/lead features built from the TRANSFORMED series ---
    features, feature_names, missing_names = contextual_features_mod.build_contextual_features(
        transformed, target_col, other_cols, max_lag=max_lag, max_lead=max_lead,
        other_max_lag=other_max_lag, other_max_lead=other_max_lead,
    )
    all_feature_cols = feature_names + missing_names

    # --- Step 4: zero-fill for model input only ---
    zero_filled = contextual_features_mod.zero_fill_for_model_input(features, feature_names)

    # --- Step 5: fit on rows where the target is genuinely observed ---
    y_transformed = transformed[target_col]
    X_train = zero_filled.loc[training_mask, all_feature_cols]
    y_train = y_transformed.loc[training_mask]

    model = TRADITIONAL_MODEL_CANDIDATES[model_name]()
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # --- Step 6: predict at rows where the target is missing ---
    predict_mask = target_series.isna()
    imputed = target_series.copy()
    was_imputed = pd.Series(False, index=target_series.index)
    could_not_fill = pd.Series(False, index=target_series.index)

    if predict_mask.sum() > 0:
        X_predict = zero_filled.loc[predict_mask, all_feature_cols]
        y_pred_transformed = model.predict(X_predict.to_numpy())
        y_pred_transformed_series = pd.Series(y_pred_transformed, index=X_predict.index)

        # --- Step 7: inverse-transform back to the target's original scale ---
        y_pred_original = fold_transform_mod.inverse_transform(y_pred_transformed_series, fold_transforms[target_col])

        imputed.loc[predict_mask] = y_pred_original
        was_imputed.loc[predict_mask] = True

    return imputed, was_imputed, could_not_fill


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: genuinely correlated multivariate data -- Elastic Net should produce accurate estimates")
    print("=" * 60)
    df = pd.read_csv("Rainfall_multivariate_test.csv")
    df["Time_Step"] = range(len(df))
    df = df.set_index("Time_Step")

    other_series = {"Specific Humidity": df["Specific Humidity"], "Temperature": df["Temperature"]}
    imputed, was_imputed, could_not_fill = fit_and_predict_contextual_model(
        df["Precipitation"], other_series, "elastic_net", max_lag=2, max_lead=2
    )
    missing_steps = df.index[df["Precipitation"].isna()].tolist()
    for step in missing_steps:
        true_expectation = 50 + 3.5 * df.loc[step, "Specific Humidity"]
        print(f"Time_Step {step}: imputed={imputed.loc[step]:.1f}  (true Humidity-based expectation ≈ {true_expectation:.1f})")
    print("Output is on the ORIGINAL scale (not standardized/detrended residuals):", imputed.loc[missing_steps].between(0, 300).all())

    print("\n" + "=" * 60)
    print("TEST 2: all six model candidates run without crashing on the same data")
    print("=" * 60)
    for model_name in TRADITIONAL_MODEL_CANDIDATES:
        imputed_m, was_imputed_m, could_not_fill_m = fit_and_predict_contextual_model(
            df["Precipitation"], other_series, model_name, max_lag=2, max_lead=2
        )
        print(f"  {model_name}: imputed values = {imputed_m.loc[missing_steps].round(1).tolist()}, could_not_fill = {could_not_fill_m.sum()}")

    print("\n" + "=" * 60)
    print("TEST 3: too few training rows -- must decline gracefully, not crash or overfit garbage")
    print("=" * 60)
    tiny_series = pd.Series([1.0, 2.0, np.nan, 4.0, np.nan], index=pd.Index(range(5), name="Time_Step"))
    tiny_other = {"Y": pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=pd.Index(range(5), name="Time_Step"))}
    imputed_tiny, was_imputed_tiny, could_not_fill_tiny = fit_and_predict_contextual_model(
        tiny_series, tiny_other, "ridge", min_training_rows=10
    )
    print("Declined gracefully -- could_not_fill count (should be 2, matching the genuine gaps):", could_not_fill_tiny.sum())
    print("was_imputed (should be all False):", was_imputed_tiny.tolist())

    print("\n" + "=" * 60)
    print("TEST 4: genuinely unrelated predictor -- must not crash, prediction just won't be very accurate")
    print("=" * 60)
    unrelated_series = pd.Series(
        list(np.random.RandomState(3).normal(50, 10, 30)), index=pd.Index(range(30), name="Time_Step")
    )
    unrelated_other = {"Noise": pd.Series(np.random.RandomState(4).normal(0, 1, 30), index=pd.Index(range(30), name="Time_Step"))}
    unrelated_series_with_gaps = unrelated_series.copy()
    unrelated_series_with_gaps.loc[[5, 15, 25]] = np.nan
    imputed_u, was_imputed_u, could_not_fill_u = fit_and_predict_contextual_model(
        unrelated_series_with_gaps, unrelated_other, "elastic_net", max_lag=2, max_lead=2
    )
    print("Ran without crashing, n_imputed:", was_imputed_u.sum(), " n_could_not_fill:", could_not_fill_u.sum())

    print("\n" + "=" * 60)
    print("TEST 5: unknown model name must raise clearly")
    print("=" * 60)
    try:
        fit_and_predict_contextual_model(df["Precipitation"], other_series, "not_a_real_model")
        print("BUG: did not raise")
    except ValueError as e:
        print("Correctly raised:", e)
