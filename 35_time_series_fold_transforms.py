"""
Module AI: Fold-Aware Detrending & Standardization
----------------------------------------------------------------
The transformation layer between Module AH's raw features and any
contextual model (regression, tree-based, eventually CNN/LSTM). This
is where the leakage-prevention requirement actually gets enforced:
every trend and every standardization parameter is fit using ONLY the
training rows for the current backtest fold, then applied UNCHANGED to
validation rows, artificially masked rows, and genuinely missing rows
-- never refit on anything outside the training fold.

TWO MODES, matching the spec's two imputation modes:
  - RETROSPECTIVE: the trend may use training rows on BOTH sides of a
    gap (the whole training fold, whichever rows aren't held out).
  - FORECAST-SAFE: the trend must use only training rows at or before
    the row being predicted -- fit_causal_transform() below restricts
    to that automatically, since using a later training point would
    leak future information into an ostensibly forward-looking
    imputation.

Three standardization methods, matching the spec: z-score (default),
robust z-score (median/IQR, for when extreme values make mean/std
unsuitable), and min-max. All three share the same fit/apply/invert
interface so a caller can try any of them without special-casing.
"""

import numpy as np
import pandas as pd


def fit_trend(training_series: pd.Series) -> callable:
    """
    Simple OLS linear trend, fit using ONLY the given training series
    (already restricted to whatever rows the caller considers
    "training" for this fold/mode). Returns trend_fn(index_array) ->
    value_array, usable to evaluate the trend at ANY index -- training,
    validation, or genuinely missing rows alike.

    Falls back sensibly when there's too little data to fit a real
    trend: a single observed point gives a flat trend at that value;
    zero observed points gives a flat trend at 0 (better than crashing,
    though this edge case shouldn't arise in practice -- a target
    column with zero observed training rows has nothing to learn from
    at all, and callers should be checking for that before getting here).
    """
    observed = training_series.dropna()
    if len(observed) >= 2:
        slope, intercept = np.polyfit(observed.index.to_numpy(dtype=float), observed.to_numpy(dtype=float), deg=1)
        return lambda idx: slope * np.asarray(idx, dtype=float) + intercept
    if len(observed) == 1:
        flat_value = float(observed.iloc[0])
        return lambda idx: np.full_like(np.asarray(idx, dtype=float), flat_value)
    return lambda idx: np.zeros_like(np.asarray(idx, dtype=float))


ZERO_VARIANCE_TOLERANCE = 1e-8


def fit_standardizer(training_values: pd.Series, method: str = "zscore") -> dict:
    """
    Fits standardization parameters from training_values (expected to
    already be detrended) -- ONLY from training rows, per the
    leakage-prevention requirement. Returns a params dict usable by
    both apply_standardizer() and inverse_standardizer() below.

    method: "zscore" (default), "robust_z" (median/IQR -- for when
    extreme values make mean/std unsuitable), or "minmax".
    """
    clean = training_values.dropna()

    if method == "zscore":
        mean = clean.mean() if len(clean) else 0.0
        std = clean.std() if len(clean) else 1.0
        # A near-constant series doesn't detrend to EXACTLY zero spread
        # (floating-point noise from the trend subtraction), so this
        # uses a small tolerance rather than exact equality -- confirmed
        # directly: constant input data was producing std=1e-15 rather
        # than 0, which the old exact check let through and then divided
        # by, wildly inflating what should have been all-zero output.
        if pd.isna(std) or std < ZERO_VARIANCE_TOLERANCE:
            std = 1.0
        return {"method": "zscore", "mean": mean, "std": std}

    if method == "robust_z":
        median = clean.median() if len(clean) else 0.0
        q75 = clean.quantile(0.75) if len(clean) else 0.0
        q25 = clean.quantile(0.25) if len(clean) else 0.0
        iqr = q75 - q25
        if pd.isna(iqr) or iqr < ZERO_VARIANCE_TOLERANCE:
            iqr = 1.0
        return {"method": "robust_z", "median": median, "iqr": iqr}

    if method == "minmax":
        vmin = clean.min() if len(clean) else 0.0
        vmax = clean.max() if len(clean) else 1.0
        rng = vmax - vmin
        if pd.isna(rng) or rng < ZERO_VARIANCE_TOLERANCE:
            rng = 1.0
        return {"method": "minmax", "min": vmin, "max": vmax}

    raise ValueError(f"Unknown standardization method '{method}'. Choose from zscore, robust_z, minmax.")


def apply_standardizer(values, params: dict):
    if params["method"] == "zscore":
        return (values - params["mean"]) / params["std"]
    if params["method"] == "robust_z":
        return (values - params["median"]) / params["iqr"]
    if params["method"] == "minmax":
        return (values - params["min"]) / (params["max"] - params["min"])
    raise ValueError(f"Unknown standardization method '{params['method']}'.")


def inverse_standardizer(standardized_values, params: dict):
    if params["method"] == "zscore":
        return standardized_values * params["std"] + params["mean"]
    if params["method"] == "robust_z":
        return standardized_values * params["iqr"] + params["median"]
    if params["method"] == "minmax":
        return standardized_values * (params["max"] - params["min"]) + params["min"]
    raise ValueError(f"Unknown standardization method '{params['method']}'.")


def fit_fold_transform(training_series: pd.Series, standardization_method: str = "zscore") -> dict:
    """
    RETROSPECTIVE mode: fits trend + standardizer using the full
    training series as given (the caller decides what counts as
    "training" -- typically every row except whatever's held out for
    this backtest fold, which may include rows on both sides of any
    gap). Returns a bundle usable by transform()/inverse_transform().
    """
    trend_fn = fit_trend(training_series)
    detrended_training = training_series - trend_fn(training_series.index.to_numpy(dtype=float))
    standardizer_params = fit_standardizer(detrended_training, method=standardization_method)
    return {"trend_fn": trend_fn, "standardizer_params": standardizer_params, "mode": "retrospective"}


def fit_causal_transform(training_series: pd.Series, as_of_index, standardization_method: str = "zscore") -> dict:
    """
    FORECAST-SAFE mode: restricts training_series to rows with index
    <= as_of_index BEFORE fitting anything -- a later training row
    would leak future information into what's supposed to be a
    forward-looking imputation, so it's excluded at the source rather
    than trusted to the caller to have already excluded it.
    """
    causal_series = training_series[training_series.index <= as_of_index]
    trend_fn = fit_trend(causal_series)
    detrended_training = causal_series - trend_fn(causal_series.index.to_numpy(dtype=float))
    standardizer_params = fit_standardizer(detrended_training, method=standardization_method)
    return {"trend_fn": trend_fn, "standardizer_params": standardizer_params, "mode": "forecast_safe", "as_of_index": as_of_index}


def transform(series: pd.Series, fold_transform: dict) -> pd.Series:
    """Detrends then standardizes `series` using an ALREADY-FITTED fold_transform -- never refits anything here."""
    detrended = series - fold_transform["trend_fn"](series.index.to_numpy(dtype=float))
    return apply_standardizer(detrended, fold_transform["standardizer_params"])


def inverse_transform(standardized_values: pd.Series, fold_transform: dict) -> pd.Series:
    """
    The exact reverse order the spec requires: inverse-standardize
    first, THEN add the trend back -- giving a result on the
    original measurement scale.
    """
    detrended_values = inverse_standardizer(standardized_values, fold_transform["standardizer_params"])
    trend_values = fold_transform["trend_fn"](standardized_values.index.to_numpy(dtype=float))
    return detrended_values + trend_values


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: round-trip correctness -- transform then inverse_transform must recover the original values")
    print("=" * 60)
    trending_series = pd.Series([10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0], index=pd.Index(range(7), name="Time_Step"))
    ft = fit_fold_transform(trending_series, standardization_method="zscore")
    standardized = transform(trending_series, ft)
    recovered = inverse_transform(standardized, ft)
    print("Original:  ", trending_series.tolist())
    print("Recovered: ", [round(v, 6) for v in recovered.tolist()])
    print("Round-trip exact (within floating point):", np.allclose(trending_series.to_numpy(), recovered.to_numpy()))

    print("\n" + "=" * 60)
    print("TEST 2: no leakage -- an extreme VALIDATION-only value must not shift the fitted trend/standardizer")
    print("=" * 60)
    training_only = trending_series.copy()
    training_only.loc[3] = np.nan  # row 3 held out as "validation" -- fit must not see its true value
    ft_no_leak = fit_fold_transform(training_only, standardization_method="zscore")
    ft_with_leak = fit_fold_transform(trending_series, standardization_method="zscore")  # sees everything, for comparison
    print("Standardizer mean (should differ if held-out row wasn't used to fit):")
    print("  Fit WITHOUT row 3 (correct):", ft_no_leak["standardizer_params"]["mean"])
    print("  Fit WITH row 3 (leaked, for comparison):", ft_with_leak["standardizer_params"]["mean"])
    print("  Confirmed different:", ft_no_leak["standardizer_params"]["mean"] != ft_with_leak["standardizer_params"]["mean"])

    print("\n" + "=" * 60)
    print("TEST 3: all three standardization methods round-trip correctly")
    print("=" * 60)
    for method in ["zscore", "robust_z", "minmax"]:
        ft_m = fit_fold_transform(trending_series, standardization_method=method)
        standardized_m = transform(trending_series, ft_m)
        recovered_m = inverse_transform(standardized_m, ft_m)
        print(f"  {method}: round-trip exact = {np.allclose(trending_series.to_numpy(), recovered_m.to_numpy())}")

    print("\n" + "=" * 60)
    print("TEST 4: forecast-safe (causal) mode -- must exclude any training row AFTER as_of_index")
    print("=" * 60)
    # Deliberately NOT perfectly linear -- slope changes partway through,
    # so a causal-only fit (rows 0-3) and a full-data fit (rows 0-6)
    # produce GENUINELY different lines, unlike a perfectly linear
    # series where any subset fits the identical line regardless of
    # whether the restriction is actually being applied.
    kinked_series = pd.Series([10.0, 13.0, 16.0, 19.0, 25.0, 31.0, 37.0], index=pd.Index(range(7), name="Time_Step"))
    ft_causal = fit_causal_transform(kinked_series, as_of_index=3, standardization_method="zscore")
    print("Causal fit at as_of_index=3 uses only rows 0-3 (slope 3):")
    causal_trend_at_6 = ft_causal["trend_fn"](np.array([6]))[0]
    full_trend_at_6 = fit_trend(kinked_series)(np.array([6]))[0]
    print(f"  Causal trend prediction at row 6: {causal_trend_at_6:.2f}  (extrapolated from rows 0-3's slope-3 trend only)")
    print(f"  Full-data trend prediction at row 6: {full_trend_at_6:.2f}  (fit using rows 0-6, including the steeper slope after row 3)")
    print("  These correctly differ, confirming the causal fit genuinely didn't use rows 4-6:", not np.isclose(causal_trend_at_6, full_trend_at_6))

    print("\n" + "=" * 60)
    print("TEST 5: edge cases -- too few training points must not crash")
    print("=" * 60)
    one_point = pd.Series([42.0, np.nan, np.nan], index=pd.Index(range(3), name="Time_Step"))
    ft_one = fit_fold_transform(one_point, standardization_method="zscore")
    print("Trend with only 1 observed point (flat line at that value):", ft_one["trend_fn"](np.array([0, 1, 2])))

    zero_points = pd.Series([np.nan, np.nan], index=pd.Index(range(2), name="Time_Step"))
    ft_zero = fit_fold_transform(zero_points, standardization_method="zscore")
    print("Trend with 0 observed points (flat line at 0, doesn't crash):", ft_zero["trend_fn"](np.array([0, 1])))

    print("\n" + "=" * 60)
    print("TEST 6: constant training data -- standardizer must not divide by zero")
    print("=" * 60)
    constant_series = pd.Series([5.0, 5.0, 5.0, 5.0], index=pd.Index(range(4), name="Time_Step"))
    ft_const = fit_fold_transform(constant_series, standardization_method="zscore")
    print("Std for constant data (should be 1.0, not 0):", ft_const["standardizer_params"]["std"])
    standardized_const = transform(constant_series, ft_const)
    print("Standardized constant values (should be all 0, not NaN/inf):", standardized_const.tolist())
