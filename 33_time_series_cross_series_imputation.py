"""
Module AG: Cross-Series Imputation
------------------------------------------
The genuinely new capability multivariate data enables: every existing
method (Module Z/26's linear/PCHIP/cubic-spline/seasonal, and their
deseasonalized variants) only looks at the TARGET series' own past/
future values. None of them can use a correlated OTHER series' value
at the same timestamp -- which is exactly the extra information a
multivariate dataset provides that a univariate one doesn't.

APPROACH: simple OLS linear regression of the target series on
whichever OTHER series is most strongly correlated with it (computed
only from rows where BOTH are genuinely observed) -- not a full
multiple regression against every other series at once. This is a
deliberate scope choice: with potentially few overlapping observed
rows and several candidate predictor series, a single best-correlated
predictor is far more robust against overfitting than throwing every
other series in at once, and it's slotted into the SAME backtest
framework as every other method (Module AA/27) specifically so the
evidence decides whether it's actually better for a given dataset --
never assumed. If the correlation is weak or spurious, the backtest's
own MAE ranking will naturally rank this method poorly, without
needing a separate hard-coded correlation threshold.

Works on plain pd.Series indexed by Time_Step, matching the exact
convention every other imputation function in this pipeline uses.

A row can only be filled this way if the CHOSEN predictor series is
itself observed at that same row -- if it's also missing there, this
method genuinely can't help for that specific point, and reports it
as could_not_fill (falls back to the backtest's other candidates),
exactly the same honesty convention as every other imputation method
here.
"""

import numpy as np
import pandas as pd


def find_best_correlated_series(target_series: pd.Series, other_series_dict: dict, min_overlap: int = 4) -> dict:
    """
    other_series_dict: {column_name: pd.Series}, each indexed the same
    way as target_series (by Time_Step).

    Returns {"predictor": col_name, "correlation": float} for the
    single strongest (by absolute value) correlation among candidates
    with at least min_overlap genuinely-overlapping observed rows, or
    {"predictor": None, "correlation": None} if nothing qualifies.
    """
    best_predictor = None
    best_abs_corr = -1
    best_corr_value = None

    for col, other_series in other_series_dict.items():
        combined = pd.DataFrame({"target": target_series, "predictor": other_series}).dropna()
        if len(combined) < min_overlap:
            continue
        corr = combined["target"].corr(combined["predictor"])
        if pd.isna(corr):
            continue
        if abs(corr) > best_abs_corr:
            best_abs_corr = abs(corr)
            best_predictor = col
            best_corr_value = corr

    return {"predictor": best_predictor, "correlation": best_corr_value}


def impute_cross_series_regression(target_series: pd.Series, predictor_series: pd.Series) -> tuple:
    """
    Fits target = a + b * predictor via ordinary least squares, using
    rows where BOTH are observed. Fills target's missing values using
    the predictor's value at that SAME row, wherever the predictor is
    itself observed there.

    Returns (imputed_series, was_imputed, could_not_fill) -- same
    three-part interface as every Module Z/26 method.
    """
    training_rows = pd.DataFrame({"target": target_series, "predictor": predictor_series}).dropna()
    if len(training_rows) < 2:
        was_imputed = pd.Series(False, index=target_series.index)
        could_not_fill = target_series.isna()
        return target_series.copy(), was_imputed, could_not_fill

    slope, intercept = np.polyfit(
        training_rows["predictor"].to_numpy(dtype=float), training_rows["target"].to_numpy(dtype=float), deg=1
    )

    was_missing = target_series.isna()
    imputed = target_series.copy()
    predictor_available = was_missing & predictor_series.notna()
    imputed.loc[predictor_available] = intercept + slope * predictor_series.loc[predictor_available]

    was_imputed = was_missing & imputed.notna()
    could_not_fill = was_missing & imputed.isna()  # predictor was ALSO missing at that exact row
    return imputed, was_imputed, could_not_fill


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: genuinely correlated series -- must find the right predictor and produce accurate estimates")
    print("=" * 60)
    df = pd.read_csv("Rainfall_multivariate_test.csv")
    df["Time_Step"] = range(len(df))
    df = df.set_index("Time_Step")

    predictors = {"Specific Humidity": df["Specific Humidity"], "Temperature": df["Temperature"]}
    corr_result = find_best_correlated_series(df["Precipitation"], predictors)
    print("Best predictor for Precipitation:", corr_result)
    print("(Precipitation was built as 50 + 3.5*Humidity + noise -- Humidity should win, strong positive correlation)")

    imputed, was_imputed, could_not_fill = impute_cross_series_regression(df["Precipitation"], df[corr_result["predictor"]])
    missing_steps = df.index[df["Precipitation"].isna()].tolist()
    print()
    for step in missing_steps:
        true_expectation = 50 + 3.5 * df.loc[step, "Specific Humidity"]
        print(f"Time_Step {step}: imputed={imputed.loc[step]:.1f}  (true Humidity-based expectation ≈ {true_expectation:.1f})")

    print("\n" + "=" * 60)
    print("TEST 2: predictor ALSO missing at the same row -- must report could_not_fill, not guess")
    print("=" * 60)
    df2 = df.copy()
    df2.loc[df2["Precipitation"].isna(), "Specific Humidity"] = None
    imputed2, was_imputed2, could_not_fill2 = impute_cross_series_regression(df2["Precipitation"], df2["Specific Humidity"])
    print("Forcing Humidity as predictor when it's missing at the same rows -- could_not_fill count:", could_not_fill2.sum(), "(should be 2)")

    print("\n" + "=" * 60)
    print("TEST 3: no usable correlation at all (unrelated random series) -- must not crash")
    print("=" * 60)
    rng_a = pd.Series(np.random.RandomState(1).normal(size=20), index=range(20))
    rng_b = pd.Series(np.random.RandomState(2).normal(size=20), index=range(20))
    rng_a.loc[5] = None
    corr_result3 = find_best_correlated_series(rng_a, {"B": rng_b})
    print("Correlation result for unrelated series:", corr_result3)

    print("\n" + "=" * 60)
    print("TEST 4: too few overlapping observed rows -- must decline rather than compute a meaningless correlation")
    print("=" * 60)
    x = pd.Series([1.0, None, None, None, 5.0], index=range(5))
    y = pd.Series([10.0, None, None, None, 50.0], index=range(5))
    corr_result4 = find_best_correlated_series(x, {"Y": y}, min_overlap=4)
    print("Result (should be predictor=None -- only 2 overlapping rows, below min_overlap=4):", corr_result4)
