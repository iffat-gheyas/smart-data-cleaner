"""
Module AH: Contextual Feature Engineering for Multivariate Imputation
------------------------------------------------------------------------------
Foundation for the new contextual multivariate imputation family (per
the client's expanded specification). Every subsequent model in this
family -- regularized regression, tree-based models, and eventually
1D CNN / LSTM -- consumes the SAME feature structure built here, so
this module's correctness matters far beyond just one method.

WHAT THIS MODULE DOES: pure feature construction. Given a target
column and a set of "other" columns (already on the SAME regular
Time_Step grid -- guaranteed by Module S/19's gap-filling, which is
exactly why lag/lead computation here can be a simple .shift() rather
than needing real date arithmetic), builds:
  - the target's own lag_1..lag_k and lead_1..lead_k columns
  - each other column's contemporaneous value (and optionally its own
    lags/leads too, if requested)
  - a missingness indicator (0/1) for EVERY feature column above

WHAT THIS MODULE DELIBERATELY DOES NOT DO: detrending, standardization,
model fitting, or temporarily zero-filling missing feature values.
Per the spec, detrending and standardization must be fit using ONLY
the training rows within a given backtest fold -- that's fold-specific
information this module has no business assuming, so it stays purely
about WHICH columns exist, leaving WHEN/HOW they get transformed to
the model-fitting layer built next. Missingness indicators are built
here (since "is this raw value missing" doesn't depend on any fold),
but the temporary zero-fill for actual model input happens later too,
right before each fold's model is fit -- keeping this module a clean,
reusable, fold-agnostic building block.
"""

import pandas as pd


def build_contextual_features(
    df: pd.DataFrame,
    target_col: str,
    other_cols: list,
    max_lag: int = 3,
    max_lead: int = 3,
    other_max_lag: int = 0,
    other_max_lead: int = 0,
) -> tuple:
    """
    df: must be indexed by Time_Step (or have it as a clean, regular
    integer sequence set as the index) -- the complete, gap-filled
    sequence from Module S/19, not the raw pre-cleaning data.

    max_lag / max_lead: the target column's own lag/lead window (e.g.
    max_lag=3, max_lead=3 builds X_lag_1..X_lag_3 and X_lead_1..X_lead_3).

    other_max_lag / other_max_lead: OPTIONAL lag/lead window for the
    OTHER columns (default 0 -- contemporaneous values only, per the
    spec's "optional lagged and leading values of other variables").
    Keeping this off by default holds dimensionality down; a caller
    can opt in explicitly for a richer (and slower, more overfit-prone)
    feature set.

    Returns (features_df, feature_names) where features_df has one
    row per Time_Step, containing every raw feature column PLUS a
    "<feature>_missing" (0/1) indicator column for each one. Feature
    values themselves are left as genuine NaN where unavailable --
    NOT zero-filled here (that's a model-input-time operation, done
    later, never written to final output).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the provided dataframe.")
    for col in other_cols:
        if col not in df.columns:
            raise ValueError(f"Other column '{col}' not found in the provided dataframe.")
    if target_col in other_cols:
        raise ValueError(f"'{target_col}' cannot be both the target and one of the 'other' columns.")

    features = pd.DataFrame(index=df.index)
    feature_names = []

    # --- Target's own lags and leads ---
    for lag in range(1, max_lag + 1):
        name = f"{target_col}_lag_{lag}"
        features[name] = df[target_col].shift(lag)
        feature_names.append(name)
    for lead in range(1, max_lead + 1):
        name = f"{target_col}_lead_{lead}"
        features[name] = df[target_col].shift(-lead)
        feature_names.append(name)

    # --- Other columns: contemporaneous, plus optional lags/leads ---
    for col in other_cols:
        current_name = f"{col}_current"
        features[current_name] = df[col]
        feature_names.append(current_name)

        for lag in range(1, other_max_lag + 1):
            name = f"{col}_lag_{lag}"
            features[name] = df[col].shift(lag)
            feature_names.append(name)
        for lead in range(1, other_max_lead + 1):
            name = f"{col}_lead_{lead}"
            features[name] = df[col].shift(-lead)
            feature_names.append(name)

    # --- Missingness indicators: one per feature column, built from
    # the RAW (not yet zero-filled) feature values above ---
    missingness_names = []
    for name in feature_names:
        missing_name = f"{name}_missing"
        features[missing_name] = features[name].isna().astype(int)
        missingness_names.append(missing_name)

    return features, feature_names, missingness_names


def zero_fill_for_model_input(features: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Temporarily replaces missing feature values with 0, strictly for
    feeding into a model -- per the spec, this is an internal
    model-input operation only and must NEVER be written to the final
    cleaned dataset as an imputed result. The missingness indicator
    columns (built above) are what let the model distinguish a
    genuinely-observed 0 from a value that was filled in as a
    placeholder here.
    """
    result = features.copy()
    result[feature_names] = result[feature_names].fillna(0)
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: basic lag/lead construction, target + one other column")
    print("=" * 60)
    df = pd.DataFrame(
        {"X": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], "Y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        index=pd.Index(range(6), name="Time_Step"),
    )
    features, feature_names, missing_names = build_contextual_features(df, "X", ["Y"], max_lag=2, max_lead=2)
    print(features[feature_names].to_string())
    print()
    print("Feature names:", feature_names)
    print("Row 2 (index 2): X_lag_1 should be 20 (X at t-1), X_lead_1 should be 40 (X at t+1), Y_current should be 3")
    print("  Actual:", features.loc[2, ["X_lag_1", "X_lead_1", "Y_current"]].to_dict())

    print("\n" + "=" * 60)
    print("TEST 2: missingness indicators correctly flag NaN at the sequence boundaries")
    print("=" * 60)
    print(features[missing_names].to_string())
    print("Row 0: X_lag_1_missing and X_lag_2_missing should be 1 (no data before the start)")
    print("  Actual:", features.loc[0, ["X_lag_1_missing", "X_lag_2_missing"]].to_dict())
    print("Row 5 (last): X_lead_1_missing and X_lead_2_missing should be 1 (no data after the end)")
    print("  Actual:", features.loc[5, ["X_lead_1_missing", "X_lead_2_missing"]].to_dict())

    print("\n" + "=" * 60)
    print("TEST 3: missingness indicators also correctly reflect GENUINE gaps in the middle of the data")
    print("=" * 60)
    df2 = df.copy()
    df2.loc[3, "Y"] = None  # a genuine missing value in the middle
    features2, feature_names2, missing_names2 = build_contextual_features(df2, "X", ["Y"], max_lag=1, max_lead=1)
    print("Y_current at row 3 (should be NaN):", features2.loc[3, "Y_current"])
    print("Y_current_missing at row 3 (should be 1):", features2.loc[3, "Y_current_missing"])
    print("This missingness at row 3 correctly propagates into row 4's Y_lag_1 -- wait, other-column lags are off by default (other_max_lag=0), confirming none exist:")
    print("  Y_lag_1 present in feature_names2?", "Y_lag_1" in feature_names2, "(should be False -- other_max_lag defaults to 0)")

    print("\n" + "=" * 60)
    print("TEST 4: zero_fill_for_model_input -- fills NaN with 0 for model consumption, indicators unaffected")
    print("=" * 60)
    zero_filled = zero_fill_for_model_input(features2, feature_names2)
    print("Y_current at row 3 after zero-fill (should be 0.0):", zero_filled.loc[3, "Y_current"])
    print("Y_current_missing at row 3 (should STILL be 1, unaffected by the zero-fill):", zero_filled.loc[3, "Y_current_missing"])
    print("Original features2 unmodified (should still be NaN, not 0):", features2.loc[3, "Y_current"])

    print("\n" + "=" * 60)
    print("TEST 5: optional lags/leads of OTHER columns, when explicitly requested")
    print("=" * 60)
    features3, feature_names3, _ = build_contextual_features(df, "X", ["Y"], max_lag=1, max_lead=1, other_max_lag=1, other_max_lead=1)
    print("Feature names with other_max_lag/lead=1:", feature_names3)
    print("Y_lag_1 present?", "Y_lag_1" in feature_names3, " Y_lead_1 present?", "Y_lead_1" in feature_names3)

    print("\n" + "=" * 60)
    print("TEST 6: edge cases -- target as its own 'other' column, and an unknown column, must raise clearly")
    print("=" * 60)
    try:
        build_contextual_features(df, "X", ["X"])
        print("BUG: did not raise")
    except ValueError as e:
        print("Correctly raised:", e)
    try:
        build_contextual_features(df, "X", ["Z_does_not_exist"])
        print("BUG: did not raise")
    except ValueError as e:
        print("Correctly raised:", e)
