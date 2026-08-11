"""
Module AQ: Standardized Method Wrapper
------------------------------------------
Per the client's explicit request: standardize each column (detrend +
scale to a common range) BEFORE feeding it to an imputation method,
then transform the result back to that column's own original scale
afterward. This is the foundation for a UNIFIED, cross-column ranking
-- comparing raw errors from a column scaled 0-400 (e.g. Precipitation)
against one scaled 7-20 (e.g. Specific Humidity) would unfairly let
whichever column has the largest numbers dominate the aggregate
decision; standardizing first makes errors genuinely comparable
across columns.

Reuses Module AI/35's fold-transform machinery exactly as-is (already
built and tested) -- this module is a thin, generic wrapper: fit a
trend+standardizer from the column's own observed values, transform
the whole series, run ANY existing method (matching the standard
(series) -> (imputed, was_imputed, could_not_fill) interface every
method in this pipeline already uses) on the standardized series, then
inverse-transform its OUTPUT back to the original scale. was_imputed
and could_not_fill are scale-independent boolean masks and pass
through unchanged; only the VALUES need transforming.
"""

from importlib import import_module

import pandas as pd

fold_transform_mod = import_module("35_time_series_fold_transforms")


def standardize_wrap_method(series: pd.Series, method_func, standardization_method: str = "zscore") -> tuple:
    """
    series: the raw column, indexed by Time_Step, with genuine gaps
    (or masked positions, when called from a backtest) as NaN.
    method_func: any function matching the (series) -> (imputed,
    was_imputed, could_not_fill) interface -- e.g. Module Z/26's
    impute_linear, impute_seasonal (with cycle_length already bound
    via a lambda), Module AG/33's cross-series regression, etc.

    Returns the SAME 3-tuple interface, with values on the ORIGINAL
    scale -- the method itself only ever sees standardized input, but
    the caller never has to think about that.
    """
    fold_transform = fold_transform_mod.fit_fold_transform(series, standardization_method)
    standardized_series = fold_transform_mod.transform(series, fold_transform)

    imputed_standardized, was_imputed, could_not_fill = method_func(standardized_series)

    imputed_original_scale = fold_transform_mod.inverse_transform(imputed_standardized, fold_transform)
    return imputed_original_scale, was_imputed, could_not_fill


if __name__ == "__main__":
    import numpy as np
    from importlib import import_module as _import_module

    single_imp_mod = _import_module("26_time_series_single_imputation")

    print("=" * 60)
    print("TEST 1: linear interpolation -- provably scale-invariant, so wrapped result")
    print("should be MATHEMATICALLY IDENTICAL to running linear directly on raw data")
    print("=" * 60)
    sales = [266.0, 145.9, 183.1, 119.3, 180.3, 168.5, 231.8, 224.5, 192.8, 122.9, 336.5, 185.9]
    series = pd.Series(sales, index=pd.Index(range(12), name="Time_Step"))
    series.loc[[3, 7]] = None

    direct_imputed, direct_was_imputed, _ = single_imp_mod.impute_linear(series)
    wrapped_imputed, wrapped_was_imputed, _ = standardize_wrap_method(series, single_imp_mod.impute_linear)

    print("Direct linear:  ", direct_imputed.loc[[3, 7]].round(6).tolist())
    print("Wrapped linear: ", wrapped_imputed.loc[[3, 7]].round(6).tolist())
    print("Identical (within floating point):", np.allclose(direct_imputed.dropna(), wrapped_imputed.dropna()))

    print("\n" + "=" * 60)
    print("TEST 2: seasonal method -- wrapped version should still produce sensible, comparable results")
    print("=" * 60)
    sales_full = [266.0, 145.9, 183.1, 119.3, 180.3, 168.5, 231.8, 224.5, 192.8, 122.9, 336.5, 185.9,
                  194.3, 149.5, 210.1, 273.3, 191.4, 287.0, 226.0, 303.6, 289.9, 421.6, 264.5, 342.3,
                  339.7, 440.4, 315.9, 439.3, 401.3, 437.4, 575.5, 407.6, 682.0, 475.3, 581.3, 646.9]
    series_full = pd.Series(sales_full, index=pd.Index(range(36), name="Time_Step"))
    series_full.loc[[8, 13]] = None

    seasonal_direct, _, _ = single_imp_mod.impute_seasonal(series_full, 12)
    seasonal_wrapped, _, _ = standardize_wrap_method(series_full, lambda s: single_imp_mod.impute_seasonal(s, 12))
    print("Direct seasonal:  ", seasonal_direct.loc[[8, 13]].round(1).tolist())
    print("Wrapped seasonal: ", seasonal_wrapped.loc[[8, 13]].round(1).tolist())
    print("(true values: 192.8, 149.5 -- both should be reasonably close)")

    print("\n" + "=" * 60)
    print("TEST 3: observed values round-trip correctly (transform + inverse_transform recovers them)")
    print("=" * 60)
    observed_positions = series_full.index[series_full.notna()].tolist()[:5]
    print("Original:", series_full.loc[observed_positions].tolist())
    print("After wrap (should be identical, within floating point):", seasonal_wrapped.loc[observed_positions].round(6).tolist())
    print("Match:", np.allclose(series_full.loc[observed_positions].to_numpy(), seasonal_wrapped.loc[observed_positions].to_numpy()))

    print("\n" + "=" * 60)
    print("TEST 4: works correctly on a completely different SCALE (values in the thousands)")
    print("=" * 60)
    big_scale_series = series_full * 1000  # same shape, 1000x the scale
    big_scale_wrapped, _, _ = standardize_wrap_method(big_scale_series, lambda s: single_imp_mod.impute_seasonal(s, 12))
    print("Original scale imputed (x1):   ", seasonal_wrapped.loc[[8, 13]].round(1).tolist())
    print("1000x scale imputed, /1000:    ", (big_scale_wrapped.loc[[8, 13]] / 1000).round(1).tolist())
    print("Consistent regardless of scale:", np.allclose(seasonal_wrapped.loc[[8, 13]].to_numpy(), (big_scale_wrapped.loc[[8, 13]] / 1000).to_numpy(), atol=0.5))
