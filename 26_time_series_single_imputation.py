"""
Module Z: Single-Value Imputation Algorithms
--------------------------------------------------
Implements the four methods Module Y's questions offer. Every function
takes a dataframe already at the canonical Time_Step (1..N, contiguous,
no gaps in the STEP itself -- Module S already guarantees this) and
returns (df_with_column_imputed, report).

BOUNDARY RULE, taken directly from the spec and enforced here: Linear,
PCHIP, and Cubic Spline all require REAL observed values on BOTH sides
of a gap to interpolate -- none of them can honestly fill a missing
value sitting at the very start or end of the series, since that would
mean extrapolating beyond the data rather than interpolating within
it. Rather than silently extrapolate (which pandas' own
.interpolate(limit_direction='both') would happily do without telling
you), boundary-missing values are deliberately LEFT missing by these
three methods and clearly reported as "could not be filled" -- the
Seasonal method doesn't have this restriction, since it draws from
OTHER cycles rather than neighbouring time steps, so it can fill a
boundary gap as long as at least one other observation exists at the
same seasonal phase anywhere else in the series.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, CubicSpline


def _leading_trailing_missing_mask(series: pd.Series) -> tuple:
    """
    Returns (is_leading, is_trailing) boolean masks, same index as
    series -- True for a missing value that sits before the first
    observed value (leading) or after the last observed value
    (trailing). Used to identify which gaps NO interpolation-based
    method can honestly fill.
    """
    is_missing = series.isna()
    first_observed = series.first_valid_index()
    last_observed = series.last_valid_index()

    is_leading = pd.Series(False, index=series.index)
    is_trailing = pd.Series(False, index=series.index)
    if first_observed is not None:
        is_leading = is_missing & (series.index < first_observed)
    if last_observed is not None:
        is_trailing = is_missing & (series.index > last_observed)
    return is_leading, is_trailing


def impute_linear(series: pd.Series) -> tuple:
    """Straight line between the two surrounding known values. Cannot fill boundary gaps."""
    is_leading, is_trailing = _leading_trailing_missing_mask(series)
    was_missing = series.isna()
    # limit_area="inside", NOT limit_direction=None -- verified directly
    # that pandas' default behaviour asymmetrically forward-fills
    # TRAILING gaps (propagating the last known value) while correctly
    # leaving LEADING gaps alone, which silently violated the "never
    # extrapolate boundary values" rule for exactly the trailing case.
    # limit_area="inside" is the parameter that actually restricts
    # filling to interior gaps only, in both directions.
    imputed = series.interpolate(method="linear", limit_area="inside")
    was_imputed = was_missing & imputed.notna()
    return imputed, was_imputed, (is_leading | is_trailing) & was_missing


def impute_pchip(series: pd.Series) -> tuple:
    """
    Curved but shape-preserving: guaranteed to never overshoot beyond
    the surrounding known values (unlike cubic spline). Needs at least
    2 real observed points to fit; interior gaps only, same boundary
    rule as linear.
    """
    is_leading, is_trailing = _leading_trailing_missing_mask(series)
    was_missing = series.isna()

    known_x = series.index[series.notna()].to_numpy()
    known_y = series[series.notna()].to_numpy()
    imputed = series.copy()

    if len(known_x) >= 2:
        interpolator = PchipInterpolator(known_x, known_y, extrapolate=False)
        interior_missing = was_missing & ~(is_leading | is_trailing)
        fill_x = series.index[interior_missing].to_numpy()
        if len(fill_x):
            imputed.loc[interior_missing] = interpolator(fill_x)

    was_imputed = was_missing & imputed.notna()
    return imputed, was_imputed, (is_leading | is_trailing) & was_missing


def impute_cubic_spline(series: pd.Series) -> tuple:
    """
    Smoothest curve of the three interpolation methods, but CAN
    overshoot beyond the surrounding known values (e.g. a dip between
    two rising points) -- offered because it's sometimes genuinely the
    better fit for smooth underlying trends, with that trade-off made
    explicit in Module Y's question text. Needs at least 4 points for
    a stable cubic fit; interior gaps only.
    """
    is_leading, is_trailing = _leading_trailing_missing_mask(series)
    was_missing = series.isna()

    known_x = series.index[series.notna()].to_numpy()
    known_y = series[series.notna()].to_numpy()
    imputed = series.copy()

    if len(known_x) >= 4:
        interpolator = CubicSpline(known_x, known_y, extrapolate=False)
        interior_missing = was_missing & ~(is_leading | is_trailing)
        fill_x = series.index[interior_missing].to_numpy()
        if len(fill_x):
            imputed.loc[interior_missing] = interpolator(fill_x)

    was_imputed = was_missing & imputed.notna()
    return imputed, was_imputed, (is_leading | is_trailing) & was_missing


def impute_seasonal_naive(series: pd.Series, cycle_length: int) -> tuple:
    """
    Fills each missing value with the MEDIAN of all other OBSERVED
    values at the same phase of the cycle (e.g. every other September,
    for monthly data with cycle_length=12) -- median rather than mean
    since it's more robust to a single unusual year skewing the
    estimate. Median, not skipped, for the case where only ONE other
    observation exists at that phase (median of one value is just that
    value). Can fill boundary gaps, unlike the three interpolation
    methods, since it doesn't depend on neighbouring time steps at all.
    Genuinely cannot fill a gap only if EVERY OTHER observation at that
    same phase is also missing.

    KNOWN LIMITATION, kept here for reference/fallback only -- ignores
    trend entirely, which showed up as a real accuracy problem on a
    genuinely trending series (confirmed by testing: this performed
    WORST of all four methods on real shampoo data specifically
    because of this). impute_seasonal_detrended() below is the actual
    production method; this stays available as the fallback it uses
    when there are too few points to fit a trend line at all.
    """
    was_missing = series.isna()
    imputed = series.copy()
    phase = series.index % cycle_length

    for missing_idx in series.index[was_missing]:
        same_phase_mask = (phase == phase[series.index.get_loc(missing_idx)]) & series.notna()
        same_phase_values = series[same_phase_mask]
        if len(same_phase_values) > 0:
            imputed.loc[missing_idx] = same_phase_values.median()

    was_imputed = was_missing & imputed.notna()
    still_missing = was_missing & imputed.isna()
    return imputed, was_imputed, still_missing


def impute_seasonal(series: pd.Series, cycle_length: int) -> tuple:
    """
    THE production seasonal method -- detrend first, then take the
    median RESIDUAL at the same seasonal phase, then add the trend
    back. Fixes the naive version's confirmed blind spot: a straight
    median of "same month, other years" ignores any overall trend,
    which systematically biased estimates on the real shampoo test
    (worst of all four methods there, specifically because of this).

    Trend is a simple ordinary-least-squares straight line -- a
    deliberate, honest scope choice: good enough when the underlying
    trend is roughly linear, without pulling in a full STL
    decomposition for a first version. Falls back to the naive version
    when fewer than 2 observed points exist (can't fit a trend line
    through fewer than 2 points at all).
    """
    observed = series.dropna()
    if len(observed) < 2:
        return impute_seasonal_naive(series, cycle_length)

    slope, intercept = np.polyfit(observed.index.to_numpy(dtype=float), observed.to_numpy(dtype=float), deg=1)
    trend_at = lambda x: slope * x + intercept  # noqa: E731

    residuals = observed - trend_at(observed.index.to_numpy(dtype=float))
    phase_of = lambda idx: idx % cycle_length  # noqa: E731

    was_missing = series.isna()
    imputed = series.copy()
    for missing_idx in series.index[was_missing]:
        same_phase_mask = phase_of(residuals.index.to_numpy(dtype=float)) == phase_of(missing_idx)
        same_phase_residuals = residuals[same_phase_mask]
        if len(same_phase_residuals) > 0:
            imputed.loc[missing_idx] = trend_at(missing_idx) + same_phase_residuals.median()

    was_imputed = was_missing & imputed.notna()
    still_missing = was_missing & imputed.isna()
    return imputed, was_imputed, still_missing


def _detrend_then_interpolate(series: pd.Series, interpolation_func) -> tuple:
    """
    Removes a simple OLS linear trend, runs the given LOCAL
    interpolation method on the detrended series, adds the trend back
    at whichever positions actually got filled. Falls back to the
    plain (non-detrended) version when there isn't enough data to fit
    a trend at all.

    Deliberately doesn't need cycle_length -- unlike deseasonalizing,
    detrending alone works even for data with no seasonal cycle at
    all, so this generalizes more broadly than the deseasonalized
    variants below.

    NOT offered for impute_linear specifically: detrend-then-linearly-
    interpolate-then-add-trend-back is PROVABLY mathematically
    identical to plain linear interpolation for any straight-line
    trend (verified directly) -- linear interpolation is invariant to
    subtracting and re-adding a linear function, so a "detrended
    linear" variant would be pure wasted computation with zero
    difference in output, not a genuine alternative.
    """
    observed = series.dropna()
    if len(observed) < 2:
        return interpolation_func(series)

    slope, intercept = np.polyfit(observed.index.to_numpy(dtype=float), observed.to_numpy(dtype=float), deg=1)
    trend_at = lambda x: slope * x + intercept  # noqa: E731

    detrended = series - trend_at(series.index.to_numpy(dtype=float))
    imputed_detrended, was_imputed, could_not_fill = interpolation_func(detrended)

    imputed = series.copy()
    imputed.loc[was_imputed] = imputed_detrended.loc[was_imputed] + trend_at(
        series.index[was_imputed].to_numpy(dtype=float)
    )
    return imputed, was_imputed, could_not_fill


def impute_pchip_detrended(series: pd.Series) -> tuple:
    return _detrend_then_interpolate(series, impute_pchip)


# NOTE: no impute_cubic_spline_detrended -- confirmed both mathematically
# and empirically (bit-identical results) that detrending is redundant
# for cubic spline too, same reasoning as linear interpolation: a
# straight-line trend has zero second AND third derivative everywhere,
# so it doesn't change any of the conditions scipy's CubicSpline (or a
# natural spline) is solving for. Only PCHIP genuinely benefits, since
# its shape-preserving logic operates on raw between-point slopes,
# which DO change when a trend is removed first.


def _classical_seasonal_decomposition(series: pd.Series, cycle_length: int) -> tuple:
    """
    Fits a simple OLS trend, then the median RESIDUAL at each phase of
    the cycle -- classical additive decomposition, same trend-fitting
    approach as impute_seasonal() above, just exposed separately so
    the interpolation methods below can remove the seasonal component
    before interpolating and add it back afterward.

    Returns (trend_at, seasonal_component) where trend_at is a
    callable and seasonal_component is a dict {phase: value}. Returns
    (None, None) if there's not enough observed data to fit a trend
    (fewer than 2 points) -- callers should fall back to the
    non-deseasonalized version in that case.
    """
    observed = series.dropna()
    if len(observed) < 2:
        return None, None

    slope, intercept = np.polyfit(observed.index.to_numpy(dtype=float), observed.to_numpy(dtype=float), deg=1)
    trend_at = lambda x: slope * x + intercept  # noqa: E731

    detrended = observed - trend_at(observed.index.to_numpy(dtype=float))
    phase_of_index = observed.index.to_numpy(dtype=float) % cycle_length
    seasonal_component = {}
    for phase in range(cycle_length):
        phase_values = detrended[phase_of_index == phase]
        seasonal_component[phase] = phase_values.median() if len(phase_values) > 0 else 0.0

    return trend_at, seasonal_component


def _deseasonalize_then_interpolate(series: pd.Series, cycle_length: int, interpolation_func) -> tuple:
    """
    Removes the seasonal component (classical decomposition), runs the
    given LOCAL interpolation method on the deseasonalized series, then
    adds the seasonal component back at whichever positions actually
    got filled. Falls back to the plain (non-deseasonalized) version
    when there isn't enough data to decompose at all.
    """
    trend_at, seasonal_component = _classical_seasonal_decomposition(series, cycle_length)
    if trend_at is None:
        return interpolation_func(series)

    phase_of = lambda idx: int(idx) % cycle_length  # noqa: E731
    seasonal_at = series.index.map(lambda idx: seasonal_component[phase_of(idx)])
    deseasonalized = series - pd.Series(seasonal_at, index=series.index)

    imputed_deseason, was_imputed, could_not_fill = interpolation_func(deseasonalized)

    imputed = series.copy()
    imputed.loc[was_imputed] = imputed_deseason.loc[was_imputed] + pd.Series(seasonal_at, index=series.index).loc[was_imputed]

    return imputed, was_imputed, could_not_fill


def impute_linear_deseasonalized(series: pd.Series, cycle_length: int) -> tuple:
    return _deseasonalize_then_interpolate(series, cycle_length, impute_linear)


def impute_pchip_deseasonalized(series: pd.Series, cycle_length: int) -> tuple:
    return _deseasonalize_then_interpolate(series, cycle_length, impute_pchip)


def impute_cubic_spline_deseasonalized(series: pd.Series, cycle_length: int) -> tuple:
    return _deseasonalize_then_interpolate(series, cycle_length, impute_cubic_spline)


def _fully_decompose_then_interpolate(series: pd.Series, cycle_length: int, interpolation_func) -> tuple:
    """
    Removes BOTH the trend AND the seasonal component (full classical
    decomposition), runs the given interpolation method on the pure
    residual, adds both back. Genuinely distinct from
    _deseasonalize_then_interpolate() above (which leaves the trend
    IN what gets interpolated) -- not offered for linear or cubic
    spline, since both are separately provable to be invariant to the
    trend-removal half of this (see impute_pchip_detrended's
    docstring for the proof), which would make a "fully decomposed"
    cubic-spline/linear variant mathematically identical to their
    already-existing deseasonalized versions -- not a new option, just
    a relabeled one. PCHIP has no such invariance (confirmed: detrending
    alone already changes its output), so this combination is worth
    actually offering and testing for PCHIP specifically.
    """
    trend_at, seasonal_component = _classical_seasonal_decomposition(series, cycle_length)
    if trend_at is None:
        return interpolation_func(series)

    phase_of = lambda idx: int(idx) % cycle_length  # noqa: E731
    seasonal_at = series.index.map(lambda idx: seasonal_component[phase_of(idx)])
    trend_at_index = pd.Series(trend_at(series.index.to_numpy(dtype=float)), index=series.index)
    seasonal_at_index = pd.Series(seasonal_at, index=series.index)

    fully_decomposed = series - trend_at_index - seasonal_at_index
    imputed_residual, was_imputed, could_not_fill = interpolation_func(fully_decomposed)

    imputed = series.copy()
    imputed.loc[was_imputed] = (
        imputed_residual.loc[was_imputed] + trend_at_index.loc[was_imputed] + seasonal_at_index.loc[was_imputed]
    )
    return imputed, was_imputed, could_not_fill


def impute_pchip_fully_decomposed(series: pd.Series, cycle_length: int) -> tuple:
    return _fully_decompose_then_interpolate(series, cycle_length, impute_pchip)


SINGLE_IMPUTATION_FUNCTIONS = {
    "linear": impute_linear,
    "pchip": impute_pchip,
    "cubic_spline": impute_cubic_spline,
    "pchip_detrended": impute_pchip_detrended,
    # "seasonal" handled separately below -- needs cycle_length, the others don't
}


def apply_single_imputation(df: pd.DataFrame, value_column: str, method: str, cycle_length: int = None) -> tuple:
    """
    Returns (df_with_imputation, report). Adds two new columns:
      "<value_column>_was_imputed" (0/1, matches the cross-sectional
        pipeline's convention from Module 14's encode_imputation_indicator_columns)
      "<value_column>_imputation_method" (the method name, only on
        rows that were actually imputed -- NA everywhere else)

    Rows whose gap couldn't honestly be filled (boundary case for the
    three interpolation methods, or a seasonal gap with zero other
    observations at that phase) are left genuinely missing and
    reported separately -- never silently zero-filled or dropped.
    """
    df = df.copy()
    series = df.set_index("Time_Step")[value_column]

    deseasonalized_methods = {
        "linear_deseasonalized": impute_linear_deseasonalized,
        "pchip_deseasonalized": impute_pchip_deseasonalized,
        "cubic_spline_deseasonalized": impute_cubic_spline_deseasonalized,
        "pchip_fully_decomposed": impute_pchip_fully_decomposed,
    }

    if method == "seasonal":
        if not cycle_length or cycle_length <= 1:
            raise ValueError(
                "Seasonal imputation needs a real cycle length (e.g. 12 for monthly data) -- "
                "this dataset's detected frequency doesn't have one, so seasonal imputation "
                "isn't a coherent choice here."
            )
        imputed, was_imputed, could_not_fill = impute_seasonal(series, cycle_length)
    elif method in deseasonalized_methods:
        if not cycle_length or cycle_length <= 1:
            raise ValueError(
                f"'{method}' needs a real cycle length to remove a seasonal component -- "
                f"this dataset's detected frequency doesn't have one."
            )
        imputed, was_imputed, could_not_fill = deseasonalized_methods[method](series, cycle_length)
    elif method in SINGLE_IMPUTATION_FUNCTIONS:
        imputed, was_imputed, could_not_fill = SINGLE_IMPUTATION_FUNCTIONS[method](series)
    else:
        raise ValueError(
            f"Unknown single-imputation method '{method}'. Choose from linear, pchip, cubic_spline, "
            f"seasonal, linear_deseasonalized, pchip_deseasonalized, cubic_spline_deseasonalized."
        )

    imputed_aligned = imputed.reindex(df["Time_Step"]).to_numpy()
    was_imputed_aligned = was_imputed.reindex(df["Time_Step"]).fillna(False).to_numpy()
    could_not_fill_aligned = could_not_fill.reindex(df["Time_Step"]).fillna(False).to_numpy()

    df[value_column] = imputed_aligned
    df[f"{value_column}_was_imputed"] = was_imputed_aligned.astype(int)
    df[f"{value_column}_imputation_method"] = np.where(was_imputed_aligned, method, None)

    report = {
        "method": method,
        "n_imputed": int(was_imputed_aligned.sum()),
        "n_could_not_fill": int(could_not_fill_aligned.sum()),
        "time_steps_could_not_fill": df.loc[could_not_fill_aligned, "Time_Step"].tolist(),
    }
    return df, report


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
    print("TEST 1: real reconstructed data, all four methods, checked against KNOWN true values")
    print("   (1-09 true value = 192.8, 2-02 true value = 149.5)")
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

    # Identify which Time_Step correspond to the two known-missing points
    missing_rows = final_df[final_df["Sales"].isna()][["Time_Step", "Relative_Year", "Month_Number"]]
    print("Missing rows:")
    print(missing_rows.to_string(index=False))
    print()

    for method in ["linear", "pchip", "cubic_spline", "seasonal"]:
        cycle_length = freq_info["cycle_length"] if method == "seasonal" else None
        result_df, report = apply_single_imputation(final_df, "Sales", method, cycle_length)
        vals = result_df.loc[result_df["Time_Step"].isin(missing_rows["Time_Step"]), ["Time_Step", "Sales"]]
        print(f"Method: {method}")
        print(f"  n_imputed={report['n_imputed']}, n_could_not_fill={report['n_could_not_fill']}")
        for _, row in vals.iterrows():
            true_val = 192.8 if row["Time_Step"] == missing_rows.iloc[0]["Time_Step"] else 149.5
            error = abs(row["Sales"] - true_val)
            print(f"    Time_Step {int(row['Time_Step'])}: imputed={row['Sales']:.1f}  true={true_val}  abs_error={error:.1f}")
        print()

    print("=" * 60)
    print("TEST 2: boundary missing value -- must be LEFT missing by interpolation methods, reported")
    print("=" * 60)
    boundary_series = pd.Series(
        [np.nan, np.nan, 10.0, 12.0, 15.0, 13.0, np.nan],
        index=pd.Index(range(1, 8), name="Time_Step"),
    )
    for method_name, func in [("linear", impute_linear), ("pchip", impute_pchip), ("cubic_spline", impute_cubic_spline)]:
        imputed, was_imputed, could_not_fill = func(boundary_series)
        print(f"{method_name}: result={imputed.tolist()}")
        print(f"  could_not_fill positions: {boundary_series.index[could_not_fill].tolist()} (should be [1, 2, 7] -- 2 leading + 1 trailing)")

    print()
    print("=" * 60)
    print("TEST 3: seasonal CAN fill a boundary gap (draws from other cycles, not neighbours)")
    print("=" * 60)
    seasonal_series = pd.Series(
        [np.nan, 20, 30, 40, 22, 31, 41, 25, 29, 39],
        index=pd.Index(range(1, 11), name="Time_Step"),
    )
    imputed, was_imputed, could_not_fill = impute_seasonal(seasonal_series, cycle_length=3)
    print("result:", imputed.tolist())
    print("Time_Step 1 (phase 1 mod 3 = 1, same phase as steps 4,7,10 = 40,25,39, median=39.0):", imputed.iloc[0])
