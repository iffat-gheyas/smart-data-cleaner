"""
Module AP: Plausibility Bounds
------------------------------------
Prevents imputation from producing physically impossible values (e.g.
negative rainfall) -- deliberately built LAST in this whole imputation
system, per the client's own reasoning: bounds should only be applied
once method performance has already been properly evaluated, since
clipping could otherwise mask a genuinely poor-performing method's
errors rather than fixing the real problem (choosing a better method).

CORE PRINCIPLE, consistent with everything else in this pipeline:
bounds are applied ONLY to values THIS PRODUCT estimates, never to
genuinely observed data -- even if a real observed value happens to
fall outside the detected/confirmed range (e.g. a sensor quirk
producing one slightly-negative rainfall reading). That's the
client's actual data, not something to silently "correct" just because
it looks implausible; plausibility bounds constrain what we invent,
not what the client already has.

AUTO-SUGGESTION IS DELIBERATELY CONSERVATIVE: only a LOWER bound of 0
is ever auto-suggested, and only when every genuinely observed value
in that column is already >= 0 -- a common, low-risk pattern for
real-world non-negative quantities (rainfall, prices, counts,
distances). No upper bound is ever auto-guessed; that's far easier to
get wrong (e.g. assuming the observed maximum IS the true ceiling, when
it's just the largest value seen so far). A client can add a custom
upper (or lower) bound explicitly if they know something this
heuristic can't infer, such as a percentage column capped at 100.
"""

import pandas as pd


def detect_suggested_lower_bound(observed_values: pd.Series):
    """
    Returns 0.0 if every genuinely observed (non-NaN) value in the
    series is already >= 0, else None (no suggestion -- the data
    itself contains negative values, so 0 clearly isn't a real floor
    for this variable).
    """
    observed = observed_values.dropna()
    if len(observed) == 0:
        return None
    return 0.0 if observed.min() >= 0 else None


def clip_to_bounds(series: pd.Series, lower: float = None, upper: float = None) -> pd.Series:
    """Clips every value in `series` to [lower, upper]. Either bound may be None (no clipping on that side)."""
    result = series.copy()
    if lower is not None:
        result = result.clip(lower=lower)
    if upper is not None:
        result = result.clip(upper=upper)
    return result


def clip_only_imputed_values(series: pd.Series, was_imputed_mask: pd.Series, lower: float = None, upper: float = None) -> pd.Series:
    """
    Clips ONLY the positions marked as imputed in `was_imputed_mask` --
    genuinely observed values are NEVER touched, even if (unusually)
    one falls outside the confirmed bounds. `was_imputed_mask` should
    align with `series`'s index (a boolean Series or array, same shape
    as an existing "<col>_was_imputed" column, 1/True = imputed).
    """
    result = series.copy()
    was_imputed_bool = pd.Series(was_imputed_mask, index=series.index).astype(bool)
    clipped_everywhere = clip_to_bounds(series, lower, upper)
    result.loc[was_imputed_bool] = clipped_everywhere.loc[was_imputed_bool]
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: lower-bound auto-suggestion -- all-non-negative data suggests 0.0")
    print("=" * 60)
    all_nonneg = pd.Series([0.0, 5.2, 100.3, 0.0, 42.1])
    print("All non-negative:", detect_suggested_lower_bound(all_nonneg), "(should be 0.0)")

    has_negative = pd.Series([-5.0, 5.2, 100.3])
    print("Has a genuine negative value:", detect_suggested_lower_bound(has_negative), "(should be None -- 0 clearly isn't a real floor)")

    empty_series = pd.Series([None, None], dtype=float)
    print("All-missing series:", detect_suggested_lower_bound(empty_series), "(should be None -- nothing to infer from)")

    print("\n" + "=" * 60)
    print("TEST 2: clip_to_bounds -- basic clipping both directions")
    print("=" * 60)
    values = pd.Series([-10.0, 5.0, 50.0, 150.0])
    clipped = clip_to_bounds(values, lower=0.0, upper=100.0)
    print("Original:", values.tolist())
    print("Clipped [0, 100]:", clipped.tolist(), "(should be [0.0, 5.0, 50.0, 100.0])")

    print("\n" + "=" * 60)
    print("TEST 3: CORE PRINCIPLE -- genuinely observed values are NEVER clipped, even if implausible")
    print("=" * 60)
    series_with_quirk = pd.Series([-2.0, 10.0, -50.0, 30.0])  # -2.0 is a genuine OBSERVED quirky value, -50.0 is an IMPUTED bad estimate
    was_imputed = pd.Series([False, False, True, False])
    result = clip_only_imputed_values(series_with_quirk, was_imputed, lower=0.0)
    print("Original:      ", series_with_quirk.tolist())
    print("was_imputed:   ", was_imputed.tolist())
    print("After clipping:", result.tolist())
    print("Position 0 (-2.0, genuinely OBSERVED) left untouched:", result.iloc[0] == -2.0, "(should be True -- never silently corrected)")
    print("Position 2 (-50.0, IMPUTED) correctly clipped to 0:", result.iloc[2] == 0.0, "(should be True)")

    print("\n" + "=" * 60)
    print("TEST 4: no bounds specified -- clip_to_bounds is a no-op")
    print("=" * 60)
    unclipped = clip_to_bounds(values, lower=None, upper=None)
    print("Unchanged when no bounds given:", unclipped.tolist() == values.tolist())
