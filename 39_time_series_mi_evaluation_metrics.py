"""
Module AM: Multiple Imputation Evaluation Metrics
------------------------------------------------------------
Two accuracy measures, per the client's explicit specification, used
to rank EVERY imputation method under Multiple Imputation (not just
the single seasonal-bootstrap method currently available):

1. INTERVAL ACCURACY: the % of masked test points whose TRUE value
   genuinely falls within the method's own constructed 95% interval
   (2.5th/97.5th percentile of its M draws) -- an empirical coverage
   rate. A well-calibrated method should cover close to 95% of the
   time; this measures whether that actually holds for THIS method on
   THIS dataset, not just assumed.

2. MAAPE-BASED ACCURACY: Mean Arctangent Absolute Percentage Error,
   using the MEDIAN of each method's M draws as its point estimate.
   MAAPE fixes ordinary MAPE's real flaw -- MAPE blows up to infinity
   whenever the true value is at or near zero (a genuine problem for
   data like rainfall, where zero is a normal, frequent value, not an
   edge case). MAAPE instead takes arctan(|error|/|actual|), which
   stays bounded between 0 and pi/2 even as that ratio approaches
   infinity, then converts to an intuitive 0-100% accuracy score.

COMBINED SCORE: a weighted average of the two, with interval accuracy
counted TWICE as heavily as MAAPE-based accuracy -- an explicit,
disclosed weighting choice, not treated as equally-important by
default.
"""

import numpy as np


def compute_maape(actual, predicted) -> float:
    """
    Raw MAAPE value, ranging 0 (perfect) to pi/2 (worst possible,
    when the relative error approaches infinity).

    actual == 0 and predicted == 0 (a genuine 0/0 case -- both
    correctly say "zero") is treated as PERFECT (contributes 0 error),
    not undefined -- this is the correct interpretation, distinct from
    actual == 0 with predicted != 0 (a genuine miss, correctly
    contributing the maximum possible error, pi/2, since ANY nonzero
    prediction is infinitely wrong relative to a true value of exactly
    zero).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(predicted - actual) / np.abs(actual)

    # 0/0 (both actual and predicted are 0) -> nan -> treat as 0 error (perfect), not undefined.
    # nonzero/0 -> inf -> arctan(inf) = pi/2 automatically, no special-casing needed.
    ratio = np.where(np.isnan(ratio), 0.0, ratio)

    return float(np.mean(np.arctan(ratio)))


def maape_to_accuracy_pct(maape_value: float) -> float:
    """
    Converts a raw MAAPE value (0 to pi/2) into an intuitive 0-100%
    accuracy score: 100% for perfect predictions (MAAPE=0), 0% for the
    theoretical worst case (MAAPE=pi/2, i.e. arctan saturated at its
    maximum).
    """
    return 100.0 * (1.0 - maape_value / (np.pi / 2))


def compute_interval_coverage_pct(actual, lower, upper) -> float:
    """
    The % of actual values that genuinely fall within [lower, upper]
    -- the empirical coverage rate of a method's own constructed
    intervals, NOT assumed to be 95% just because that's the nominal
    target. Returns a plain 0-100 percentage.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (actual >= lower) & (actual <= upper)
    return 100.0 * float(np.mean(covered))


def compute_combined_score(interval_accuracy_pct: float, maape_accuracy_pct: float, interval_weight: float = 2.0, maape_weight: float = 1.0) -> float:
    """
    Weighted average of the two accuracy measures. Per the client's
    explicit instruction, interval accuracy counts TWICE as heavily as
    MAAPE-based accuracy by default (interval_weight=2, maape_weight=1)
    -- an intentional, disclosed choice, not an equal-weighting default.
    """
    total_weight = interval_weight + maape_weight
    return (interval_weight * interval_accuracy_pct + maape_weight * maape_accuracy_pct) / total_weight


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: perfect predictions -- MAAPE=0, accuracy=100%")
    print("=" * 60)
    actual = [10.0, 20.0, 30.0, 40.0]
    predicted_perfect = [10.0, 20.0, 30.0, 40.0]
    maape_perfect = compute_maape(actual, predicted_perfect)
    print("MAAPE:", maape_perfect, "(should be 0)")
    print("Accuracy %:", maape_to_accuracy_pct(maape_perfect), "(should be 100)")

    print("\n" + "=" * 60)
    print("TEST 2: wildly wrong predictions -- MAAPE approaches pi/2, accuracy approaches 0%")
    print("=" * 60)
    actual2 = [10.0, 20.0, 30.0]
    predicted_terrible = [10000.0, 20000.0, 30000.0]  # ~1000x too high, ratio huge
    maape_terrible = compute_maape(actual2, predicted_terrible)
    print("MAAPE:", maape_terrible, "(should be close to pi/2 =", np.pi / 2, ")")
    print("Accuracy %:", maape_to_accuracy_pct(maape_terrible), "(should be close to 0)")

    print("\n" + "=" * 60)
    print("TEST 3: actual=0 edge cases -- both 0 (perfect) vs actual=0 with nonzero prediction (max error)")
    print("=" * 60)
    both_zero_maape = compute_maape([0.0], [0.0])
    print("actual=0, predicted=0 (genuine agreement):", both_zero_maape, "(should be 0 -- NOT undefined)")
    miss_at_zero_maape = compute_maape([0.0], [5.0])
    print("actual=0, predicted=5 (genuine miss):", miss_at_zero_maape, "(should be pi/2 =", np.pi / 2, "-- max error)")

    print("\n" + "=" * 60)
    print("TEST 4: interval coverage -- hand-verifiable cases (100%, 0%, 50%)")
    print("=" * 60)
    actual_cov = [10.0, 20.0, 30.0, 40.0]
    lower_all_cover = [5.0, 15.0, 25.0, 35.0]
    upper_all_cover = [15.0, 25.0, 35.0, 45.0]
    print("All 4 covered:", compute_interval_coverage_pct(actual_cov, lower_all_cover, upper_all_cover), "(should be 100.0)")

    lower_none_cover = [100.0, 100.0, 100.0, 100.0]
    upper_none_cover = [200.0, 200.0, 200.0, 200.0]
    print("None covered:", compute_interval_coverage_pct(actual_cov, lower_none_cover, upper_none_cover), "(should be 0.0)")

    lower_half = [5.0, 15.0, 100.0, 100.0]  # first two covered, last two not
    upper_half = [15.0, 25.0, 200.0, 200.0]
    print("Half covered:", compute_interval_coverage_pct(actual_cov, lower_half, upper_half), "(should be 50.0)")

    print("\n" + "=" * 60)
    print("TEST 5: combined score -- exact hand-verified weighted average (interval x2, MAAPE x1)")
    print("=" * 60)
    combined = compute_combined_score(interval_accuracy_pct=80.0, maape_accuracy_pct=50.0)
    expected = (2 * 80.0 + 1 * 50.0) / 3
    print(f"combine(interval=80, maape=50) = {combined}  (hand-computed expected = {expected})")
    print("Exact match:", combined == expected)

    print("\n" + "=" * 60)
    print("TEST 6: combined score sanity -- equal inputs should return that same value regardless of weights")
    print("=" * 60)
    combined_equal = compute_combined_score(interval_accuracy_pct=75.0, maape_accuracy_pct=75.0)
    print("combine(75, 75):", combined_equal, "(should be exactly 75.0)")
