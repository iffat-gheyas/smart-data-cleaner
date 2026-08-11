"""
Module AF: Dominant Interval Detection
--------------------------------------------
The core fix for a real flaw in the earlier design: "the finest unit
that varies" was being used as a PROXY for "the sampling frequency,"
which silently assumes consecutive observations are exactly 1 tick
apart. That assumption breaks the moment sampling isn't at every
tick -- e.g. genuinely 15-minute-interval data has a varying Minute
column, but the true interval is 15, not 1.

ALGORITHM (per the client's own strategy document):
  1. Sort all rows chronologically (assumed already done by the caller
     -- Module Q/17 already handles this).
  2. Compute every consecutive difference between UNIQUE observed
     position values, in the finest-varying unit's own counting
     system (exact calendar units for Year/Month/Quarter -- reusing
     Module S/19's existing modular position formula unchanged; fixed-
     calendar elapsed seconds for Day/Hour/Minute/Second -- reusing
     Module S/19's existing _row_to_absolute_seconds unchanged).
  3. Find the DOMINANT interval: the most common difference value.
  4. Check what fraction of ALL differences are a clean integer
     multiple of that dominant candidate (within a small tolerance) --
     if the vast majority are, larger gaps are correctly recognized as
     N missing periods, not evidence of a different frequency.
  5. If that consistency check passes a threshold, report HIGH
     confidence with the dominant interval. Otherwise, fall back to
     the MEDIAN difference (robust to a few odd outliers), report LOW
     confidence, and the caller is expected to ask the client to
     confirm before regularising the series -- never silently proceed
     on a shaky detection.
"""

from collections import Counter

CONSISTENCY_THRESHOLD = 0.9  # >=90% of gaps must be a clean integer multiple of the dominant candidate for "high" confidence
NEAR_INTEGER_TOLERANCE = 0.01  # how close a ratio must be to a whole number to count as "a clean multiple"


def detect_dominant_interval(sorted_unique_positions: list) -> dict:
    """
    sorted_unique_positions: sorted, de-duplicated list of position
    values already computed in whatever unit the finest-varying
    component uses (integers -- exact calendar units, or absolute
    seconds).

    Returns:
      {"interval": int, "confidence": "high"|"low",
       "consistency_ratio": float, "n_diffs": int,
       "dominant_candidate_count": int}

    Raises ValueError if fewer than 2 distinct positions exist -- can't
    detect an interval from a single point.
    """
    if len(sorted_unique_positions) < 2:
        raise ValueError(
            "Need at least 2 distinct time points to detect a sampling interval -- "
            f"only {len(sorted_unique_positions)} found."
        )

    diffs = [b - a for a, b in zip(sorted_unique_positions[:-1], sorted_unique_positions[1:])]

    diff_counts = Counter(diffs)
    dominant_candidate, dominant_count = diff_counts.most_common(1)[0]

    if dominant_candidate == 0:
        raise ValueError(
            "Two rows resolved to the exact same time position with a zero difference -- "
            "this indicates a duplicate-period problem that should have been resolved before "
            "interval detection runs, not a valid sampling interval."
        )

    consistent = 0
    for d in diffs:
        ratio = d / dominant_candidate
        if abs(ratio - round(ratio)) <= NEAR_INTEGER_TOLERANCE and round(ratio) >= 1:
            consistent += 1
    consistency_ratio = consistent / len(diffs)

    if consistency_ratio >= CONSISTENCY_THRESHOLD:
        return {
            "interval": dominant_candidate,
            "confidence": "high",
            "consistency_ratio": consistency_ratio,
            "n_diffs": len(diffs),
            "dominant_candidate_count": dominant_count,
        }

    sorted_diffs = sorted(diffs)
    mid = len(sorted_diffs) // 2
    median_diff = sorted_diffs[mid] if len(sorted_diffs) % 2 == 1 else round((sorted_diffs[mid - 1] + sorted_diffs[mid]) / 2)

    return {
        "interval": median_diff,
        "confidence": "low",
        "consistency_ratio": consistency_ratio,
        "n_diffs": len(diffs),
        "dominant_candidate_count": dominant_count,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1 (from the client's own doc): Month example -- 1 Jan, 1 Feb, 1 Mar 2001")
    print("   Positions in exact calendar months since some epoch: e.g. 0, 1, 2")
    print("=" * 60)
    result = detect_dominant_interval([0, 1, 2])
    print(result, " (should be interval=1, confidence=high)")

    print("\n" + "=" * 60)
    print("TEST 2 (from the doc): Day example -- 24/07/2001, 23/11/2001, 26/06/2002 (sorted)")
    print("   Positions as elapsed days: computed directly")
    print("=" * 60)
    import datetime
    dates = [datetime.date(2001, 7, 24), datetime.date(2001, 11, 23), datetime.date(2002, 6, 26)]
    epoch = dates[0]
    positions = sorted([(d - epoch).days for d in dates])
    print("Positions (days):", positions)
    result2 = detect_dominant_interval(positions)
    print(result2)
    print("Diffs should be:", [positions[1] - positions[0], positions[2] - positions[1]], "(122 and 215 days)")
    print("With only 2 diffs and no repeat, this correctly falls to LOW confidence / median -- "
          "exactly right, since 2 wildly different gaps genuinely don't establish a dominant interval.")

    print("\n" + "=" * 60)
    print("TEST 3 (from the doc): Minute example -- 10:15, 10:45, 11:15 (30 min apart)")
    print("=" * 60)
    positions3 = [0, 30, 60]  # in minutes
    result3 = detect_dominant_interval(positions3)
    print(result3, " (should be interval=30, confidence=high)")

    print("\n" + "=" * 60)
    print("TEST 4 (from the doc): Second example -- 10:15:20, 10:15:50, 10:16:20 (30 sec apart)")
    print("=" * 60)
    positions4 = [0, 30, 60]  # in seconds
    result4 = detect_dominant_interval(positions4)
    print(result4, " (should be interval=30, confidence=high)")

    print("\n" + "=" * 60)
    print("TEST 5: the ORIGINAL flaw this fixes -- genuinely 15-minute data must detect 15, not 1")
    print("=" * 60)
    positions5 = [0, 15, 30, 45, 60, 75, 90]
    result5 = detect_dominant_interval(positions5)
    print(result5, " (should be interval=15, confidence=high -- the whole point of this module)")

    print("\n" + "=" * 60)
    print("TEST 6: dominant interval WITH a genuine gap that's a clean multiple (2 missing periods)")
    print("=" * 60)
    positions6 = [0, 30, 60, 150, 180]  # 150-60=90=3x30, a real 2-period gap (2 missing 30-steps), not a new frequency
    result6 = detect_dominant_interval(positions6)
    print(result6, " (should still be interval=30, confidence=high -- 90 is a clean 3x multiple of 30)")

    print("\n" + "=" * 60)
    print("TEST 7: genuinely irregular data -- no dominant interval, low confidence, median fallback")
    print("=" * 60)
    positions7 = [0, 7, 19, 22, 41, 43]  # essentially random gaps
    result7 = detect_dominant_interval(positions7)
    print(result7, " (should be confidence=low)")

    print("\n" + "=" * 60)
    print("TEST 8: edge case -- only one distinct position (must raise, not silently return something)")
    print("=" * 60)
    try:
        detect_dominant_interval([5])
        print("BUG: did not raise")
    except ValueError as e:
        print("Correctly raised:", e)

    print("\n" + "=" * 60)
    print("TEST 9: edge case -- a zero difference (duplicate position) must raise, not silently accept")
    print("=" * 60)
    try:
        detect_dominant_interval([0, 0, 5])
        print("BUG: did not raise on zero difference")
    except ValueError as e:
        print("Correctly raised:", e)
