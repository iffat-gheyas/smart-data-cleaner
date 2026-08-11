"""
Module AD: Time Unit Hierarchy & Calendar Constants
----------------------------------------------------------
Foundational module for generalizing time-stamp handling beyond the
current Year+Month-only structure. Everything else in this rebuild
(multi-column time stamps, single-element columns, generalized
frequency detection, real date construction) builds on the unit
definitions here.

CALENDAR CONVENTION -- an explicit product decision, not a default
recommendation: every year is treated as exactly 365 days, and
February defaults to 28 days for structural/sequencing purposes.
UPDATE: a real Feb 29 appearing in actual client data IS accepted (not
rejected as an error) -- it's mapped onto the same calendar slot as
Feb 28, rather than extending the year to 366 days (which would break
the fixed model elsewhere). See day_of_year()'s docstring for the
full reasoning and the one intentional consequence: a dataset with
both Feb 28 and Feb 29 rows in the same year will collide at the same
computed period, which the existing duplicate-period detection/
resolution machinery (Section 5) already handles correctly.

The 11 non-February months use their REAL lengths (Jan=31, Sep=30,
etc.) -- these don't vary with leap years at all, so there's no
simplification involved there; it's just the calendar.
"""

# Ordered finest-to-coarsest. Used to determine "which unit is
# smaller" when detecting the smallest unit actually present across a
# time stamp's components -- that smallest unit IS the data's real
# sampling frequency (e.g. if Year and Month are both present, Month
# is finer, so the frequency is monthly).
UNIT_HIERARCHY = ["second", "minute", "hour", "day", "week", "month", "quarter", "year"]


def unit_rank(unit_name: str) -> int:
    """Lower rank = finer (smaller) unit. Raises if the unit isn't recognized -- an unrecognized
    unit should never silently sort as though it were a known one."""
    if unit_name not in UNIT_HIERARCHY:
        raise ValueError(f"Unrecognized time unit '{unit_name}'. Known units: {UNIT_HIERARCHY}")
    return UNIT_HIERARCHY.index(unit_name)


def is_finer_than(unit_a: str, unit_b: str) -> bool:
    """True if unit_a is a smaller (finer) unit than unit_b -- e.g. is_finer_than('day', 'month') is True."""
    return unit_rank(unit_a) < unit_rank(unit_b)


# --- Fixed-ratio conversions (these are genuinely fixed regardless of
# calendar convention -- 60 seconds really is always 60 seconds) ---
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
MONTHS_PER_QUARTER = 3
MONTHS_PER_YEAR = 12

SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
SECONDS_PER_WEEK = SECONDS_PER_DAY * DAYS_PER_WEEK

# --- Fixed calendar (the actual product decision) ---
DAYS_PER_YEAR = 365  # every year, no exceptions, no leap years

FIXED_MONTH_LENGTHS = {
    1: 31,   # January
    2: 28,   # February -- fixed, no leap year ever, per explicit decision
    3: 31,   # March
    4: 30,   # April
    5: 31,   # May
    6: 30,   # June
    7: 31,   # July
    8: 31,   # August
    9: 30,   # September
    10: 31,  # October
    11: 30,  # November
    12: 31,  # December
}
assert sum(FIXED_MONTH_LENGTHS.values()) == DAYS_PER_YEAR, (
    "FIXED_MONTH_LENGTHS must sum to exactly 365 -- this is the internal "
    "consistency check for the fixed-calendar convention itself."
)


def days_in_month(month_number: int) -> int:
    """
    Returns the FIXED length of the given month (1-12), under this
    product's calendar convention -- NOT dependent on which year, by
    design. February's fixed length is 28 (see day_of_year() below for
    how a real Feb 29 in actual data is still accepted, not rejected).
    """
    if month_number not in FIXED_MONTH_LENGTHS:
        raise ValueError(f"Month number must be 1-12, got {month_number}")
    return FIXED_MONTH_LENGTHS[month_number]


def day_of_year(month_number: int, day_number: int) -> int:
    """
    Returns which day of the (fixed, 365-day) year a given month/day
    falls on -- e.g. day_of_year(1, 1) == 1 (Jan 1st), day_of_year(3, 1)
    == 60 (Jan 31 + Feb 28 + 1). Used for building a single sequential
    day-count under the fixed calendar, the Day-level equivalent of how
    Month_Number already works within a Relative_Year today.

    FEBRUARY 29 IS ACCEPTED, NOT REJECTED: a real leap day that
    genuinely occurred in the client's data shouldn't be treated as an
    error just because this product's default calendar assumes 28 days
    -- but accepting it doesn't mean extending the year to 366 days
    either, since that would break the fixed-365-day model everywhere
    else. Instead, Feb 29 is mapped to the SAME day-of-year slot as
    Feb 28 (both return 59). Consequence, intentional: if a dataset has
    a row for both Feb 28 and Feb 29 in the same year, they'll collide
    at the same computed period -- which is exactly what Section 5's
    existing duplicate-period detection/resolution already handles
    (sum/mean/keep first/keep last/manual/stop), rather than needing
    new special-case logic here.

    Any OTHER out-of-range day (Feb 30, April 31, etc.) still raises,
    exactly as before -- this is a narrow, explicit exception for Feb
    29 specifically, not a general loosening of day validation.
    """
    if month_number not in FIXED_MONTH_LENGTHS:
        raise ValueError(f"Month number must be 1-12, got {month_number}")

    max_day = FIXED_MONTH_LENGTHS[month_number]
    if month_number == 2 and day_number == 29:
        effective_day = 28  # collapses onto the same slot as Feb 28 -- see docstring above
    elif not (1 <= day_number <= max_day):
        extra_note = " (Feb 29 IS accepted -- see this function's docstring; Feb 30+ is not)" if month_number == 2 else ""
        raise ValueError(
            f"Day {day_number} is invalid for month {month_number} under this fixed calendar "
            f"(month {month_number} has {max_day} days){extra_note}"
        )
    else:
        effective_day = day_number

    days_before = sum(FIXED_MONTH_LENGTHS[m] for m in range(1, month_number))
    return days_before + effective_day


# --- Recognized alternate spellings, centralized here as the single
# source of truth (Module P/16 imports MONTH_NAME_TO_NUMBER from here
# rather than keeping its own copy, so these can never drift apart) ---
def month_day_from_day_of_year(day_in_year: int) -> tuple:
    """
    The inverse of day_of_year(): given a day-in-year number (1-365,
    under this fixed 365-day calendar), returns (month, day). Needed
    to reconstruct a real date for an inserted gap row in a sub-
    monthly sequence (Day/Hour/Minute/Second-level data) -- given a
    computed position, we need to convert it back into actual
    Month_Number/Day_Number values to write into that row.

    day_of_year(month_day_from_day_of_year(x)) == x for every valid x
    in 1..365 (verified in the test suite below).
    """
    if not (1 <= day_in_year <= DAYS_PER_YEAR):
        raise ValueError(f"day_in_year must be 1-{DAYS_PER_YEAR}, got {day_in_year}")

    remaining = day_in_year
    for month in range(1, 13):
        month_length = FIXED_MONTH_LENGTHS[month]
        if remaining <= month_length:
            return month, remaining
        remaining -= month_length
    raise AssertionError("unreachable -- FIXED_MONTH_LENGTHS sums to exactly DAYS_PER_YEAR, checked at import")


MONTH_NAME_TO_NUMBER = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

QUARTER_NAME_TO_NUMBER = {
    "q1": 1, "quarter 1": 1, "1st quarter": 1,
    "q2": 2, "quarter 2": 2, "2nd quarter": 2,
    "q3": 3, "quarter 3": 3, "3rd quarter": 3,
    "q4": 4, "quarter 4": 4, "4th quarter": 4,
}


def parse_month(raw: str):
    """
    Returns (month_number, error). Accepts a plain number (1-12), a
    full month name, or a standard abbreviation, case-insensitively,
    tolerant of a trailing period ("Jan."). Returns (None, error_message)
    for anything else -- never guesses.
    """
    raw_stripped = raw.strip()
    if raw_stripped == "":
        return None, "Month is empty"
    try:
        value = int(raw_stripped)
    except ValueError:
        normalized = raw_stripped.lower().rstrip(".")
        if normalized in MONTH_NAME_TO_NUMBER:
            return MONTH_NAME_TO_NUMBER[normalized], None
        return None, f"Month '{raw_stripped}' is not a whole number or a recognized month name"
    if not (1 <= value <= 12):
        return None, f"Month '{value}' is outside the valid range 1-12"
    return value, None


def parse_quarter(raw: str):
    """Same idea as parse_month() -- accepts a plain number (1-4) or a recognized quarter label."""
    raw_stripped = raw.strip()
    if raw_stripped == "":
        return None, "Quarter is empty"
    try:
        value = int(raw_stripped)
    except ValueError:
        normalized = raw_stripped.lower().rstrip(".")
        if normalized in QUARTER_NAME_TO_NUMBER:
            return QUARTER_NAME_TO_NUMBER[normalized], None
        return None, f"Quarter '{raw_stripped}' is not a whole number or a recognized quarter label"
    if not (1 <= value <= 4):
        return None, f"Quarter '{value}' is outside the valid range 1-4"
    return value, None


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: fixed calendar sums to exactly 365 (already asserted at import -- confirming here too)")
    print("=" * 60)
    print("Sum of FIXED_MONTH_LENGTHS:", sum(FIXED_MONTH_LENGTHS.values()), "(should be 365)")

    print("\n" + "=" * 60)
    print("TEST 2: unit hierarchy ordering")
    print("=" * 60)
    print("is_finer_than('day', 'month'):", is_finer_than("day", "month"), "(should be True)")
    print("is_finer_than('year', 'month'):", is_finer_than("year", "month"), "(should be False)")
    print("is_finer_than('second', 'year'):", is_finer_than("second", "year"), "(should be True)")
    try:
        unit_rank("fortnight")
        print("BUG: did not raise on unrecognized unit")
    except ValueError as e:
        print("Correctly rejected unrecognized unit:", e)

    print("\n" + "=" * 60)
    print("TEST 3: day_of_year -- sequential day-count under the fixed calendar")
    print("=" * 60)
    print("day_of_year(1, 1) [Jan 1]:", day_of_year(1, 1), "(should be 1)")
    print("day_of_year(2, 1) [Feb 1]:", day_of_year(2, 1), "(should be 32)")
    print("day_of_year(3, 1) [Mar 1]:", day_of_year(3, 1), "(should be 60 -- 31+28+1, Feb fixed at 28)")
    print("day_of_year(12, 31) [Dec 31]:", day_of_year(12, 31), "(should be 365 -- the last day of the fixed year)")
    print("day_of_year(2, 28) [Feb 28]:", day_of_year(2, 28), "(should be 59)")
    print("day_of_year(2, 29) [Feb 29 -- a real leap day]:", day_of_year(2, 29),
          "(should ALSO be 59 -- accepted, collapsed onto the same slot as Feb 28, not rejected)")
    try:
        day_of_year(2, 30)
        print("BUG: did not reject Feb 30 (invalid even with the Feb 29 exception)")
    except ValueError as e:
        print("Correctly rejected Feb 30:", e)
    try:
        day_of_year(4, 31)
        print("BUG: did not reject April 31st (April only has 30 days)")
    except ValueError as e:
        print("Correctly rejected April 31st:", e)

    print("\n" + "=" * 60)
    print("TEST 3b: month_day_from_day_of_year -- inverse round-trips correctly across the whole year")
    print("=" * 60)
    all_correct = True
    for m in range(1, 13):
        for d in range(1, FIXED_MONTH_LENGTHS[m] + 1):
            doy = day_of_year(m, d)
            back_m, back_d = month_day_from_day_of_year(doy)
            if (back_m, back_d) != (m, d):
                print(f"  MISMATCH: ({m},{d}) -> day {doy} -> back to ({back_m},{back_d})")
                all_correct = False
    print("All 365 days round-trip correctly:", all_correct)
    print("month_day_from_day_of_year(1):", month_day_from_day_of_year(1), "(should be (1, 1))")
    print("month_day_from_day_of_year(365):", month_day_from_day_of_year(365), "(should be (12, 31))")
    print("month_day_from_day_of_year(60):", month_day_from_day_of_year(60), "(should be (3, 1) -- Mar 1st)")

    print("\n" + "=" * 60)
    print("TEST 4: month parsing -- numbers, names, abbreviations, mixed case, trailing periods")
    print("=" * 60)
    for raw in ["1", "01", "Jan", "jan", "JANUARY", "Jan.", "december", "Foo", "13", "0"]:
        value, error = parse_month(raw)
        print(f"  '{raw}' -> value={value}, error={error}")

    print("\n" + "=" * 60)
    print("TEST 5: quarter parsing")
    print("=" * 60)
    for raw in ["1", "Q1", "q3", "3rd quarter", "Q5"]:
        value, error = parse_quarter(raw)
        print(f"  '{raw}' -> value={value}, error={error}")
