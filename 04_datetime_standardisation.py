"""
Module D: Date/Time Standardisation
--------------------------------------
Parses inconsistently formatted date and date+time strings into ONE
standard, consistent format. This module does NOT split dates into
separate year/month/day/hour/minute/second columns -- that's Module E
(Column Splitting), which runs after this one, since it needs a
properly parsed datetime to split apart in the first place.

KEY DESIGN DECISION: ambiguous numeric dates (e.g. "03/04/2026" -- is
that 3 April or 4 March?) are resolved using an explicit `dayfirst`
setting that YOU choose per job, never auto-guessed. Guessing wrong
here would silently corrupt data, which is worse than asking.

HONEST LIMITATION: if a column mixes genuinely ambiguous formats (some
rows day-first, some rows month-first -- which our synthetic dataset
deliberately does, to stress-test this), no single `dayfirst` setting
can get every row right. Rows where the day value is above 12 resolve
correctly regardless of the setting, because a month can't exceed 12.
Only the genuinely ambiguous rows (both parts 12 or below) depend on
the dayfirst assumption -- and for those, if the source data mixed
conventions, some will inevitably be misread. This is a real limit of
working with already-inconsistent data, not a bug in this code.

A REAL BUG FIXED after real-file testing, serious enough to explain in
full: pandas' to_datetime(..., dayfirst=True, format="mixed") does NOT
limit the dayfirst reinterpretation to genuinely ambiguous values --
it was silently applying it to ALREADY-UNAMBIGUOUS ISO-ordered dates
too. "2022-08-07" is ALWAYS Year-Month-Day in ISO order -- there is no
day-vs-month question to resolve there at all -- yet with dayfirst=True
it was being flipped to "2022-07-08", a genuine silent date corruption,
not a parse failure. This is now fixed by detecting ISO-ordered values
up front and parsing them on their own, always with dayfirst=False
(irrelevant to them anyway, but this guarantees pandas' heuristic never
touches them), while every other value still goes through the normal
dayfirst-dependent path exactly as before. See standardise_date_column
for the implementation.

A SECOND, MUCH LARGER round of fixes, added after testing against a
genuinely adversarial real client file full of spoken-English times --
"half past three", "fifteen minutes to nine o'clock", "thirty past
10", "fourteen-twenty", ordinal days ("6th", "3rd"), noisy colon-heavy
separators ("27:01:2022: 17: 20:10", "10-04-2022-20:18:17"), and even
a misspelled number word ("forteen" for "fourteen"). None of this is
parseable by pandas on its own -- see normalize_natural_language_
datetime() and its helper functions below, which rewrite all of this
into plain HH:MM(:SS) text BEFORE pandas ever sees it. Verified against
all 19 distinct values from the real file that prompted this; every
one now parses to a sensible date and time.

HONEST LIMITATION on the natural-language side: when no AM/PM is
stated at all ("2 o'clock", "half past three"), there's no way to know
which the person meant -- this resolves to the literal hour value
(e.g. "2 o'clock" -> 02:00, not 14:00), the same "don't guess beyond
what's stated" principle as the dayfirst setting above, just with no
equivalent question to ask (there's no reliable per-column signal for
AM/PM the way there is for day-vs-month order). If an AM/PM-sensitive
column matters a lot for a given job, this is worth a manual spot check.
"""

import re
import sys
import difflib
from importlib import import_module

import pandas as pd

text_num = import_module("49_text_number_parsing")

# ---------------------------------------------------------------------------
# 1. Natural-language time normalisation -- rewrites spoken-English
#    time expressions and noisy separators into plain HH:MM(:SS) text,
#    BEFORE the main parser (Section 2, below) ever runs. Every
#    individual step here is a narrow, well-tested text transformation
#    -- none of them ever change what date/time a value represents,
#    only how it's written.
# ---------------------------------------------------------------------------

# Step 1: ordinal day suffixes -- "6th" -> "6", "3rd" -> "3", "30th" -> "30"
_ORDINAL_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)


def _strip_ordinals(text):
    return _ORDINAL_RE.sub(r"\1", text)


# Step 2: an all-colon DATE prefix ("27:01:2022") -- turned into dashes,
# but only when followed by a clean 4-digit year, which is what
# distinguishes this from a time expression (never 4 digits).
_COLON_DATE_RE = re.compile(r"^(\d{1,2}):(\d{1,2}):(\d{4})\b")


def _fix_colon_date_prefix(text):
    return _COLON_DATE_RE.sub(r"\1-\2-\3", text)


# Step 3: hyphen used as the date/time BOUNDARY itself, not just within
# the date -- "10-04-2022-20:18:17" -- insert a space between the year
# and the following HH:MM:SS block.
_HYPHEN_DATE_TIME_BOUNDARY_RE = re.compile(r"(\d{4})-(\d{1,2}:\d{2}:\d{2})")


def _fix_hyphen_date_time_boundary(text):
    return _HYPHEN_DATE_TIME_BOUNDARY_RE.sub(r"\1 \2", text)


# Step 4: a colon used as the date/time BOUNDARY right after the year
# ("2022: 17:20:10", "2022: thirty past 10") -- turned into a space, so
# it's never confused with a colon that's actually part of the time
# itself. Always fires on a 4-digit-year-then-colon, regardless of
# what follows -- the year is unambiguous, so this is never a guess.
_YEAR_COLON_BOUNDARY_RE = re.compile(r"(\d{4}):\s*")


def _fix_year_colon_boundary(text):
    return _YEAR_COLON_BOUNDARY_RE.sub(r"\1 ", text)


# Step 5: fuzzy-correct a misspelled number word (e.g. "forteen" for
# "fourteen") -- a real typo caught in testing, not a hypothetical.
# Deliberately narrow: only ever corrects a WHOLE token against Module
# 49's own number-word vocabulary, EXCLUDING an explicit stoplist of
# common short time/connector words ("to", "mins", "hrs", "secs", ...)
# that are otherwise dangerously close to a number word by edit
# distance alone -- caught in testing when this step, before the
# stoplist existed, corrupted "mins" into "minus" and "to" into "two".
# A cutoff this tight (0.8) plus the stoplist is a deliberately
# conservative combination: better to leave a genuine rare typo
# uncorrected than to silently mangle a correctly-spelled common word.
_NUMBER_WORD_CANDIDATES = sorted(text_num.ALL_NUMBER_VOCAB)
_FUZZY_CORRECTION_STOPLIST = {
    "to", "past", "at", "and", "half", "quarter", "the", "in", "on", "is", "am", "pm",
    "hr", "hrs", "hour", "hours", "min", "mins", "minute", "minutes", "sec", "secs", "second", "seconds",
}


def _fuzzy_correct_number_words(text):
    def _replace(match):
        word = match.group(0)
        word_lower = word.lower()
        if word_lower in text_num.ALL_NUMBER_VOCAB or word_lower in _FUZZY_CORRECTION_STOPLIST:
            return word  # already correct, or a protected common word -- never touch either
        if len(word_lower) < 5:
            return word  # too short for a safe fuzzy match either way
        matches = difflib.get_close_matches(word_lower, _NUMBER_WORD_CANDIDATES, n=1, cutoff=0.8)
        return matches[0] if matches else word
    return re.sub(r"[a-zA-Z']+", _replace, text)


# Step 6: "word-word" HH:MM shorthand -- "fourteen-twenty" -> "14:20",
# "one-thirty" -> "1:30". This works BECAUSE Module 49's number grammar
# correctly REFUSES to treat these as one compound number (a teen
# followed by a tens word, or a units word followed by a tens word,
# isn't valid English) -- that "invalid pair" rejection is exactly the
# signal that this hyphenated pair is actually an hour-minute
# shorthand, not a broken number.
_WORD_PAIR_RE = re.compile(r"\b([a-zA-Z]+)-([a-zA-Z]+)\b")


def _word_to_int(word):
    _, converted, ok = text_num.convert_leading_word_number(word)
    if not ok:
        return None
    try:
        return int(converted)
    except ValueError:
        return None


def _convert_word_time_shorthand(text):
    def _replace(match):
        w1, w2 = match.group(1), match.group(2)
        hour = _word_to_int(w1)
        minute = _word_to_int(w2)
        if hour is not None and minute is not None and 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour}:{minute:02d}"
        return match.group(0)
    return _WORD_PAIR_RE.sub(_replace, text)


# Step 7: "N o'clock" -> "N:00"
_OCLOCK_RE = re.compile(r"\b(\d{1,2})\s*o'?\s*clock\b", re.IGNORECASE)


def _fix_oclock(text):
    return _OCLOCK_RE.sub(lambda m: f"{int(m.group(1))}:00", text)


# Step 8: "half past N" / "quarter past N" / "quarter to N" / "N past M"
# / "N minutes to M" -- all converted to plain HH:MM.
_HALF_PAST_RE = re.compile(r"\bhalf\s+past\s+(\d{1,2})\b", re.IGNORECASE)
_QUARTER_PAST_RE = re.compile(r"\bquarter\s+past\s+(\d{1,2})\b", re.IGNORECASE)
_QUARTER_TO_RE = re.compile(r"\bquarter\s+to\s+(\d{1,2})\b", re.IGNORECASE)
_N_PAST_RE = re.compile(r"\b(\d{1,2})\s+past\s+(\d{1,2})\b", re.IGNORECASE)
_N_MINUTES_TO_RE = re.compile(r"\b(\d{1,2})\s*(?:minutes?)?\s+to\s+(\d{1,2})(?::00)?\b", re.IGNORECASE)


def _prev_hour(hour):
    return 12 if hour <= 1 else hour - 1


def _fix_relative_time_phrases(text):
    text = _HALF_PAST_RE.sub(lambda m: f"{int(m.group(1))}:30", text)
    text = _QUARTER_PAST_RE.sub(lambda m: f"{int(m.group(1))}:15", text)
    text = _QUARTER_TO_RE.sub(lambda m: f"{_prev_hour(int(m.group(1)))}:45", text)
    text = _N_PAST_RE.sub(lambda m: f"{int(m.group(2))}:{int(m.group(1)):02d}", text)
    text = _N_MINUTES_TO_RE.sub(
        lambda m: f"{_prev_hour(int(m.group(2)))}:{60 - int(m.group(1)):02d}", text
    )
    return text


# Step 9: "N hrs M mins [S secs]" word-suffix time (also accepts the
# full words "hour(s)"/"minute(s)"/"second(s)") -- "13 hrs 15 mins" ->
# "13:15", "10hrs: 15mins:20 secs" -> "10:15:20".
_HMS_WORDS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:hrs?|hours?)\b(?:\s*[:,]?\s*(\d{1,2})\s*(?:mins?|minutes?)\b"
    r"(?:\s*[:,]?\s*(\d{1,2})\s*(?:secs?|seconds?)\b)?)?",
    re.IGNORECASE,
)


def _fix_hms_words(text):
    def _replace(match):
        h, m, s = match.groups()
        if m is None:
            return match.group(0)
        result = f"{int(h)}:{int(m):02d}"
        if s is not None:
            result += f":{int(s):02d}"
        return result
    return _HMS_WORDS_RE.sub(_replace, text)


# Step 10: AM/PM punctuation/spacing normalisation -- "p.m.", "P.M",
# "a.m" all become a plain "PM"/"AM" pandas recognises reliably.
_AMPM_RE = re.compile(r"\b([aApP])\.?\s*[mM]\.?\b")


def _fix_ampm(text):
    return _AMPM_RE.sub(lambda m: m.group(1).upper() + "M", text)


# Step 11: leftover noisy colon spacing -- "12 :30" -> "12:30",
# "17: 20" -> "17:20" (anything not already resolved by the more
# specific steps above).
def _fix_colon_spacing(text):
    text = re.sub(r"(\d)\s+:\s*(\d)", r"\1:\2", text)
    text = re.sub(r":\s+(\d)", r":\1", text)
    return text


def normalize_natural_language_datetime(raw):
    """
    Runs all eleven steps above, in the order that makes each one
    reliable (e.g. ordinal-stripping and date/time-boundary fixes
    happen before anything touches the time portion; number-word
    conversion happens before anything that expects digits already).
    A no-op for anything that isn't a string (NaN, already-parsed
    datetimes, etc.) -- returned completely unchanged.
    """
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    text = _strip_ordinals(text)
    text = _fix_colon_date_prefix(text)
    text = _fix_hyphen_date_time_boundary(text)
    text = _fix_year_colon_boundary(text)
    text = text.replace(";", ",")
    text = _fuzzy_correct_number_words(text)
    text = _convert_word_time_shorthand(text)
    text = text_num.convert_all_word_numbers_in_text(text)
    text = _fix_oclock(text)
    text = _fix_relative_time_phrases(text)
    text = _fix_hms_words(text)
    text = _fix_ampm(text)
    text = _fix_colon_spacing(text)
    return text


# ---------------------------------------------------------------------------
# 2. Main parsing entry point
# ---------------------------------------------------------------------------

# Matches a value that STARTS with YYYY-M-D (1 or 2 digit month/day) --
# unambiguous ISO order, regardless of what follows (a time component,
# trailing text, etc.).
_ISO_ORDER_RE = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}")


def standardise_date_column(
    df: pd.DataFrame,
    column: str,
    dayfirst: bool = True,
    output_format: str = "%Y-%m-%d",
) -> tuple[pd.DataFrame, int]:
    """
    Parse a column of inconsistently formatted date (or date+time)
    strings and rewrite it into ONE standard format.

    Returns a tuple: (the updated DataFrame, number of genuine parse
    failures). A "genuine parse failure" is a value that WAS present
    in the original data but couldn't be understood as a date at all
    -- this is different from a value that was already missing before
    we even started (those stay as missing, not counted as failures).

    format="mixed" tells pandas to interpret each value independently
    rather than assuming the whole column shares one format -- exactly
    what we need, since every row may use a different format.

    ISO-ordered values (e.g. "2022-08-07", always Year-Month-Day) are
    parsed SEPARATELY from everything else, always with dayfirst=False
    -- see the module docstring's bug note for why: pandas' own
    dayfirst handling doesn't stay confined to genuinely ambiguous
    values otherwise, and was silently corrupting unambiguous ISO
    dates. Every other format still goes through the normal
    dayfirst-dependent path, unaffected by this fix.
    """
    df = df.copy()

    was_present_before = df[column].notna()

    # Rewrite spoken-English times and noisy separators into plain
    # HH:MM(:SS) text BEFORE parsing -- see normalize_natural_
    # language_datetime's own docstring for the full list of what this
    # handles. Purely a text transformation; never changes what
    # date/time a value actually represents.
    cleaned = df[column].apply(normalize_natural_language_datetime)

    text_form = cleaned.astype(str)
    iso_mask = text_form.str.match(_ISO_ORDER_RE) & was_present_before

    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(
            cleaned.loc[iso_mask], dayfirst=False, format="mixed", errors="coerce"
        )

    remaining_mask = ~iso_mask
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(
            cleaned.loc[remaining_mask], dayfirst=dayfirst, format="mixed", errors="coerce"
        )

    # A row counts as a genuine parse failure only if it HAD a value
    # before, but that value couldn't be turned into a real date.
    parse_failed = was_present_before & parsed.isna()
    n_failures = int(parse_failed.sum())

    # .dt.strftime() automatically leaves missing/failed values as NaN
    # in the output -- we don't need to handle that case separately.
    df[column] = parsed.dt.strftime(output_format)

    return df, n_failures


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")

    print("=== Regression test: the exact real-file bug reports (earlier fix) ===")
    test_df = pd.DataFrame({
        "Delivery DateTime": [
            "20-11-2020: 7:20:40",   # noisy colon separator -- used to fail entirely
            "2021-07-27 00:00:00",   # standard ISO -- used to parse fine already
            "Aug 12, 2022",          # month-name -- unambiguous, used to parse fine already
            "2001-05-25 00:00:00",   # ISO
            "2022-08-07 00:00:00",   # ISO -- used to be SILENTLY CORRUPTED to 2022-07-08 by dayfirst=True
        ],
    })
    out, n_fail = standardise_date_column(
        test_df, "Delivery DateTime", dayfirst=True, output_format="%Y-%m-%d %H:%M:%S"
    )
    print(out)
    print(f"Genuine parse failures: {n_fail} (expect 0)\n")

    print("=== Regression test: genuine dayfirst ambiguity still resolves correctly ===")
    ambiguous_df = pd.DataFrame({"d": ["03/04/2026", "25/11/2023", "5 July 2022"]})
    out_dayfirst_true, _ = standardise_date_column(ambiguous_df.copy(), "d", dayfirst=True)
    out_dayfirst_false, _ = standardise_date_column(ambiguous_df.copy(), "d", dayfirst=False)
    print("dayfirst=True: ", out_dayfirst_true["d"].tolist())
    print("dayfirst=False:", out_dayfirst_false["d"].tolist())

    print("\n=== NEW: real client-file test -- 19 spoken-English / noisy-separator values ===")
    natural_language_df = pd.DataFrame({
        "Purchase DateTime": [
            "27:01:2022: 17: 20:10",
            "6th February, 2022, fourteen-twenty",
            "20/06/2022  5:20 p.m.",
            "07/03/2022, forteen hours 30 mins",
            "24/03/2022: 10hrs: 15mins:20 secs",
            "28 January 2022, 2 o'clock",
            "19 June 2022, 12 :30 P.M.",
            "10 feb 2022, one-thirty p.m",
            "Feb 5, 2022, 14:30:25",
            "08/03/2022, thirteen hrs fifteen mins",
            "18/06/2022, 09:15:20",
            "March 4 2022, fifteen to four pm",
            "27 Jul 2022, 15:00:05",
            "05/03/2022: 16:05:25",
            "26 Jul 2022: thirty past 10",
            "3rd March 2022, half past three",
            "June 20 2022; 11:10 A.M.",
            "10-04-2022-20:18:17",
            "30th July 2022, 15 minutes to nine o'clock",
        ],
    })
    out2, n_fail2 = standardise_date_column(
        natural_language_df, "Purchase DateTime", dayfirst=True, output_format="%Y-%m-%d %H:%M:%S"
    )
    for before, after in zip(natural_language_df["Purchase DateTime"], out2["Purchase DateTime"]):
        print(f"  {before!r:50} -> {after}")
    print(f"\nGenuine parse failures: {n_fail2} (expect 0 -- every one of these is actually recoverable)")

    print("\n=== Full pipeline smoke test (dirty_customer_data.xlsx, if present) ===")
    try:
        filepath = "dirty_customer_data.xlsx"
        df = ingestion.load_data(filepath)

        df, fail_signup = standardise_date_column(df, "signup_date", dayfirst=True)
        df, fail_last_order = standardise_date_column(df, "last_order_date", dayfirst=True)
        df, fail_timestamp = standardise_date_column(
            df, "transaction_timestamp", dayfirst=True, output_format="%Y-%m-%d %H:%M:%S"
        )
        print(f"  signup_date failures: {fail_signup}")
        print(f"  last_order_date failures: {fail_last_order}")
        print(f"  transaction_timestamp failures: {fail_timestamp}")
    except FileNotFoundError:
        print("(dirty_customer_data.xlsx not present in this environment -- skipped)")
