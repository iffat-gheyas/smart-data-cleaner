"""
Module: Duration/Age Standardisation -- Detection & Parsing
--------------------------------------------------------------
Sibling to Modules 45 (weight/length) and 47 (currency). Handles
DURATION values -- "5 days", "3 days 3 hours", "5 years", bare ages
like "42" in an Age column -- explicitly NOT datetime timestamps
("2024-01-15", "March 3rd"). A duration is an ELAPSED AMOUNT of time
(how long, how old), not a POINT in time -- Module 4
(datetime_standardisation) already handles the latter; this module
would have nothing useful to say about an actual date and doesn't try.

Same three-part design as Module 45, built from the start with the
per-value + default-gate approach (rather than retrofitted onto an
older shared-question design, the way weight/length and currency both
were): parse_duration_value() for single cells, detect_duration_
columns() for column-level candidates, find_unresolved_duration_
details() for case-by-case resolution with fuzzy-typo suggestions.

CONVERSION RATIOS, stated honestly: seconds/minutes/hours/days/weeks
convert exactly (60 seconds always makes a minute). Months and years
do NOT have a single exact length -- a month is 28-31 days, a year is
365 or 366. This module uses the standard AVERAGE Gregorian values
(1 year = 365.25 days, 1 month = 1 year / 12) for converting between
these units, same "state the approximation rather than hide it"
principle as Module 45's ton/tonne ambiguity note. This is more than
precise enough for typical data-cleaning purposes (e.g. converting a
mix of "2 years" and "18 months" into one consistent unit), but isn't
exact calendar arithmetic against real dates.
"""

import re
import difflib

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Duration unit alias tables
# ---------------------------------------------------------------------------

DURATION_UNIT_ALIASES = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "hr": "hr", "hrs": "hr", "hour": "hr", "hours": "hr",
    "d": "day", "day": "day", "days": "day",
    "wk": "week", "wks": "week", "week": "week", "weeks": "week",
    "mo": "month", "mos": "month", "month": "month", "months": "month",
    "yr": "year", "yrs": "year", "year": "year", "years": "year",
}

# NOTE on ambiguity, same principle as Module 45's ton/tonne note:
# - Bare "m" is deliberately NOT an alias for anything here -- it could
#   mean "minute" or "month" with no way to tell from the text alone.
#   Only the fuller forms ("min"/"mo" and their variants) are
#   recognised, avoiding a silent wrong guess between the two. Genuinely
#   ambiguous tokens like this are resolved through the per-TOKEN
#   clarification question (see find_unresolved_duration_tokens below),
#   asked ONCE per distinct token, not once per value it appears in.
# - Bare "h" is also excluded for the same reason -- reads too easily
#   as a stray single letter rather than confidently meaning "hour" in
#   arbitrary data; "hr"/"hrs"/"hour"/"hours" are all still recognised.
# - Bare "d" IS included (added after real-file testing), unlike "m"/
#   "h" -- there's no other duration unit "d" could plausibly mean, so
#   there's no genuine ambiguity to protect against here.
# - Month/year conversions are AVERAGES, not exact calendar lengths --
#   see the module docstring.

DURATION_UNITS = set(DURATION_UNIT_ALIASES.values())

# Fixed conversion ratios to a common base (seconds) -- exact for
# s/min/hr/day/week; averages for month/year (see module docstring).
DURATION_UNIT_TO_SECONDS = {
    "s": 1,
    "min": 60,
    "hr": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2629800,     # 365.25 days / 12
    "year": 31557600,     # 365.25 days
}

# Shared "I don't know, treat as missing" sentinel -- same concept as
# Module 45's unit_std.TREAT_AS_MISSING, kept as its own definition
# here rather than importing Module 45 just for this one constant
# (this module has no other dependency on weight/length at all).
TREAT_AS_MISSING = "__treat_as_missing__"
LEAVE_AS_IS = "__leave_as_is__"

# Column-NAME keyword signals, same STRONG/WEAK split as Modules 45/47.
# "age" is treated as a STRONG duration signal per explicit request --
# an age IS a duration (elapsed time since birth), just one that's
# almost always expressed as a bare number with an implied "years".
STRONG_DURATION_NAME_KEYWORDS = ["duration", "age", "tenure", "lifespan"]
WEAK_DURATION_NAME_KEYWORDS = ["period", "term", "experience", "elapsed", "service"]


# ---------------------------------------------------------------------------
# 2. Single-value quantity parser
# ---------------------------------------------------------------------------

_LEADING_NUMBER_RE = re.compile(r"^\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*(.*)$")

# Finds EVERY (number, unit-word) pair anywhere in a value's text --
# not just one, unlike the old 2-part-only compound parser this
# replaced. Applied across the WHOLE text (not just a "remainder"
# after a leading number), so "30 hrs 30mins 50 sec" correctly yields
# THREE pairs, not just the first two. Tolerates zero whitespace
# between a number and its unit ("30mins"), matching real messy data.
_NUMBER_UNIT_PAIR_RE = re.compile(r"(-?\d+(?:,\d+)*(?:\.\d+)?)\s*([a-zA-Z]+)")


def _extract_number_unit_pairs(text: str) -> list:
    pairs = []
    for m in _NUMBER_UNIT_PAIR_RE.finditer(text):
        num_str, unit_str = m.groups()
        try:
            num = float(num_str.replace(",", ""))
        except ValueError:
            continue
        pairs.append((num, unit_str.lower()))
    return pairs


def parse_duration_value(raw, token_overrides: dict = None) -> tuple:
    """
    Extracts an (amount, unit_code, unresolved_tokens) triple from a
    single cell -- same general shape as Module 45's parse_quantity(),
    with one difference worth being explicit about: on FAILURE, the
    third element is a TUPLE of every unresolved unit token found (not
    a single string), since a compound value can have more than one
    (e.g. "5 m 6 x" has two: "m" and "x"). On SUCCESS or for a bare
    number, the third element is None.

      "5 days"           -> (5.0, "day", None)
      "3 days 3 hours"   -> (3.125, "day", None)      -- 2-part compound
      "30 hrs 30mins 50 sec" -> (30.51.., "hr", None) -- 3-PART compound,
                                the bug this rewrite fixes: the old
                                parser only ever handled exactly 2 parts
      "42"               -> (42.0, None, None)         -- bare, no unit
      "5 fortnights"     -> (5.0, None, ("fortnights",))  -- one
                                unresolved token
      "5 m 6 d"          -> (5.0, None, ("m",))         -- "d" resolves
                                fine on its own now ("d" was added as an
                                unambiguous day alias); only "m" (still
                                genuinely ambiguous between minute/
                                month) is reported as unresolved
      "N/A"              -> (None, None, None)

    `token_overrides`: optional {lowercase_token: canonical_unit} --
    this is how a client's answer to "what does 'm' mean here?"
    (Module 11's per-token clarification question, asked ONCE per
    distinct unrecognised token rather than once per value) gets
    applied during real conversion. A token mapped to TREAT_AS_MISSING
    means "the client explicitly said this token's meaning can't be
    known" -- any value containing it comes back with TREAT_AS_MISSING
    included in the unresolved-tokens tuple, and Module 51 (the apply
    step) treats the WHOLE value as missing when that's present, since
    a duration can't be safely computed with one of its components
    undefined.
    """
    if pd.isna(raw):
        return (None, None, None)

    if isinstance(raw, (int, float)):
        return (float(raw), None, None)

    text = str(raw).strip()
    if text == "":
        return (None, None, None)

    text = re.sub(r"[()\[\]]", "", text).strip()  # same bracket-stripping fix as Module 47
    if text == "":
        return (None, None, None)

    match = _LEADING_NUMBER_RE.match(text)
    if not match:
        return (None, None, None)

    number_str, remainder = match.groups()
    try:
        number = float(number_str.replace(",", ""))
    except ValueError:
        return (None, None, None)

    remainder = remainder.strip()
    if not remainder:
        return (number, None, None)  # bare number, no unit text at all

    pairs = _extract_number_unit_pairs(text)
    if not pairs:
        # Remainder present but nothing letter-shaped enough to be a
        # unit word (e.g. stray punctuation) -- nothing recoverable,
        # treated the same as a bare number rather than inventing a
        # "problem" out of noise.
        return (number, None, None)

    resolved = []       # list of (num, canonical_unit), in the order found
    unresolved = []     # unique unresolved tokens, in first-seen order (may include TREAT_AS_MISSING / LEAVE_AS_IS)

    for num, tok in pairs:
        canonical = DURATION_UNIT_ALIASES.get(tok)
        if canonical is None and token_overrides:
            override = token_overrides.get(tok)
            if override in (TREAT_AS_MISSING, LEAVE_AS_IS):
                if override not in unresolved:
                    unresolved.append(override)
                continue
            elif override:
                canonical = override
        if canonical is None:
            if tok not in unresolved:
                unresolved.append(tok)
        else:
            resolved.append((num, canonical))

    if unresolved:
        return (number, None, tuple(unresolved))

    total_seconds = sum(n * DURATION_UNIT_TO_SECONDS[u] for n, u in resolved)
    first_unit = resolved[0][1]
    combined = total_seconds / DURATION_UNIT_TO_SECONDS[first_unit]
    return (combined, first_unit, None)


# ---------------------------------------------------------------------------
# 3. Column-level detection
# ---------------------------------------------------------------------------

def _name_signal(col_name: str) -> dict:
    name_lower = col_name.lower()
    strong = any(kw in name_lower for kw in STRONG_DURATION_NAME_KEYWORDS)
    weak = any(kw in name_lower for kw in WEAK_DURATION_NAME_KEYWORDS)

    implied_unit = None
    for token, code in DURATION_UNIT_ALIASES.items():
        if len(token) < 2:
            continue  # skip bare 1-char tokens for name matching -- too noisy
        if re.search(rf"(?:^|[_\s(]){re.escape(token)}(?:$|[_\s)])", name_lower):
            implied_unit = code
            break

    return {"strong": strong, "weak": weak, "implied_unit_from_name": implied_unit}


def detect_duration_columns(df: pd.DataFrame, sample_size: int = 200) -> dict:
    """
    Returns {column_name: info_dict}, same shape and thresholds as
    Modules 45/47's detection. Deliberately does NOT trigger on
    anything that looks like an actual date/timestamp -- this only
    fires on values that parse as a NUMBER (optionally with a duration
    unit), never on date-formatted text, so a real date column and a
    duration column don't compete for the same values.
    """
    results = {}

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        sample = sample.sample(n=min(sample_size, len(sample)), random_state=0) if len(sample) > sample_size else sample

        parsed = [parse_duration_value(v) for v in sample]
        n = len(parsed)

        numbers = [p[0] for p in parsed]
        units = [p[1] for p in parsed]

        numeric_rate = sum(nm is not None for nm in numbers) / n
        duration_hits = [u for u in units if u in DURATION_UNITS]
        duration_rate = len(duration_hits) / n

        bare_number_count = sum(
            nm is not None and unit is None and tok is None
            for nm, unit, tok in parsed
        )
        bare_number_examples = [
            raw for raw, (nm, unit, tok) in zip(sample, parsed)
            if nm is not None and unit is None and tok is None
        ][:5]
        unrecognised_unit_samples = [
            raw for raw, (nm, unit, tok) in zip(sample, parsed)
            if nm is not None and tok is not None and unit is None
        ][:5]

        name_sig = _name_signal(str(col))

        measurement_confidence = 0.0
        is_duration_column = False

        if duration_rate >= 0.3:
            is_duration_column = True
            measurement_confidence = 0.95 if duration_rate >= 0.6 else 0.75
        elif numeric_rate >= 0.8 and name_sig["strong"]:
            is_duration_column = True
            measurement_confidence = 0.6
        elif numeric_rate >= 0.8 and name_sig["weak"]:
            is_duration_column = True
            measurement_confidence = 0.35

        if not is_duration_column:
            continue

        detected_units = {}
        for u in duration_hits:
            detected_units[u] = detected_units.get(u, 0) + 1

        results[col] = {
            "confidence": round(measurement_confidence, 2),
            "detected_units": detected_units,
            "implied_unit_from_name": name_sig["implied_unit_from_name"],
            "bare_number_count": bare_number_count,
            "bare_number_examples": bare_number_examples,
            "unrecognised_unit_samples": unrecognised_unit_samples,
            "n_sampled": n,
        }

    return results


def needs_standardization(info: dict) -> bool:
    """Identical reasoning to Module 45's function of the same name."""
    return bool(info["detected_units"]) or (
        info["bare_number_count"] > 0 and not info.get("implied_unit_from_name")
    )


# ---------------------------------------------------------------------------
# 4. Per-value BARE-number detail (for the default-gate mechanism)
# ---------------------------------------------------------------------------

_FUZZY_DURATION_CANDIDATES = sorted({token for token in DURATION_UNIT_ALIASES if len(token) >= 3})


def _suggest_likely_unit(raw_token: str):
    """Fuzzy-match suggestion for an unrecognised duration unit, e.g. "yeras" -> "year"."""
    if not raw_token or len(raw_token) < 3:
        return None
    matches = difflib.get_close_matches(raw_token.lower(), _FUZZY_DURATION_CANDIDATES, n=1, cutoff=0.6)
    if not matches:
        return None
    return DURATION_UNIT_ALIASES[matches[0]]


def find_unresolved_duration_details(df: pd.DataFrame, col: str, info: dict) -> list:
    """
    Enumerates every DISTINCT BARE value in `col` -- a number with NO
    unit text at all (e.g. "42") -- used by the default-unit gate and
    its accompanying per-value resolution (Module 11's
    generate_duration_default_gate_questions / generate_duration_
    value_resolution_questions).

    REDESIGNED after client feedback: this used to ALSO cover values
    with an unrecognised UNIT TOKEN (e.g. "5 fortnights", "5 m 6 d"),
    asking about each whole DISTINCT VALUE separately. That's
    inefficient for compound values specifically -- a column can have
    many different values that all share the SAME ambiguous token
    (e.g. "5 m", "10 m 2 d", "3 m 1 yr" all contain "m"), and asking
    once "what does 'm' mean" resolves every one of them at once. That
    case is now handled separately, at the TOKEN level, by
    find_unresolved_duration_tokens below -- this function only ever
    returns genuinely bare values now.
    """
    implied_unit = info.get("implied_unit_from_name")
    if implied_unit:
        return []

    series = df[col]
    seen = {}

    for idx, raw in series.items():
        if pd.isna(raw):
            continue
        number, canonical_unit, raw_token = parse_duration_value(raw)
        if number is None or canonical_unit is not None:
            continue
        if raw_token is not None:
            continue  # has unresolved TOKEN(s) -- handled by find_unresolved_duration_tokens instead

        key = str(raw)
        if key not in seen:
            example_row = df.loc[idx].to_dict()
            seen[key] = {
                "raw_value": raw,
                "count": 0,
                "example_row_index": idx,
                "example_row": example_row,
                "suggested_unit": None,
                "is_bare": True,
            }
        seen[key]["count"] += 1

    return list(seen.values())


# ---------------------------------------------------------------------------
# 5. Per-TOKEN unresolved-unit detail (the client's requested redesign)
# ---------------------------------------------------------------------------

def find_unresolved_duration_tokens(df: pd.DataFrame, col: str, info: dict) -> list:
    """
    Enumerates every DISTINCT unrecognised UNIT TOKEN appearing
    anywhere in `col` -- e.g. if the column contains "5 m 6 d", "10 m
    3 yr", "2 m", the token "m" is ONE entry here (not three), since
    asking once "what does 'm' mean" resolves every occurrence of it,
    however many different full values it happens to appear in.

    This is the direct answer to the client's own example: "5 m 6 d"
    -- "d" now resolves on its own (an unambiguous day alias, added
    alongside this feature), so only "m" (genuinely ambiguous between
    minute/month) needs asking about, and it's asked about exactly
    once for the whole column, not once per compound value it's part of.

    Returns a list of dicts: token, count (how many VALUES in the
    column contain this token), example_values (a few raw cells for
    context), suggested_unit (a fuzzy-match guess, e.g. "yeras" ->
    "year", or None).
    """
    implied_unit = info.get("implied_unit_from_name")
    if implied_unit:
        return []

    token_info = {}

    for raw in df[col].dropna():
        _, canonical_unit, raw_token = parse_duration_value(raw)
        if canonical_unit is not None or raw_token is None:
            continue  # either fully resolved, or genuinely bare -- not a token issue

        for tok in raw_token:
            if tok not in token_info:
                token_info[tok] = {
                    "token": tok,
                    "count": 0,
                    "example_values": [],
                    "suggested_unit": _suggest_likely_unit(tok),
                }
            token_info[tok]["count"] += 1
            if raw not in token_info[tok]["example_values"] and len(token_info[tok]["example_values"]) < 3:
                token_info[tok]["example_values"].append(raw)

    return list(token_info.values())


if __name__ == "__main__":
    print("=== parse_duration_value() spot checks ===")
    tests = [
        "5 days", "3 days 3 hours", "5 years", "42", "1 year 6 months",
        "2 weeks", "18 months", "5 fortnights", "N/A", "10hrs", "1,000 days",
        "30 hrs 30mins 50 sec",  # the client's exact 3-part example -- old parser failed on this entirely
        "5 m 6 d",               # the client's exact ambiguous-token example
    ]
    for t in tests:
        print(f"  {t!r:25} -> {parse_duration_value(t)}")

    print("\n=== parse_duration_value() with a resolved token_override ===")
    print("  '5 m 6 d' with m->month:", parse_duration_value("5 m 6 d", token_overrides={"m": "month"}))
    print("  '5 m 6 d' with m->TREAT_AS_MISSING:", parse_duration_value("5 m 6 d", token_overrides={"m": TREAT_AS_MISSING}))

    print("\n=== detect_duration_columns() ===")
    test_df = pd.DataFrame({
        "Age": ["42", "35", "fifty", "28", "3 years", "61", "19", "37"],
        "Tenure": ["3 years", "5 years 2 months", "18 months", "2 yrs", "4 years", "6 years", "1 year", "9 months"],
        "Employee_Name": ["John Smith", "Jane Doe", "Bob Lee", "Sara Khan", "Tom Hill", "Ann Bell", "Joe Park", "Kim Ray"],
        "Salary": [50000, 62000, 48000, 71000, 55000, 49500, 60000, 45000],
    })
    print(test_df)
    detected = detect_duration_columns(test_df)
    for col, info in detected.items():
        print(f"\nColumn: {col}")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print(f"  needs_standardization: {needs_standardization(info)}")

    print("\n=== Columns NOT flagged (should be Employee_Name, Salary) ===")
    print([c for c in test_df.columns if c not in detected])

    print("\n=== find_unresolved_duration_tokens() on a mixed-token column ===")
    token_test_df = pd.DataFrame({
        "Duration": ["5 m 6 d", "10 m 3 yr", "2 m", "30 hrs 30mins 50 sec", "18 months"],
    })
    token_detected = detect_duration_columns(token_test_df)
    tokens = find_unresolved_duration_tokens(token_test_df, "Duration", token_detected["Duration"])
    for t in tokens:
        print(t)
