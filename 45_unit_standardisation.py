"""
Module: Weight & Length Unit Standardisation -- Sub-step A (Detection & Parsing)
--------------------------------------------------------------------------------
Generic-mode feature. This sub-step ONLY does two things:
  1. parse_quantity() -- pull a (number, unit) pair out of a single raw cell,
     e.g. "5.5 kg" -> (5.5, "kg"), "170cm" -> (170.0, "cm"), "42" -> (42.0, None).
  2. detect_measurement_columns() -- scan a whole DataFrame and decide, per
     column, whether it LOOKS like a weight or length measurement, using TWO
     independent signals combined (not just column name, since names like
     "Size" or "Distance" are ambiguous, and not just content, since a column
     of clean numbers with no unit text gives no content signal at all):
       (a) does the COLUMN NAME contain a weight/length keyword?
       (b) do the VALUES actually carry a recognisable weight/length unit token?

  Conversion into a single target unit, and the dynamic "which column is
  this / which unit do you want" questions, are DELIBERATELY NOT built here
  -- that's the next sub-step, once this detection layer is verified against
  real messy data.

SCOPE NOTE (being upfront): this sub-step does NOT parse word-form numbers
("five kilograms"). That's a separate feature (text-to-number parsing) we
agreed to build next. Once both exist, the pipeline will run text-to-number
parsing BEFORE this module, so "five kg" becomes "5 kg" and then parses
normally here. Handling it here too would duplicate that logic.
"""

import re
import difflib
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Unit alias tables -- every text variant a client's spreadsheet might
#    plausibly use, mapped to one canonical short code per unit.
# ---------------------------------------------------------------------------

WEIGHT_UNIT_ALIASES = {
    "kg": "kg", "kgs": "kg", "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
    "g": "g", "gram": "g", "grams": "g", "gm": "g", "gms": "g",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "ton": "tonne", "tons": "tonne", "tonne": "tonne", "tonnes": "tonne", "metric ton": "tonne",
    "stone": "stone", "stones": "stone", "st": "stone",
}

LENGTH_UNIT_ALIASES = {
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "in": "in", "inch": "in", "inches": "in",
    "ft": "ft", "foot": "ft", "feet": "ft",
    "yd": "yd", "yard": "yd", "yards": "yd",
    "mi": "mi", "mile": "mi", "miles": "mi",
}

# NOTE on ambiguity, being explicit rather than silently guessing:
# - "ton"/"tonne" is treated as the metric tonne (1000 kg). The US short
#   ton (907.185 kg) and UK long ton (1016.05 kg) use the SAME word --
#   there is no text cue to tell them apart. Flagged here as a known
#   limitation rather than picked silently; worth a note in the client
#   deliverable if a "ton" value is ever encountered.
# - Bare single-letter "t" is deliberately NOT included as a tonne alias --
#   too easy to collide with an unrelated single-character code in real
#   data. Only full "ton(s)"/"tonne(s)" trigger the match.
# - "st"/"stone" (UK weight unit, e.g. "12 stone", "11 st") IS included
#   despite being a short token -- caught missing entirely during real
#   testing (a client file with British-convention weights produced
#   "unresolved unit" results for every stone value). "st" carries a
#   small collision risk (e.g. an unrelated code that happens to be
#   "st"), judged an acceptable trade-off given how common the unit is
#   for personal weight data specifically.

# ---------------------------------------------------------------------------
# 1b. Fixed conversion ratios -- canonical source of truth. Module 46 (the
#     conversion-application step) imports these rather than keeping its
#     own copy, so there is exactly one place to update if a unit or its
#     ratio ever needs correcting.
# ---------------------------------------------------------------------------

WEIGHT_UNIT_TO_KG = {
    "kg": 1.0,
    "g": 0.001,
    "mg": 0.000001,
    "lb": 0.45359237,
    "oz": 0.028349523125,
    "tonne": 1000.0,
    "stone": 6.35029318,
}

LENGTH_UNIT_TO_M = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mi": 1609.344,
}

WEIGHT_UNITS = set(WEIGHT_UNIT_ALIASES.values())
LENGTH_UNITS = set(LENGTH_UNIT_ALIASES.values())

# Shared sentinel for the "I don't know what unit this is" answer to
# Module 11's unresolved-unit clarification question. Lives here (not in
# Module 11 or 46 individually) so both import the SAME constant rather
# than each hardcoding their own copy of the string, which would risk
# silently drifting apart if one file's copy were ever edited alone.
TREAT_AS_MISSING = "__treat_as_missing__"

# ADDED per explicit client request: previously "I don't know" only
# ever meant "treat as missing" (blank the value out). The client
# wants a genuine choice here -- sometimes "I don't know what unit
# this is" should mean "just leave the original value exactly as
# typed" rather than losing it to a blank cell. Both sentinels are
# handled the same way everywhere they appear: LEAVE_AS_IS keeps the
# original raw cell untouched (no conversion attempted at all),
# TREAT_AS_MISSING blanks it -- see Module 46's apply logic for where
# each is actually applied.
LEAVE_AS_IS = "__leave_as_is__"

# Column-NAME keyword signals. Deliberately kept separate from the unit
# aliases above -- a name containing "size" or "distance" is a WEAK signal
# on its own (see WEAK_NAME_KEYWORDS) because those words are also used for
# entirely non-measurement columns ("Sample_Size", "Family_Size").
STRONG_WEIGHT_NAME_KEYWORDS = ["weight", "mass"]
STRONG_LENGTH_NAME_KEYWORDS = ["length", "height", "width", "depth"]
WEAK_WEIGHT_NAME_KEYWORDS = ["wt", "kg", "lbs", "pound"]
WEAK_LENGTH_NAME_KEYWORDS = ["size", "distance", "dimension", "cm", "meter", "metre", "inch"]


# ---------------------------------------------------------------------------
# 2. Single-value quantity parser
# ---------------------------------------------------------------------------

# CHANGED after real-file testing: this used to require STRICT Western
# 3-digit comma grouping ("65,000" ok, but "5,50,500" -- Indian-style
# lakh/crore grouping -- was NOT, since the first group after the comma
# is 2 digits, not 3). That strictness backfired: instead of flagging
# "5,50,500" as an ambiguous number worth asking about, the old regex
# just silently stopped matching after the first digit and returned "5",
# discarding "50,500" entirely with no error and no flag -- a much worse
# outcome than either interpretation. Per explicit instruction, comma
# groups of ANY size are now accepted and simply stripped, so "5,50,500"
# becomes the number 550500 -- treating the comma purely as a visual
# grouping separator, regardless of which country's convention grouped
# it, rather than validating the grouping pattern at all.
_LEADING_NUMBER_RE = re.compile(r"^\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*(.*)$")


def parse_quantity(raw) -> tuple:
    """
    Extracts a (number, canonical_unit, raw_unit_token) triple from a single
    cell value.

      "5.5 kg"    -> (5.5,   "kg",  "kg")
      "170cm"     -> (170.0, "cm",  "cm")
      "42"        -> (42.0,  None,  None)     -- a bare number, no unit at all
      "13 quid"   -> (13.0,  None,  "quid")   -- a number WITH a unit token,
                                                  but one we don't recognise
      "5 ft 8 in" -> (5.667, "ft",  "ft 8 in") -- COMPOUND value, combined
                                                   into a single number in
                                                   the first-mentioned unit
                                                   (see _try_parse_compound)
      "5'8\""     -> (5.667, "ft",  "5'8\"")    -- QUOTE-MARK feet/inches
                                                   notation, same combination
                                                   as above (see
                                                   _try_parse_feet_inches_quote)
      "N/A"       -> (None,  None,  None)      -- no number found at all

    Keeping canonical_unit and raw_unit_token separate matters: "no unit
    present" (raw_unit_token is None) and "an unrecognised unit was present"
    (raw_unit_token is a string, canonical_unit is None) are different
    situations for the caller -- the first is a normal bare number, the
    second is a real oddity worth surfacing to the client rather than
    silently treating the same as a bare number.

    BUG FIXED after real-file testing: this used to return just the
    LEADING number for a compound value like "5 ft 8 in" -- i.e. it threw
    away the "8 in" entirely and silently reported a height of 5 (as if
    the whole thing were "5 ft"), which is wrong by more than 30cm, not
    just imprecise. Now attempts to combine two same-family units (feet+
    inches being the common real-world case for height) before giving up.
    """
    if pd.isna(raw):
        return (None, None, None)

    if isinstance(raw, (int, float)):
        return (float(raw), None, None)

    text = str(raw).strip()
    if text == "":
        return (None, None, None)

    # Quote-mark feet/inches notation (5'8", 5', 8") is checked FIRST and
    # separately from the generic path below -- the apostrophe sits
    # directly against the number with no unit WORD at all, so the
    # generic letter-based unit lookup and compound parser (both built
    # around alphabetic unit tokens like "ft"/"in") never see it as
    # anything but noise attached to the leading number. Easier and
    # clearer to give it its own dedicated check than to bend the
    # generic regex to also understand punctuation-as-unit.
    quote_result = _try_parse_feet_inches_quote(text)
    if quote_result is not None:
        return quote_result

    match = _LEADING_NUMBER_RE.match(text)
    if not match:
        return (None, None, None)

    number_str, remainder = match.groups()
    try:
        number = float(number_str.replace(",", ""))
    except ValueError:
        return (None, None, None)

    remainder = remainder.strip().lower().rstrip(".")
    if not remainder:
        return (number, None, None)

    canonical = WEIGHT_UNIT_ALIASES.get(remainder) or LENGTH_UNIT_ALIASES.get(remainder)
    if canonical:
        return (number, canonical, remainder)

    # Simple single-unit lookup failed -- try treating this as a COMPOUND
    # value (e.g. remainder "ft 8 in" from the full text "5 ft 8 in")
    # before giving up.
    compound = _try_parse_compound(number, remainder)
    if compound is not None:
        return compound

    return (number, None, remainder)


# ---------------------------------------------------------------------------
# 2a. Quote-mark feet/inches notation (5'8", 5', 8")
# ---------------------------------------------------------------------------

# Full "feet' inches"" -- inches part and closing quote both optional, so
# this also matches feet-only ("5'") and, via the separate regex below,
# inches-only ("8\""). The closing inches mark accepts EITHER a proper
# double-quote (") OR two consecutive apostrophes ('') -- the latter
# caught in real-file testing as a common typo (typing the apostrophe
# key twice instead of reaching for the actual double-quote/inch
# symbol, e.g. a phone/tablet keyboard without an easy " key). Both are
# treated identically -- there's no meaningful difference in what the
# person meant.
_FEET_INCHES_QUOTE_RE = re.compile(r'^(-?\d+(?:\.\d+)?)\s*\'\s*(-?\d+(?:\.\d+)?)?\s*(?:"|\'\')?$')
_INCHES_ONLY_QUOTE_RE = re.compile(r'^(-?\d+(?:\.\d+)?)\s*(?:"|\'\')$')


def _try_parse_feet_inches_quote(text: str):
    """
    Handles the quote-mark height notation: 5'8" (5 feet 8 inches), 5'
    (5 feet, no inches given), 8" (8 inches alone). Combined into a
    single decimal-feet number the same way _try_parse_compound handles
    the written-out "5 ft 8 in" form, so everything downstream (target-
    unit conversion) treats both notations identically.

    Returns None if the text doesn't match either shape at all -- caller
    falls through to the generic parsing path.
    """
    match = _FEET_INCHES_QUOTE_RE.match(text)
    if match:
        feet_str, inches_str = match.groups()
        feet = float(feet_str)
        if inches_str:
            inches = float(inches_str)
            total_feet = feet + (inches / 12.0)
            return (total_feet, "ft", text)
        return (feet, "ft", text)

    match = _INCHES_ONLY_QUOTE_RE.match(text)
    if match:
        return (float(match.group(1)), "in", text)

    return None


# ---------------------------------------------------------------------------
# 2b. Compound-value combination (e.g. "5 ft 8 in", "1kg 200g")
# ---------------------------------------------------------------------------

_COMPOUND_REMAINDER_RE = re.compile(
    r"^([a-zA-Z]+)\.?\s+(-?\d+(?:\.\d+)?)\s*([a-zA-Z]*)\.?$"
)


def _try_parse_compound(first_number: float, remainder: str):
    """
    Handles a SPECIFIC, common compound shape: <number1> <unit1> <number2>
    <unit2>, e.g. "5 ft 8 in" (first_number=5, remainder="ft 8 in").

    Only combines when BOTH unit tokens are explicitly present and BOTH
    resolve to the SAME measurement family (both weight, or both length)
    -- deliberately does NOT guess a second unit when only one is given
    (e.g. "5 ft 8" with no trailing unit stays unresolved, same "don't
    guess" principle as everywhere else in this pipeline).

    Combines by converting both parts to the smaller/base unit's family
    table, summing, then expressing the total back in unit1 (the
    first-mentioned unit) -- e.g. "5 ft 8 in" becomes 5.667, still
    labelled "ft", so the normal single-unit conversion path downstream
    can convert that 5.667 ft to whatever the client's target unit is,
    with no special-casing needed past this point.

    Returns None (not a compound match) if the pattern doesn't apply --
    caller falls back to its existing "unresolved" handling.
    """
    match = _COMPOUND_REMAINDER_RE.match(remainder)
    if not match:
        return None

    unit1_raw, number2_str, unit2_raw = match.groups()
    unit1 = WEIGHT_UNIT_ALIASES.get(unit1_raw) or LENGTH_UNIT_ALIASES.get(unit1_raw)
    unit2 = WEIGHT_UNIT_ALIASES.get(unit2_raw) or LENGTH_UNIT_ALIASES.get(unit2_raw) if unit2_raw else None

    if unit1 is None or unit2 is None:
        return None  # need BOTH units explicit and recognised -- no guessing

    same_family = (unit1 in WEIGHT_UNITS and unit2 in WEIGHT_UNITS) or \
                  (unit1 in LENGTH_UNITS and unit2 in LENGTH_UNITS)
    if not same_family:
        return None

    factor_table = WEIGHT_UNIT_TO_KG if unit1 in WEIGHT_UNITS else LENGTH_UNIT_TO_M
    number2 = float(number2_str)

    total_base = first_number * factor_table[unit1] + number2 * factor_table[unit2]
    combined_in_unit1 = total_base / factor_table[unit1]

    return (combined_in_unit1, unit1, remainder)


# ---------------------------------------------------------------------------
# 3. Column-level detection
# ---------------------------------------------------------------------------

def _name_signal(col_name: str) -> dict:
    name_lower = col_name.lower()
    strong_weight = any(kw in name_lower for kw in STRONG_WEIGHT_NAME_KEYWORDS)
    strong_length = any(kw in name_lower for kw in STRONG_LENGTH_NAME_KEYWORDS)
    weak_weight = any(kw in name_lower for kw in WEAK_WEIGHT_NAME_KEYWORDS)
    weak_length = any(kw in name_lower for kw in WEAK_LENGTH_NAME_KEYWORDS)

    # A unit token embedded directly in the column name itself (e.g.
    # "Weight_kg", "Height_cm") tells us the IMPLIED unit for every bare
    # number in that column -- captured separately so the next sub-step
    # doesn't have to re-derive it.
    implied_unit = None
    for token, canonical in {**WEIGHT_UNIT_ALIASES, **LENGTH_UNIT_ALIASES}.items():
        if re.search(rf"(?:^|[_\s]){re.escape(token)}(?:$|[_\s])", name_lower):
            implied_unit = canonical
            break

    return {
        "strong_weight": strong_weight,
        "strong_length": strong_length,
        "weak_weight": weak_weight,
        "weak_length": weak_length,
        "implied_unit_from_name": implied_unit,
    }


def detect_measurement_columns(df: pd.DataFrame, sample_size: int = 200) -> dict:
    """
    Returns {column_name: info_dict} but ONLY for columns that were flagged
    as a plausible weight or length measurement -- ordinary columns
    (customer names, IDs, prices, ordinary counts) are simply absent from
    the returned dict, same "don't surface what isn't relevant" principle
    as the rest of the profiling layer.

    info_dict contains:
      measurement_type      "weight" or "length"
      confidence             0.0-1.0 (see thresholds below)
      detected_units         {unit: count} from values that DID carry a
                              recognisable unit token
      implied_unit_from_name unit implied by the column name itself, or None
      bare_number_count       values that were numeric with NO unit token
      bare_number_examples    a few raw examples of those bare numbers --
                              used by Module 11's clarification question
                              when the column's unit can't otherwise be
                              inferred (no name-implied unit, no majority
                              unit in the content)
      unrecognised_unit_samples  a few raw examples where a unit-like token
                              was present but not one we recognise (e.g.
                              "12 furlongs") -- worth showing the client
      n_sampled
    """
    results = {}

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        sample = sample.sample(n=min(sample_size, len(sample)), random_state=0) if len(sample) > sample_size else sample

        parsed = [parse_quantity(v) for v in sample]
        n = len(parsed)

        numbers = [p[0] for p in parsed]
        units = [p[1] for p in parsed]
        raw_tokens = [p[2] for p in parsed]

        numeric_rate = sum(nm is not None for nm in numbers) / n

        weight_hits = [u for u in units if u in WEIGHT_UNITS]
        length_hits = [u for u in units if u in LENGTH_UNITS]
        weight_unit_rate = len(weight_hits) / n
        length_unit_rate = len(length_hits) / n

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

        measurement_type = None
        confidence = 0.0

        # --- Decision order, strongest evidence first ---
        if weight_unit_rate >= 0.3 and weight_unit_rate >= length_unit_rate:
            measurement_type = "weight"
            confidence = 0.95 if weight_unit_rate >= 0.6 else 0.75
        elif length_unit_rate >= 0.3:
            measurement_type = "length"
            confidence = 0.95 if length_unit_rate >= 0.6 else 0.75
        elif numeric_rate >= 0.8 and (name_sig["strong_weight"] or name_sig["strong_length"]):
            # Purely numeric column, but the NAME strongly suggests a
            # measurement (e.g. "Weight" with clean numbers, no unit text
            # anywhere -- likely a single implied unit throughout).
            measurement_type = "weight" if name_sig["strong_weight"] else "length"
            confidence = 0.6
        elif numeric_rate >= 0.8 and (name_sig["weak_weight"] or name_sig["weak_length"]):
            # Weak name signal only (e.g. "Size", "Distance") -- still
            # worth asking about, but with noticeably lower confidence,
            # since these words often mean something else entirely.
            measurement_type = "weight" if name_sig["weak_weight"] else "length"
            confidence = 0.35

        if measurement_type is None:
            continue

        detected_units = {}
        for u in (weight_hits if measurement_type == "weight" else length_hits):
            detected_units[u] = detected_units.get(u, 0) + 1

        results[col] = {
            "measurement_type": measurement_type,
            "confidence": round(confidence, 2),
            "detected_units": detected_units,
            "implied_unit_from_name": name_sig["implied_unit_from_name"],
            "bare_number_count": bare_number_count,
            "bare_number_examples": bare_number_examples,
            "unrecognised_unit_samples": unrecognised_unit_samples,
            "n_sampled": n,
        }

    return results


def needs_standardization(info: dict) -> bool:
    """
    Shared trigger condition -- lives here (not duplicated in Module 11
    and Module 46 separately) since both need the EXACT same answer to
    "does this column actually need any value-rewriting work": Module 11
    uses it to decide whether to ask the standardize-gate question at
    all, Module 46 uses it to decide whether the absence of an explicit
    answer should be treated as "nothing to do" rather than "do it
    anyway" (see apply_measurement_unit_conversion).

    True when either:
      - at least one value carries an actual unit token in the text
        (len(detected_units) > 0) -- covers mixed units AND the "single
        unit written as text" case (e.g. every value is "2kg"), since
        both need the unit text stripped out to become numeric
      - OR there are bare numbers with no name-implied unit to fall
        back on -- ambiguous, needs the unresolved-unit question

    False for a column that's already plain numeric with a name-implied
    unit and nothing else going on (e.g. "Height_cm" with clean numbers
    throughout) -- genuinely nothing to ask about or convert.
    """
    return bool(info["detected_units"]) or (
        info["bare_number_count"] > 0 and not info.get("implied_unit_from_name")
    )


def suggest_default_unit_by_magnitude(df: pd.DataFrame, col: str, info: dict):
    """
    A SMARTER default-unit suggestion for the has_default_unit__<col> /
    default_unit__<col> questions, replacing a plain "most common unit
    overall" pick -- added after real-file testing found a real
    accuracy problem with that simpler approach.

    THE PROBLEM CAUGHT: a Height column with mostly cm/ft/in values
    plus a couple of bare decimals like "1.7" and "1.8" -- and ALSO a
    couple of "1.60 meters"/"1.9 meters" entries at the SAME magnitude
    -- defaulted the bare values to "cm" (the column's overall most
    common unit), producing an absurd "1.7 cm" height. The bare values
    are obviously meant to be METERS, matching the magnitude of the
    column's OTHER metre-labelled entries, not whatever unit happens
    to be most frequent across the whole column.

    THE FIX: instead of just counting units, this converts every
    RECOGNISED value in the column to a common base unit (kg for
    weight, m for length) to find the column's typical magnitude, then
    checks which candidate unit would make the BARE values' own
    typical magnitude land closest to that same range -- "1.7" lands
    right in the metres cluster (~1.6-1.9), nowhere near the
    centimetres cluster (~160-190), so metres wins.

    Compares on a LOG scale specifically because these units differ by
    orders of magnitude (cm vs m is a 100x difference) -- a plain
    difference would make the giant cm/kg numbers dominate the
    comparison regardless of which unit actually fits better.

    Returns the suggested unit code, or None if there isn't enough
    information to do better than the plain frequency-based fallback
    (e.g. no bare values at all, or no recognised values to compare
    against) -- callers should fall back to their own simpler default
    in that case, not treat None as an error.
    """
    import math

    measurement_type = info["measurement_type"]
    conversion_table = WEIGHT_UNIT_TO_KG if measurement_type == "weight" else LENGTH_UNIT_TO_M
    implied_unit = info.get("implied_unit_from_name")

    if implied_unit:
        return None  # bare values already resolved via the name -- nothing to suggest

    bare_numbers = []
    recognised_base_values = []

    for raw in df[col].dropna():
        number, canonical_unit, raw_token = parse_quantity(raw)
        if number is None:
            continue
        if canonical_unit is None and raw_token is None:
            bare_numbers.append(number)
        elif canonical_unit is not None:
            recognised_base_values.append(number * conversion_table[canonical_unit])

    if not bare_numbers or not recognised_base_values:
        return None  # nothing to compare against -- let the caller use its plain fallback

    # Medians rather than means -- a single extreme outlier elsewhere
    # in the column shouldn't skew what "typical magnitude" means here.
    bare_numbers.sort()
    recognised_base_values.sort()
    typical_bare = bare_numbers[len(bare_numbers) // 2]
    typical_base = recognised_base_values[len(recognised_base_values) // 2]

    if typical_bare <= 0 or typical_base <= 0:
        return None  # log-scale comparison needs positive values

    best_unit = None
    best_distance = None
    for candidate_unit, factor in conversion_table.items():
        candidate_as_base = typical_bare * factor
        distance = abs(math.log(candidate_as_base / typical_base))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_unit = candidate_unit

    return best_unit


# ---------------------------------------------------------------------------
# 4. Per-value unresolved-unit detail (for case-by-case resolution)
# ---------------------------------------------------------------------------
# Mirrors Module 47's equivalent section exactly, added for consistency
# after the currency feature was redesigned this way (client feedback:
# one shared "what unit are these ambiguous values in" question per
# column is wrong when different values plausibly need different
# answers). Weight/length gets the identical treatment.

# Candidate tokens for fuzzy matching -- WORD-FORM tokens only (3+
# characters), same reasoning as Module 47: a short bare symbol like
# "g" or "in" fuzzy-matched against a large candidate pool produces
# nonsense matches far too easily, whereas a genuine typo of a longer
# unit word ("kilogramm", "puonds") is long enough for similarity
# matching to be meaningful.
_FUZZY_UNIT_CANDIDATES = sorted(
    {token for token in {**WEIGHT_UNIT_ALIASES, **LENGTH_UNIT_ALIASES} if len(token) >= 3}
)
_ALL_UNIT_TOKENS = {**WEIGHT_UNIT_ALIASES, **LENGTH_UNIT_ALIASES}


def _suggest_likely_unit(raw_token: str):
    """
    Best-effort fuzzy suggestion for an unrecognised unit-like token,
    e.g. "kgs." with stray punctuation, or "poundz" -> "lb". Returns a
    canonical unit code (for use as a dropdown default) or None if
    nothing is a close enough match.

    NEVER auto-applied -- only ever a PRE-FILLED DEFAULT for a question
    the client still sees and can override, identical principle to
    Module 47's _suggest_likely_currency (same cutoff=0.6, chosen there
    for the same reason: catches a genuine transposition/typo while
    still requiring real similarity).
    """
    if not raw_token or len(raw_token) < 3:
        return None

    matches = difflib.get_close_matches(raw_token.lower(), _FUZZY_UNIT_CANDIDATES, n=1, cutoff=0.6)
    if not matches:
        return None

    return _ALL_UNIT_TOKENS[matches[0]]


def find_unresolved_unit_details(df: pd.DataFrame, col: str, info: dict) -> list:
    """
    Enumerates every DISTINCT problem value in `col` -- mirrors Module
    47's find_unresolved_currency_details exactly, including the
    is_bare distinction (True = no unit text at all, e.g. "50"; False
    = an unrecognised unit-like token was present, e.g. "60 kgz").

    Returns a list of dicts: raw_value, count, example_row_index,
    example_row, suggested_unit, is_bare. See Module 47's docstring
    for the full reasoning -- identical here.
    """
    implied_unit = info.get("implied_unit_from_name")
    if implied_unit:
        return []  # bare numbers already resolved via the implied unit -- nothing ambiguous

    series = df[col]
    seen = {}

    for idx, raw in series.items():
        if pd.isna(raw):
            continue
        number, canonical_unit, raw_token = parse_quantity(raw)
        if number is None or canonical_unit is not None:
            continue  # either unparseable (genuinely missing) or already resolved

        key = str(raw)
        if key not in seen:
            example_row = df.loc[idx].to_dict()
            seen[key] = {
                "raw_value": raw,
                "count": 0,
                "example_row_index": idx,
                "example_row": example_row,
                "suggested_unit": _suggest_likely_unit(raw_token) if raw_token else None,
                "is_bare": raw_token is None,
            }
        seen[key]["count"] += 1

    return list(seen.values())


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "Weight": ["5kg", "10 lbs", "7", "3.2 kg", "8 lbs", "6.5kg", "N/A", "12 stone"],
        "Height_cm": [170, 165, 180, 175, "5.5 ft", 160, 172, None],
        "Product_Price": [19.99, 25.5, 12.0, 8.75, 30.0, 15.0, 20.0, 22.0],
        "Family_Size": [4, 3, 2, 5, 4, 3, 2, 1],       # decoy -- weak length keyword "size" but clearly not a measurement
        "Sample_ID": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
        "Distance_Travelled": ["12km", "5 miles", "8km", "3.5mi", "10km", "1 mile", "7km", "2mi"],
    })

    print("=== Test dataframe ===")
    print(test_df)

    print("\n=== parse_quantity() spot checks ===")
    for val in ["5.5 kg", "170cm", "42", "12 stone", "N/A", 55, "3,200 g", "5.5ft", "5 ft 8 in", "5'8\"", "5'", "8\""]:
        print(f"  {val!r:15} -> {parse_quantity(val)}")

    print("\n=== detect_measurement_columns() ===")
    detected = detect_measurement_columns(test_df)
    for col, info in detected.items():
        print(f"\nColumn: {col}")
        for k, v in info.items():
            print(f"  {k}: {v}")

    print("\n=== Columns NOT flagged (should be Product_Price, Family_Size, Sample_ID) ===")
    not_flagged = [c for c in test_df.columns if c not in detected]
    print(not_flagged)
