"""
Module: Phone Standardisation -- Sub-step A (Detection & Parsing)
----------------------------------------------------------------------
Generic-mode feature, sibling to Modules 45/47/50. Handles telephone
NUMBER formatting/validation -- distinct from every other
standardisation feature so far in one important way: this module does
NOT try to reimplement country-code or area-code knowledge by hand.
It uses the `phonenumbers` library (Google's libphonenumber, ported to
Python) -- the industry-standard tool for this exact job, already
carrying metadata for essentially every country/territory in the
world. Reinventing that metadata by hand would be reinventing a
solved problem, and worse, a WRONG one (area code rules genuinely vary
by country in ways that are easy to get subtly wrong).

WHY THIS MEANS NO "list every possible country code" QUESTION IS
NEEDED: any number that already carries a recognisable country code
(a leading "+44", "0044", etc.) gets its country identified
automatically by the library. The ONLY case that needs help from the
client is a number with NO country code at all (a bare national-format
number, e.g. "020 7946 0958") -- for those, and ONLY those, a single
"default region" answer is needed (Module 53's question chain will
ask this once per column, not once per possible country).

VALIDITY AS THE CONFIDENCE GATE: phonenumbers.is_valid_number() is
what separates "safe to auto-format" from "needs review" -- tested
against the client's own spec example ("44 055542231" with region GB
parses to an 8-digit number and correctly comes back INVALID, not
silently forced into a wrong-looking match).

SCOPE NOTE for this sub-step: only E164/INTERNATIONAL/NATIONAL target
formats are built here (all backed directly by the library). A
free-form CUSTOM template (e.g. "+44 (0) XXXX XXXXXX" typed by the
client) is a genuinely separate parsing job -- mapping arbitrary
placeholder characters onto extracted digit groups -- and is
deliberately deferred to a later sub-step rather than rushed in here.
"""

import re
from importlib import import_module

import pandas as pd
import phonenumbers
from phonenumbers import PhoneNumberFormat

TREAT_AS_MISSING = "__treat_as_missing__"

# Column-NAME keyword signals, same STRONG/WEAK split as every other
# detection module in this project.
STRONG_PHONE_NAME_KEYWORDS = ["phone", "telephone", "mobile", "cell", "fax"]
WEAK_PHONE_NAME_KEYWORDS = ["tel", "contact_number", "contact_no"]

# Lightweight PRE-FILTER before ever invoking phonenumbers on a value --
# strips common formatting characters and checks what's left is mostly
# digits of a plausible phone-number length (E.164 caps at 15 digits).
# This is deliberately cheap and permissive (a real validity check
# happens later, per-value, via phonenumbers itself) -- its only job is
# deciding whether a COLUMN is worth treating as phone numbers at all.
_FORMATTING_CHARS_RE = re.compile(r"[\s\-().+]")


def _looks_phone_like(raw) -> bool:
    if pd.isna(raw):
        return False
    text = str(raw).strip()
    if text == "":
        return False
    stripped = _FORMATTING_CHARS_RE.sub("", text)
    return stripped.isdigit() and 6 <= len(stripped) <= 15


# ---------------------------------------------------------------------------
# 1. Single-value parsing / validation
# ---------------------------------------------------------------------------

_REASON_LABELS = {
    phonenumbers.ValidationResult.IS_POSSIBLE: "plausible length, but not a recognised real number",
    phonenumbers.ValidationResult.INVALID_COUNTRY_CODE: "invalid or unrecognised country code",
    phonenumbers.ValidationResult.TOO_SHORT: "too few digits for a real number",
    phonenumbers.ValidationResult.TOO_LONG: "too many digits for a real number",
    phonenumbers.ValidationResult.INVALID_LENGTH: "wrong number of digits for the detected country",
}


def parse_phone_value(raw, default_region: str = None) -> dict:
    """
    Attempts to parse and validate a single phone value. `default_region`
    (a 2-letter code like "GB", "US") is used ONLY when the value itself
    carries no country code -- if the value already has one ("+44...",
    "0044...", "44 20 7946 0958" -- yes, phonenumbers can often infer a
    country code even without a leading "+", though a leading "+" or
    default_region makes this reliable rather than a guess), the
    default_region is not needed and is ignored.

    Returns a dict:
      raw               the original value
      parsed_ok         True if phonenumbers could parse SOME number at all
                         (even if it turns out invalid) -- False only for
                         genuinely un-parseable garbage (e.g. all letters)
      is_valid          True only if this is a REAL, dialable number for
                         its detected country -- the actual "safe to
                         auto-standardise" gate
      region             detected country code (e.g. "GB"), or None
      e164 / international / national   formatted strings, or None if
                         not valid
      reason            human-readable explanation when NOT valid (for
                         the batch review table's "Problem" column)
    """
    result = {
        "raw": raw, "parsed_ok": False, "is_valid": False, "region": None,
        "e164": None, "international": None, "national": None, "reason": None,
    }

    if pd.isna(raw):
        result["reason"] = "empty value"
        return result

    text = str(raw).strip()
    if text == "":
        result["reason"] = "empty value"
        return result

    try:
        number = phonenumbers.parse(text, default_region)
    except phonenumbers.NumberParseException as e:
        # BUG FIXED after testing: e.error_type is a plain int constant,
        # not an enum member -- it has no .name attribute. str(e)
        # already gives a clear message (e.g. "(0) Missing or invalid
        # default region."), so use that directly instead.
        result["reason"] = f"could not be parsed as a phone number ({e})"
        return result

    result["parsed_ok"] = True
    result["region"] = phonenumbers.region_code_for_number(number)

    if phonenumbers.is_valid_number(number):
        result["is_valid"] = True
        result["e164"] = phonenumbers.format_number(number, PhoneNumberFormat.E164)
        result["international"] = phonenumbers.format_number(number, PhoneNumberFormat.INTERNATIONAL)
        result["national"] = phonenumbers.format_number(number, PhoneNumberFormat.NATIONAL)
    else:
        validity_reason = phonenumbers.is_possible_number_with_reason(number)
        result["reason"] = _REASON_LABELS.get(validity_reason, "not a valid number for the detected country")

    return result


FORMAT_NAME_TO_ENUM = {
    "E164": PhoneNumberFormat.E164,
    "INTERNATIONAL": PhoneNumberFormat.INTERNATIONAL,
    "NATIONAL": PhoneNumberFormat.NATIONAL,
}


def _load_region_display_names() -> dict:
    """
    Builds {region_code: "Country Name (+CC)"} for every region
    phonenumbers actually has metadata for -- used to give the
    "default region" question real country names instead of bare ISO
    codes ("GB") that most people wouldn't immediately recognise.
    Uses pycountry for the name lookup (a small, focused dependency
    for exactly this one job) -- falls back to the bare code if a
    region isn't in pycountry's list for any reason (a few phone
    regions, like "001" for shared services, aren't real countries).
    """
    try:
        import pycountry
    except ImportError:
        return {}

    names = {}
    for region in sorted(phonenumbers.SUPPORTED_REGIONS):
        country = pycountry.countries.get(alpha_2=region)
        calling_code = phonenumbers.country_code_for_region(region)
        if country:
            names[region] = f"{country.name} ({region}, +{calling_code})"
        else:
            names[region] = f"{region} (+{calling_code})"
    return names


REGION_DISPLAY_NAMES = _load_region_display_names()


def format_phone_number(raw, default_region: str, target_format: str):
    """
    Formats a single value into the requested target format ("E164",
    "INTERNATIONAL", or "NATIONAL"). Returns the formatted string, or
    None if the value isn't a valid number (caller should treat this
    the same as any other unresolved value -- never guess a format for
    an invalid number).
    """
    parsed = parse_phone_value(raw, default_region)
    if not parsed["is_valid"]:
        return None
    key = target_format.lower()
    return parsed.get(key) or parsed["international"]


# ---------------------------------------------------------------------------
# 2. Column-level detection
# ---------------------------------------------------------------------------

def _name_signal(col_name: str) -> dict:
    name_lower = col_name.lower()
    strong = any(kw in name_lower for kw in STRONG_PHONE_NAME_KEYWORDS)
    weak = any(kw in name_lower for kw in WEAK_PHONE_NAME_KEYWORDS)
    return {"strong": strong, "weak": weak}


def detect_phone_columns(df: pd.DataFrame, sample_size: int = 200) -> dict:
    """
    Returns {column_name: info_dict} for columns worth treating as
    phone numbers. Two-stage check, same spirit as every other
    detection module here: a cheap CONTENT pre-filter (_looks_phone_
    like, no phonenumbers calls yet) combined with column NAME
    keywords.

    IMPORTANT DISTINCTION, caught during testing: a plain national-
    format number ("020 7946 0958") genuinely CANNOT be validated by
    phonenumbers without a default_region -- that's not the same as
    being invalid, it just means "not yet knowable." Since this
    detection step runs BEFORE the client has answered the default-
    region question, it deliberately does NOT attempt real validity
    checks on every value here (that would mislabel most legitimate
    numbers as "invalid" for the wrong reason). It only attempts a
    region-FREE parse on values that already carry their own country
    code (a leading "+", which phonenumbers can parse with no region
    hint at all) -- purely to produce a sensible "guessed_region"
    suggestion for that later question, not a validity verdict.

    info_dict:
      confidence         0.0-1.0, same thresholds as other modules
      n_sampled
      guessed_region     the most common detected region among values
                         that already carry their own country code, if
                         any -- used ONLY as a sensible pre-fill
                         suggestion for the "default region" question,
                         never applied silently
    """
    results = {}

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        sample = sample.sample(n=min(sample_size, len(sample)), random_state=0) if len(sample) > sample_size else sample

        phone_like = [v for v in sample if _looks_phone_like(v)]
        content_rate = len(phone_like) / len(sample)

        name_sig = _name_signal(str(col))

        is_phone_column = False
        confidence = 0.0
        # Same fix as Module 53's email detection, same reasoning: a
        # column literally named "Phone"/"Telephone" shouldn't fail to
        # be recognised as a phone column just because a meaningful
        # share of its values are genuinely malformed -- that's the
        # realistic messy-data case this feature exists for, not a
        # reason to skip it entirely.
        if name_sig["strong"]:
            is_phone_column = True
            if content_rate >= 0.6:
                confidence = 0.95
            elif content_rate >= 0.3:
                confidence = 0.75
            else:
                confidence = 0.5
        elif content_rate >= 0.6 and name_sig["weak"]:
            is_phone_column = True
            confidence = 0.75
        elif content_rate >= 0.8:
            is_phone_column = True
            confidence = 0.6

        if not is_phone_column:
            continue

        # Only values that ALREADY carry their own country code (a
        # leading "+") can be parsed with no region hint at all --
        # used purely to suggest a likely default region, never to
        # judge validity of the column as a whole.
        already_international = [v for v in phone_like if str(v).strip().startswith("+")]
        region_hits = []
        for v in already_international:
            parsed = parse_phone_value(v, None)
            if parsed["is_valid"] and parsed["region"]:
                region_hits.append(parsed["region"])
        guessed_region = max(set(region_hits), key=region_hits.count) if region_hits else None

        results[col] = {
            "confidence": confidence,
            "n_sampled": len(sample),
            "guessed_region": guessed_region,
        }

    return results


if __name__ == "__main__":
    print("=== parse_phone_value() spot checks ===")
    tests = [
        ("44 055542231", "GB"),   # the client's own ambiguous example
        ("+44 20 7946 0958", None),
        ("020 7946 0958", "GB"),
        ("4155552671", "US"),
        ("4455ABC231", "GB"),     # garbage
        ("", None),
    ]
    for raw, region in tests:
        print(f"  {raw!r:20} region={region} -> {parse_phone_value(raw, region)}")

    print("\n=== detect_phone_columns() ===")
    test_df = pd.DataFrame({
        "Phone": ["+44 20 7946 0958", "020 7946 0958", "4155552671", "4455ABC231", "+1 415 555 2671", "07911 123456"],
        "Zip_Code": ["94103", "10001", "60601", "02134", "94105", "10002"],
        "Customer_Name": ["John Smith", "Jane Doe", "Bob Lee", "Sara Khan", "Tom Hill", "Ann Bell"],
    })
    detected = detect_phone_columns(test_df)
    for col, info in detected.items():
        print(f"\nColumn: {col}")
        for k, v in info.items():
            print(f"  {k}: {v}")

    print("\n=== Columns NOT flagged (should be Zip_Code, Customer_Name) ===")
    print([c for c in test_df.columns if c not in detected])
