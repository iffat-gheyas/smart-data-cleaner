"""
Module I: Column Profiling (Phase 2 — new)
---------------------------------------------
This is the "figure out what we're looking at" step that runs right
after upload, BEFORE any question gets shown to the user. Its job is
to give the rest of the app enough information to (a) auto-fill
sensible defaults and (b) know which clarifying questions are even
worth asking for THIS specific file.

Being built incrementally, same as every Sprint 1 module:
  Sub-step 1 (this one): basic dtype + missing-value profiling,
      including detection of "disguised missing" markers like "?",
      "N/A", "unknown" that LOOK like real values to pandas but
      actually mean "missing".
  Sub-step 2 (next): semantic role detection (email-like, phone-like,
      full-name-like, address-like, date-like, categorical, free-text,
      ID-like) using column name + content heuristics.
  Sub-step 3: date-ambiguity detection (does a date-like column
      actually contain genuinely ambiguous dd/mm vs mm/dd values, or
      is the dayfirst question moot for this file?).

DESIGN NOTE: this module must work on ANY uploaded dataset, not just
our synthetic dirty_customer_data file. So nothing here is allowed to
reference a specific column name -- everything is inferred from the
data itself. dirty_customer_data.xlsx is used only to TEST it.
"""

import re
from importlib import import_module

import numpy as np
import pandas as pd

datetime_std = import_module("04_datetime_standardisation")

# Same UK postcode SHAPE Module H already validates against
# (08_validation.py), used here purely as a DETECTION signal -- "does
# this value end in something postcode-shaped" -- not a full postcode
# validity check (that's Module H's job on the already-split column,
# after splitting has run).
_UK_POSTCODE_AT_END_RE = re.compile(r"[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$")


# ---------------------------------------------------------------------------
# 1. Basic dtype classification
# ---------------------------------------------------------------------------

def classify_basic_dtype(series: pd.Series) -> str:
    """
    Classify a column into one of: "numeric", "boolean", "text".

    We deliberately keep this coarse for now. It is NOT the same thing
    as "semantic role" (sub-step 2) -- a numeric column might still be
    an ID (e.g. customer_id), and a text column might still be a date
    written as a string (pandas reads dates-as-text from CSV/Excel as
    plain object/text columns, so date detection has to happen on TOP
    of this, by inspecting the actual string content -- that's sub-
    step 2/3's job, not this one).

    We check boolean before numeric because pandas' actual bool dtype
    would otherwise also satisfy a numeric check (True/False behave
    like 1/0 under the hood).
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "text"


# ---------------------------------------------------------------------------
# 2. Suspicious "disguised missing" marker detection
# ---------------------------------------------------------------------------

# Candidate markers that OFTEN mean "missing" in real-world messy data,
# but which pandas has no way of knowing about on its own -- to pandas
# they're just an ordinary string value, no different from any other.
# Checked case-insensitively, after stripping whitespace.
DISGUISED_MISSING_CANDIDATES = {
    "?", "n/a", "na", "none", "null", "nil", "-", "--", "unknown",
    "unk", "tbd", "pending", "999", "9999", "-999", "0000",
    # Added after real-user testing found a genuine gap: a real dataset
    # used "NaN" (the literal text) and "ERROR" as placeholder values
    # in nearly every column, and neither was on the original list.
    "nan", "error", "err", "invalid", "missing", "not available",
    "not provided", "void", "blank", "unspecified", "unassigned",
    "n.a", "#n/a", "not applicable",
}

# A candidate only gets FLAGGED if it appears more than once. A single
# occurrence is more likely to be a genuine, if odd, value; a REPEATED
# non-blank marker across multiple rows is the actual signal that
# someone (or some upstream system) is using it as a stand-in for
# "no data" -- which is exactly the distinction the project spec asks
# for ("suspicious non-blank marker appears repeatedly").
MIN_OCCURRENCES_TO_FLAG = 2


# Numeric-only placeholder candidates. Kept as a SEPARATE, much shorter
# list from the text markers above, and deliberately conservative --
# unlike "?" or "unknown" (which are essentially never genuine data),
# a number like 999 legitimately occurs sometimes (e.g. a real
# order_amount of 999.00). We only flag classic, deliberately-
# unrealistic round-number placeholders, and still require repetition
# (MIN_OCCURRENCES_TO_FLAG) before raising it -- one row with age=999
# might just be a typo; several rows sharing the exact same
# suspiciously-round value is the real signal of a placeholder
# convention. We do NOT include 0 here: zero is too often a genuine
# value (e.g. order_amount=0 for a cancelled order) to safely assume
# it means "missing" without the user's confirmation.
NUMERIC_DISGUISED_MISSING_CANDIDATES = {999, 9999, -999, -1}


def detect_disguised_missing(series: pd.Series) -> dict:
    """
    Look for values that aren't technically missing (pandas doesn't
    see them as NaN) but are very likely being used AS "missing" by
    whoever entered the data.

    Handles text columns (e.g. "?", "unknown") AND numeric columns
    (e.g. age=999 used as a placeholder) -- these need different
    detection logic since a numeric column can never contain a string
    like "?" (if it did, pandas would already have been forced to read
    the whole column as text, which classify_basic_dtype() would catch
    upstream), but it CAN contain a suspiciously round placeholder
    number.

    Returns a dict of {marker_found: occurrence_count} (marker key is
    always a string for consistency, even for the numeric case),
    restricted to markers that cleared MIN_OCCURRENCES_TO_FLAG.
    """
    non_null = series.dropna()
    if non_null.empty:
        return {}

    found = {}

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        text_values = non_null.astype(str).str.strip().str.lower()
        value_counts = text_values.value_counts()
        for marker in DISGUISED_MISSING_CANDIDATES:
            count = int(value_counts.get(marker, 0))
            if count >= MIN_OCCURRENCES_TO_FLAG:
                found[marker] = count

    elif pd.api.types.is_numeric_dtype(series):
        value_counts = non_null.value_counts()
        for marker in NUMERIC_DISGUISED_MISSING_CANDIDATES:
            count = int(value_counts.get(marker, 0))
            if count >= MIN_OCCURRENCES_TO_FLAG:
                found[str(marker)] = count

    return found


# ---------------------------------------------------------------------------
# 3. Per-column profile (combines 1 + 2 + straightforward missing stats)
# ---------------------------------------------------------------------------

def profile_column(series: pd.Series) -> dict:
    """
    Build the profile for a single column.

    missing_count / missing_percent here mean the SAME thing as Module
    A's diagnostic_summary() (true pandas NaN only) -- disguised
    missing values are reported SEPARATELY as
    'disguised_missing_markers', not folded into the official missing
    count. That distinction matters: we don't want to silently treat
    "?" as missing before the user has confirmed that's what it means
    for THEIR data. Confirming it is a job for the question-generation
    step (sub-step coming later) -- this module only detects and
    reports, same "flag, don't act" philosophy as every Sprint 1
    module.
    """
    n = len(series)
    missing_count = int(series.isna().sum())

    return {
        "dtype": classify_basic_dtype(series),
        "missing_count": missing_count,
        "missing_percent": round((missing_count / n) * 100, 2) if n else 0.0,
        "disguised_missing_markers": detect_disguised_missing(series),
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Run profile_column() across every column in the DataFrame.

    Returns {column_name: profile_dict}. This is the object the rest
    of the app (question generation, advanced-settings defaults) will
    read from -- kept as a plain dict (not a custom class) so it's
    trivially JSON-serialisable for a Streamlit session or a future
    API layer.
    """
    return {col: profile_column(df[col]) for col in df.columns}


def print_profile(profile: dict) -> None:
    """
    Human-readable dump, for testing/debugging this module on its own
    before the Streamlit UI exists to display it properly.
    """
    for col, info in profile.items():
        print(f"\n{col}")
        print(f"  dtype: {info['dtype']}")
        print(f"  missing: {info['missing_count']} ({info['missing_percent']}%)")
        if info["disguised_missing_markers"]:
            markers = ", ".join(
                f"'{m}' x{c}" for m, c in info["disguised_missing_markers"].items()
            )
            print(f"  possible disguised-missing markers found: {markers}")
        if info.get("date_ambiguity") and info["date_ambiguity"]["needs_dayfirst_question"]:
            amb = info["date_ambiguity"]
            ex = amb["example"]
            print(
                f"  DAYFIRST QUESTION NEEDED: {amb['ambiguous_count']}/{amb['date_like_values_checked']} "
                f"values are ambiguous. Example: '{ex['original_value']}' -> "
                f"{ex['if_dayfirst']} (dayfirst) or {ex['if_monthfirst']} (monthfirst)?"
            )


# ---------------------------------------------------------------------------
# 4. Semantic role detection
# ---------------------------------------------------------------------------
# Two independent signals, combined:
#   - CONTENT signal: what the actual values look like (regex match
#     rates, uniqueness, length, digit density). This is the reliable
#     signal -- it works regardless of how the client happened to name
#     their columns.
#   - NAME signal: does the column header contain a hint keyword. A
#     much weaker, supporting signal, since column naming is
#     inconsistent across clients (and sometimes actively misleading).
#
# Content is weighted more heavily (CONTENT_WEIGHT) and a role is only
# assigned if the content score alone clears MIN_CONTENT_TO_QUALIFY --
# so a column named "phone_number" that's actually full of free-text
# notes won't get mislabeled just because of its header.

CONTENT_WEIGHT = 0.7
NAME_WEIGHT = 0.3
MIN_CONTENT_TO_QUALIFY = 0.5   # a role's content evidence must clear this
MIN_FINAL_SCORE = 0.4          # combined score must clear this to be assigned
SAMPLE_SIZE = 300              # cap on how many values we test, for speed on large files

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

NAME_HINTS = {
    "email": ["email", "e-mail", "mail"],
    "phone": ["phone", "tel", "mobile", "contact_no", "contactno"],
    "full_name": ["name"],
    "address": ["address", "addr", "location"],
    "date_or_datetime": ["date", "time", "dt", "timestamp"],
    "id_key": ["id", "key", "code", "no.", "number", "ref"],
    "numeric_text": ["price", "amount", "quantity", "qty", "total", "cost", "value", "count"],
    "country": ["country", "nation"],
    "state": ["state", "province"],
}

# ---------------------------------------------------------------------------
# Country name reference dictionary
# ---------------------------------------------------------------------------
# A BOUNDED, known reference set -- unlike company names, there IS one
# universally agreed canonical list of countries, which is what makes a
# static dictionary a safe approach here (added after real-user testing
# found country standardisation was incomplete: "U.K." / "UK" / "United
# Kingdom" / "england" were all left as separate values).
#
# Deliberately NOT exhaustive -- covers the countries most likely to
# appear in a typical client's customer/business data. Worth extending
# with more entries as real client files expose gaps, same as every
# other heuristic in this project.
COUNTRY_NAME_VARIANTS = {
    "United Kingdom": ["united kingdom", "uk", "u.k.", "britain", "great britain", "gb"],
    "United States": ["united states", "usa", "u.s.a.", "u.s.a", "us", "u.s.", "united states of america"],
    "France": ["france", "fr"],
    "Germany": ["germany", "de", "deutschland"],
    "Spain": ["spain", "es", "espana", "españa"],
    "Italy": ["italy", "it", "italia"],
    "Canada": ["canada", "ca"],
    "Australia": ["australia", "au", "aus"],
    "Ireland": ["ireland", "ie", "eire"],
    "Netherlands": ["netherlands", "nl", "holland"],
    "India": ["india", "in"],
    "China": ["china", "cn", "prc"],
    "Japan": ["japan", "jp"],
    "Brazil": ["brazil", "br", "brasil"],
    "Mexico": ["mexico", "mx"],
    "South Africa": ["south africa", "za"],
    "New Zealand": ["new zealand", "nz"],
    "Portugal": ["portugal", "pt"],
    "Switzerland": ["switzerland", "ch"],
    "Sweden": ["sweden", "se"],
    "Norway": ["norway", "no"],
    "Denmark": ["denmark", "dk"],
    "Poland": ["poland", "pl"],
    "Belgium": ["belgium", "be"],
    "Austria": ["austria", "at"],
    "Greece": ["greece", "gr"],
    "Turkey": ["turkey", "tr"],
    "Russia": ["russia", "ru"],
    "South Korea": ["south korea", "kr", "korea"],
    "Singapore": ["singapore", "sg"],
    "United Arab Emirates": ["united arab emirates", "uae", "u.a.e."],
}

# England/Scotland/Wales/Northern Ireland are genuinely the UK's
# constituent countries, not incorrect data -- whether to merge them
# into "United Kingdom" is a business-rule judgment call (some clients
# may want regional breakdown preserved), not an objective spelling
# fix. Handled separately from the dictionary above and gated behind
# an explicit advanced-setting toggle in Module K, default OFF.
UK_CONSTITUENT_COUNTRIES = {"england", "scotland", "wales", "northern ireland"}

# US states (full names only -- deliberately NOT abbreviations like
# "ca" or "ny", since those collide with real country codes already in
# COUNTRY_VARIANT_LOOKUP, e.g. "ca" is also Canada's abbreviation).
# Unlike the UK constituent countries, a US state appearing in a
# column literally called "country" is essentially always a data-entry
# mistake (someone typed the state instead of the country) rather than
# a deliberate regional distinction -- so this is handled with its own
# separate, default-ON toggle in Module K, not lumped in with the
# UK setting.
US_STATES_TO_COUNTRY = {
    "alabama": "United States", "alaska": "United States", "arizona": "United States",
    "arkansas": "United States", "california": "United States", "colorado": "United States",
    "connecticut": "United States", "delaware": "United States", "florida": "United States",
    "georgia": "United States", "hawaii": "United States", "idaho": "United States",
    "illinois": "United States", "indiana": "United States", "iowa": "United States",
    "kansas": "United States", "kentucky": "United States", "louisiana": "United States",
    "maine": "United States", "maryland": "United States", "massachusetts": "United States",
    "michigan": "United States", "minnesota": "United States", "mississippi": "United States",
    "missouri": "United States", "montana": "United States", "nebraska": "United States",
    "nevada": "United States", "new hampshire": "United States", "new jersey": "United States",
    "new mexico": "United States", "new york": "United States", "north carolina": "United States",
    "north dakota": "United States", "ohio": "United States", "oklahoma": "United States",
    "oregon": "United States", "pennsylvania": "United States", "rhode island": "United States",
    "south carolina": "United States", "south dakota": "United States", "tennessee": "United States",
    "texas": "United States", "utah": "United States", "vermont": "United States",
    "virginia": "United States", "washington": "United States", "west virginia": "United States",
    "wisconsin": "United States", "wyoming": "United States", "district of columbia": "United States",
}

# Correctly-CASED canonical form of every US state name -- used to fix
# casing on a genuine "state" column (e.g. "california" / "CALIFORNIA"
# -> "California"), completely separately from US_STATES_TO_COUNTRY
# above, which answers a different question ("should this be merged
# INTO the country name"). Kept as its own explicit dict (not derived
# via .title(), which would incorrectly capitalise "of" in "District
# of Columbia") for the same "explicit beats guessed" reasoning as
# COUNTRY_NAME_VARIANTS.
STATE_CANONICAL_NAMES = {
    "alabama": "Alabama", "alaska": "Alaska", "arizona": "Arizona", "arkansas": "Arkansas",
    "california": "California", "colorado": "Colorado", "connecticut": "Connecticut",
    "delaware": "Delaware", "florida": "Florida", "georgia": "Georgia", "hawaii": "Hawaii",
    "idaho": "Idaho", "illinois": "Illinois", "indiana": "Indiana", "iowa": "Iowa",
    "kansas": "Kansas", "kentucky": "Kentucky", "louisiana": "Louisiana", "maine": "Maine",
    "maryland": "Maryland", "massachusetts": "Massachusetts", "michigan": "Michigan",
    "minnesota": "Minnesota", "mississippi": "Mississippi", "missouri": "Missouri",
    "montana": "Montana", "nebraska": "Nebraska", "nevada": "Nevada",
    "new hampshire": "New Hampshire", "new jersey": "New Jersey", "new mexico": "New Mexico",
    "new york": "New York", "north carolina": "North Carolina", "north dakota": "North Dakota",
    "ohio": "Ohio", "oklahoma": "Oklahoma", "oregon": "Oregon", "pennsylvania": "Pennsylvania",
    "rhode island": "Rhode Island", "south carolina": "South Carolina", "south dakota": "South Dakota",
    "tennessee": "Tennessee", "texas": "Texas", "utah": "Utah", "vermont": "Vermont",
    "virginia": "Virginia", "washington": "Washington", "west virginia": "West Virginia",
    "wisconsin": "Wisconsin", "wyoming": "Wyoming", "district of columbia": "District of Columbia",
}


def _normalize_country_text(value) -> str:
    return str(value).strip().lower().replace(".", "").replace(",", "")


def _build_country_variant_lookup() -> dict:
    lookup = {}
    for canonical, variants in COUNTRY_NAME_VARIANTS.items():
        for variant in variants:
            lookup[variant] = canonical
    return lookup


COUNTRY_VARIANT_LOOKUP = _build_country_variant_lookup()


def _content_score_country(sample: pd.Series) -> float:
    """
    Fraction of sample values recognised as a REAL country name or a
    known variant of one (including the UK constituent countries,
    which count as "recognised" for DETECTION purposes regardless of
    the separate merge-or-not business decision made later in
    Module K).

    Deliberately does NOT count US state names as "recognised" here
    any more (they used to be included in this same set) -- that was
    the root cause of a real bug: a column genuinely asking "what US
    state do you work in?" would score just as high on "country" as
    an actual country column, because nearly every value in it
    matched something in this recognised set. The two need to be
    told apart, not lumped together -- see _content_score_state
    below, which is what now separately catches a state-only column.
    A genuine country column that happens to contain a FEW erroneous
    state entries (someone typed their state instead of their
    country) is unaffected by this change: those specific values still
    get corrected at cleaning time by _standardize_country_column in
    Module K, which checks membership in US_STATES_TO_COUNTRY directly
    rather than relying on this profiling-stage score.
    """
    if sample.empty:
        return 0.0
    normalized = sample.astype(str).apply(_normalize_country_text)
    recognized = normalized.isin(set(COUNTRY_VARIANT_LOOKUP.keys()) | UK_CONSTITUENT_COUNTRIES)
    return recognized.mean()


def _content_score_state(sample: pd.Series) -> float:
    """
    Fraction of sample values recognised as a US state name.

    A column dominated by state names but with little/no genuine
    country-name diversity (see _content_score_country's docstring)
    should be classified as its OWN "state" role rather than being
    folded into "country" -- a state column needs a completely
    different default treatment (preserve the actual state name) than
    a country column (optionally merge a stray state into "United
    States").
    """
    if sample.empty:
        return 0.0
    normalized = sample.astype(str).apply(_normalize_country_text)
    return normalized.isin(set(US_STATES_TO_COUNTRY.keys())).mean()


def _split_camel_case(name: str) -> str:
    """Inserts a space before an uppercase letter that follows a lowercase letter/digit, so "CustomerID" -> "Customer ID"."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def _name_hint_score(column_name: str, role: str) -> float:
    """
    1.0 if any hint keyword appears as a WHOLE WORD in the column
    name, else 0.0.

    Uses word-boundary tokenisation, NOT naive substring search --
    confirmed as a real, significant bug by testing against a real
    survey dataset: a column asking the user to "please PROVIDE" extra
    context was scored as an ID/reference column, because "id" is a
    substring of "provide". The same naive-substring approach would
    also have silently misfired on "date" inside "update"/"mandate"/
    "candidate", "key" inside "monkey"/"turkey", etc. -- likely causing
    other false positives we simply hadn't noticed yet.

    Column names are camelCase-split first ("CustomerID" -> "Customer
    ID") before tokenising on whitespace/punctuation, so a genuine
    no-separator convention like "CustomerID" or "OrderNo" still
    correctly matches its hint as a whole token, while an ordinary
    English word that happens to CONTAIN a hint as a substring no
    longer falsely matches.
    """
    spaced = _split_camel_case(column_name)
    normalised = spaced.lower().replace("_", " ").replace("-", " ")
    tokens = set(re.findall(r"[a-z0-9]+\.?", normalised))
    for hint in NAME_HINTS.get(role, []):
        if hint in tokens:
            return 1.0
    return 0.0


def _get_sample(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) > SAMPLE_SIZE:
        return non_null.sample(SAMPLE_SIZE, random_state=42)
    return non_null


def _content_score_email(sample: pd.Series) -> float:
    if sample.empty:
        return 0.0
    matches = sample.astype(str).str.strip().apply(lambda v: bool(EMAIL_PATTERN.match(v)))
    return matches.mean()


# A value that is STRUCTURALLY nothing but a bare numeric date (three
# digit-groups separated by /, -, or .) -- e.g. "2025-01-01" or
# "03/04/2026". Phone numbers essentially never take this exact
# shape, but a date can easily satisfy a naive "mostly digits, some
# formatting punctuation" check, since "-" is valid formatting
# punctuation for BOTH. Used to explicitly exclude dates from the
# phone check below, rather than relying on scoring alone to sort out
# the collision (confirmed by testing: without this exclusion, an
# all-ISO-date column tied with -- and sometimes beat -- the date
# detector purely on dict/iteration ordering).
_BARE_NUMERIC_DATE_SHAPE = re.compile(r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}$")


def _content_score_phone(sample: pd.Series) -> float:
    """
    A value 'looks like' a phone number if most of its characters are
    digits (allowing spaces, brackets, +, - as formatting) and the
    total digit count falls in a plausible phone-number range (7-15,
    same reasoning as Module H's validation defaults) -- AND it isn't
    just a bare numeric date wearing phone-shaped clothing.
    """
    if sample.empty:
        return 0.0

    def looks_like_phone(v):
        v = str(v).strip()
        if _BARE_NUMERIC_DATE_SHAPE.match(v):
            return False
        digit_count = sum(ch.isdigit() for ch in v)
        non_format_chars = sum(
            1 for ch in v if not (ch.isdigit() or ch in " ()+-.")
        )
        return 7 <= digit_count <= 15 and non_format_chars == 0

    return sample.astype(str).apply(looks_like_phone).mean()


def _content_score_date(sample: pd.Series) -> float:
    """
    Tries parsing with BOTH dayfirst settings and takes the better
    result -- at this profiling stage we're only asking "does this
    column look date-like at all", not resolving the dayfirst question
    itself (that's a separate, later concern -- Module D and the
    question-generation step, not this one).

    BUG FIXED after real-file testing: this used to parse the RAW text
    only. A column full of genuinely noisy-but-recoverable values --
    "27:01:2022: 17: 20:10", "half past three", "fourteen-twenty",
    ordinal days ("6th"), all things Module D's standardise_date_
    column() can now handle -- scored far too low here to ever be
    classified as a date column at all, so it never even reached
    Module D in the first place: the fix that makes those values
    parseable was useless if this earlier detection step never
    recognised the column as a candidate to apply it to.

    Now runs the SAME normalize_natural_language_datetime() Module D
    itself uses before parsing -- a pure text rewrite (ordinals
    stripped, spoken times converted, noisy separators fixed), never
    an assumption about ambiguous day/month order, so reusing it here
    doesn't change what "looks date-like" means, it just stops noisy-
    but-recoverable values from being undercounted as if they were
    unrecoverable.
    """
    if sample.empty:
        return 0.0
    str_sample = sample.astype(str).apply(datetime_std.normalize_natural_language_datetime)
    rate_dayfirst = pd.to_datetime(str_sample, dayfirst=True, format="mixed", errors="coerce").notna().mean()
    rate_monthfirst = pd.to_datetime(str_sample, dayfirst=False, format="mixed", errors="coerce").notna().mean()
    return max(rate_dayfirst, rate_monthfirst)


def _content_score_numeric_text(sample: pd.Series) -> float:
    """
    Catches a very common real-world Excel problem: a column of
    numbers that got stored as TEXT (Excel's little green
    "number stored as text" warning triangle) -- confirmed by testing
    against a real downloaded e-commerce dataset with exactly this
    issue in its Price/Quantity columns. Pandas reads such a column as
    plain text, so none of the other text-role checks (email, phone,
    date, address, name) are looking for it, and without this check
    it falls through to the free_text/categorical fallback -- meaning
    it would never get numeric cleaning (no outlier detection, no
    imputation) at all.

    Score is simply the fraction of sample values that successfully
    parse as a number. A genuinely numeric-as-text column scores very
    high (occasional contaminant values like "abd" or "four hundred"
    just slightly lower it, which is correct -- those ARE the real
    data-quality problems this column needs handled elsewhere in the
    pipeline, not something this scoring should hide).
    """
    if sample.empty:
        return 0.0
    coerced = pd.to_numeric(sample, errors="coerce")
    return coerced.notna().mean()


# Common job-title words. NOT an exhaustive list -- used as a content
# signal to tell "Marketing Specialist" / "Software Engineer" apart
# from a genuine person's name, since both are structurally identical
# (2 capitalised alphabetic words, no digits, often high uniqueness in
# a large dataset). Confirmed as a real, significant misclassification
# by testing against a real salary survey, where "Job title" was
# scored as a full_name column.
JOB_TITLE_WORDS = {
    "engineer", "manager", "specialist", "analyst", "director", "coordinator",
    "officer", "assistant", "librarian", "developer", "consultant", "executive",
    "representative", "administrator", "supervisor", "technician", "associate",
    "president", "lead", "head", "nurse", "teacher", "professor", "accountant",
    "designer", "architect", "scientist", "researcher", "writer", "editor",
    "attorney", "lawyer", "doctor", "chef", "clerk", "planner", "advisor",
    "recruiter", "controller", "strategist", "counsel", "paralegal",
}

# Column names that strongly suggest the content is a JOB TITLE, not a
# person's name, even though the values themselves can look
# structurally name-shaped.
FULL_NAME_DISQUALIFYING_NAME_HINTS = {"title", "job", "position", "role", "occupation", "designation"}


def _content_score_full_name(sample: pd.Series, column_name: str = "") -> float:
    """
    Looks like a person's name if it's 2-4 alphabetic tokens (allowing
    hyphens/apostrophes within a token, e.g. "Harper-Taylor", "O'Neil"),
    with no digits anywhere, MULTIPLIED by the sample's uniqueness
    ratio.

    The token-pattern check alone isn't enough to tell "Karl Kerr"
    apart from "Tesco PLC", a country name, OR a job title like
    "Marketing Specialist" -- all are just short runs of alphabetic
    tokens with (in a large dataset) similarly high uniqueness. Two
    extra checks handle the job-title case specifically: the column
    name is checked against a disqualifying-hint list ("title",
    "job", "position", etc.), and the sample's values are checked
    against a list of common job-title words -- if either signal
    fires, this returns 0.0 regardless of how name-shaped the text
    looks.
    """
    if sample.empty:
        return 0.0

    normalised_col = _split_camel_case(column_name).lower().replace("_", " ").replace("-", " ")
    col_tokens = set(re.findall(r"[a-z0-9]+", normalised_col))
    if col_tokens & FULL_NAME_DISQUALIFYING_NAME_HINTS:
        return 0.0

    def contains_job_title_word(v):
        words = re.findall(r"[a-z]+", str(v).lower())
        return any(w in JOB_TITLE_WORDS for w in words)

    if sample.apply(contains_job_title_word).mean() >= 0.15:
        return 0.0

    token_pattern = re.compile(r"^[A-Za-z][A-Za-z'-]*$")

    def looks_like_name(v):
        tokens = str(v).strip().split()
        if not (2 <= len(tokens) <= 4):
            return False
        return all(token_pattern.match(t) for t in tokens)

    pattern_match_rate = sample.apply(looks_like_name).mean()
    uniqueness = sample.nunique() / len(sample)
    return pattern_match_rate * uniqueness


def _content_score_address(sample: pd.Series) -> float:
    """
    Looks like an address if it mixes digits AND letters, AND EITHER
    the address separator appears at least twice (street/city/postcode
    as 3 genuinely separate segments) OR the value ends with something
    shaped like a UK postcode.

    BUG FIXED after real-file testing: the separator-count check alone
    was too strict for real messy addresses -- of 19 genuine address
    values from a real client file, only 6 had two or more separators;
    the rest had exactly one comma, or none at all ("89 King Street
    Manchester M2 4WU"), causing the WHOLE column to score too low to
    be recognised as address-like at all, and address splitting simply
    never ran. Checking for a trailing postcode shape instead is far
    more reliable for real data -- every one of those 19 values ends
    in a valid-looking UK postcode, regardless of how many commas it
    has. Module E's split_address_column() was upgraded alongside this
    fix to actually use that same postcode anchor when splitting, so
    detection and splitting stay consistent with each other, the same
    principle as the original 2-separator design.

    The separator-count check is KEPT as an alternative (not replaced)
    for addresses that don't end in a UK postcode shape at all (e.g.
    non-UK addresses) -- still requiring 2+ occurrences there, for the
    same reason as before: a single separator (e.g. "8 - 10 years") is
    structurally indistinguishable from a genuine one-separator address
    fragment, so it takes either a strong ending shape OR a doubled
    separator to count as a real signal, never just one weak one alone.
    """
    if sample.empty:
        return 0.0

    def looks_like_address(v):
        v = str(v)
        has_digit = any(ch.isdigit() for ch in v)
        has_letter = any(ch.isalpha() for ch in v)
        has_repeated_separator = any(v.count(sep) >= 2 for sep in [", ", " | ", " - ", "\n"])
        has_trailing_postcode = bool(_UK_POSTCODE_AT_END_RE.search(v))
        return has_digit and has_letter and (has_repeated_separator or has_trailing_postcode)

    return sample.apply(looks_like_address).mean()


def _uniqueness_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    return non_null.nunique() / len(non_null)


def guess_semantic_role(series: pd.Series, column_name: str) -> dict:
    """
    Guess the semantic role of ONE column, combining content evidence
    with a name-keyword hint.

    Returns {"role": ..., "confidence": 0-1, "content_score": 0-1} --
    confidence is kept alongside the winning role (rather than just
    returning the label) so the later question-generation step can
    decide "this is confident enough to just use" vs "this is shaky,
    worth confirming with the user" instead of treating every guess as
    equally certain.

    Numeric/boolean columns skip straight to a simpler numeric-specific
    check (id-like vs a genuine numeric measure) -- the text-content
    checks above (email regex, name-token pattern, etc.) don't apply
    to a column pandas already parsed as numbers.
    """
    basic_dtype = classify_basic_dtype(series)
    non_null = series.dropna()

    if len(non_null) == 0:
        return {"role": "empty", "confidence": 1.0, "content_score": 1.0}

    if basic_dtype == "boolean":
        return {"role": "categorical", "confidence": 1.0, "content_score": 1.0}

    if basic_dtype == "numeric":
        uniqueness = _uniqueness_ratio(series)
        name_hint = _name_hint_score(column_name, "id_key")
        # A continuous measure (order_amount, a price, a score) can be
        # JUST as unique as a real ID column -- uniqueness alone can't
        # separate them (confirmed by testing: order_amount and
        # customer_id had near-identical uniqueness ratios in our demo
        # data). The column NAME is what actually carries that signal
        # in practice, so it's the deciding gate here, with uniqueness
        # as a supporting check -- not the other way round.
        if name_hint == 1.0 and uniqueness > 0.5:
            return {"role": "id_key", "confidence": 0.85, "content_score": uniqueness}
        # No id-ish name, but still near-perfectly unique -- worth a
        # weaker id_key guess, BUT only on a big enough sample. On a
        # small file, an ordinary continuous measure (a price, a
        # score) can easily be 100% unique just by chance -- confirmed
        # by testing: a 6-row revenue column with plausible currency
        # values got flagged id_key purely because nothing repeated
        # yet. Below this row count, we don't have enough evidence
        # either way, so we default to the safer assumption
        # (numeric_measure) rather than risk excluding a real
        # measure from imputation/analysis.
        MIN_ROWS_FOR_UNNAMED_ID_GUESS = 20
        if uniqueness > 0.95 and len(non_null) >= MIN_ROWS_FOR_UNNAMED_ID_GUESS:
            return {"role": "id_key", "confidence": 0.5, "content_score": uniqueness}
        return {"role": "numeric_measure", "confidence": 0.9, "content_score": 1.0 - uniqueness}

    # basic_dtype == "text" from here on
    sample = _get_sample(series)

    candidates = {
        "email": _content_score_email(sample),
        "phone": _content_score_phone(sample),
        "date_or_datetime": _content_score_date(sample),
        "address": _content_score_address(sample),
        "full_name": _content_score_full_name(sample, column_name),
        "numeric_text": _content_score_numeric_text(sample),
        "country": _content_score_country(sample),
        "state": _content_score_state(sample),
    }

    scored = {}
    for role, content_score in candidates.items():
        if content_score >= MIN_CONTENT_TO_QUALIFY:
            name_score = _name_hint_score(column_name, role)
            scored[role] = {
                "final": CONTENT_WEIGHT * content_score + NAME_WEIGHT * name_score,
                "content_score": content_score,
            }

    if scored:
        best_role = max(scored, key=lambda r: scored[r]["final"])
        best = scored[best_role]
        if best["final"] >= MIN_FINAL_SCORE:
            return {
                "role": best_role,
                "confidence": round(best["final"], 2),
                "content_score": round(best["content_score"], 2),
            }

    # Nothing matched a specific pattern -- fall back to cardinality:
    # a text column repeating a small set of values is categorical
    # (e.g. company, country, region); one where almost every value is
    # unique and/or long is free text (e.g. notes) or an alphanumeric
    # ID code.
    #
    # We check BOTH the uniqueness RATIO and the absolute unique COUNT,
    # not just the ratio. Ratio alone is unreliable on small samples --
    # a genuinely categorical column (e.g. 4 regions across only 8
    # rows) can land right at 0.5 uniqueness by chance, which a
    # ratio-only threshold would misread as borderline. A low absolute
    # count of distinct values (regardless of how many rows there are)
    # is actually the stronger, more literal definition of
    # "categorical" -- few possible categories, each repeated.
    uniqueness = _uniqueness_ratio(series)
    unique_count = sample.nunique()
    avg_length = sample.astype(str).str.len().mean()
    id_name_hint = _name_hint_score(column_name, "id_key")

    if uniqueness > 0.9 and id_name_hint == 1.0:
        return {"role": "id_key", "confidence": 0.7, "content_score": round(uniqueness, 2)}
    if unique_count <= 15 and avg_length < 30:
        return {"role": "categorical", "confidence": 0.6, "content_score": round(1 - uniqueness, 2)}
    if uniqueness < 0.2:
        return {"role": "categorical", "confidence": 0.6, "content_score": round(1 - uniqueness, 2)}
    return {"role": "free_text", "confidence": 0.5, "content_score": round(uniqueness, 2)}


# ---------------------------------------------------------------------------
# 5. Date-ambiguity detection
# ---------------------------------------------------------------------------
# This is the piece that decides whether the dayfirst/monthfirst
# question is even worth asking for a given date-like column. Per the
# project spec, ambiguity is a NUMERIC-FORMAT problem only: a date
# written as "23 July 2026" is never ambiguous (the month is spelled
# out), and ISO dates ("2026-07-23") are never ambiguous (year comes
# first, so the month/day order after it is fixed by convention). Only
# a fully numeric date like "03/04/2026" -- where BOTH parts could
# plausibly be a day or a month (i.e. both <= 12) -- is genuinely
# ambiguous. This mirrors Module D's own documented limitation: a row
# where one part is > 12 resolves correctly no matter what dayfirst
# setting is chosen, because a month can't exceed 12.

# Matches a spelled-out month name, e.g. "08 October 2025" or
# "25 November 2025" -- if this matches, the value is NOT ambiguous,
# regardless of what else is in the string (e.g. a trailing time).
MONTH_NAME_DATE_PATTERN = re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}\b")

# Matches a purely numeric date with /, -, or . as the separator
# between all three parts, e.g. "03/04/2026", "2025-11-29",
# "27-12-23". Deliberately does NOT match on ":" (time separators),
# so this naturally skips past any time component that follows in the
# same string.
NUMERIC_DATE_PATTERN = re.compile(r"(\d{1,4})[/\-.](\d{1,4})[/\-.](\d{1,4})")


def detect_date_ambiguity(series: pd.Series) -> dict:
    """
    Check whether a date-like column actually contains genuinely
    ambiguous numeric dates -- and, if it does, whether the SAME
    column also contains unambiguous evidence of its own format that
    makes asking the user unnecessary.

    A value like "27/08/2026" can only be DD/MM/YYYY -- no month is
    27, so this single row proves the file's convention. If every
    unambiguous row in a column agrees on the same convention, that's
    used to auto-resolve dayfirst for the WHOLE column, and the
    dayfirst question is skipped entirely for it. This is genuinely
    different from guessing: Module D's rule ("never auto-guess
    dayfirst, a wrong guess silently corrupts data") is about picking
    an assumption with NO evidence -- this is instead reading direct,
    unambiguous proof that's already sitting in the same column,
    which is more reliable than asking a human to remember a
    convention, not less.

    Three possible outcomes once evidence is gathered:
      - Evidence is UNANIMOUS (every unambiguous row implies the same
        dayfirst value) -> auto-resolved, no question needed.
      - Evidence CONTRADICTS itself (some rows imply dayfirst=True,
        others imply dayfirst=False -- a genuinely inconsistent file,
        the exact scenario Module D's own docstring already warns
        can't be perfectly fixed) -> still asks, since picking either
        answer would be a real guess at that point.
      - NO unambiguous evidence exists anywhere in the column -> still
        asks, exactly as before.

    We scan every non-null value (not just a sample) -- unlike the
    semantic-role checks, this one directly gates whether we ask the
    user a question at all, so it needs to be exact, not estimated
    from a sample.
    """
    non_null = series.dropna().astype(str)

    total_checked = 0
    ambiguous_count = 0
    distinct_example = None   # preferred: day != month, a genuinely informative example
    fallback_example = None   # any ambiguous case, used only if no distinct one exists
    dayfirst_evidence = set()          # which dayfirst value(s) unambiguous rows imply
    dayfirst_evidence_example = None   # one real unambiguous value, for the report/log

    for value in non_null:
        if MONTH_NAME_DATE_PATTERN.search(value):
            total_checked += 1
            continue  # spelled-out month -- never ambiguous

        match = NUMERIC_DATE_PATTERN.search(value)
        if not match:
            continue  # not a recognisable date pattern at all -- not our concern here

        total_checked += 1
        part1, part2, part3 = match.groups()

        if len(part1) == 4:
            continue  # year-first (ISO-style) -- month/day order is fixed, unambiguous

        try:
            n1, n2 = int(part1), int(part2)
        except ValueError:
            continue

        if n1 <= 12 and n2 <= 12:
            ambiguous_count += 1

            if distinct_example is None or fallback_example is None:
                dayfirst_dt = pd.to_datetime(value, dayfirst=True, format="mixed", errors="coerce")
                monthfirst_dt = pd.to_datetime(value, dayfirst=False, format="mixed", errors="coerce")
                dayfirst_str = dayfirst_dt.strftime("%-d %B %Y") if pd.notna(dayfirst_dt) else None
                monthfirst_str = monthfirst_dt.strftime("%-d %B %Y") if pd.notna(monthfirst_dt) else None

                info = {
                    "original_value": value,
                    "if_dayfirst": dayfirst_str,
                    "if_monthfirst": monthfirst_str,
                }

                if fallback_example is None:
                    fallback_example = info
                if distinct_example is None and dayfirst_str != monthfirst_str:
                    distinct_example = info

        elif n1 <= 31 and n2 <= 31:
            # Exactly one of the two parts exceeds 12 -- a month can
            # never be > 12, so that part MUST be the day, which
            # pins down the whole column's convention for this row.
            # (Both > 12 would mean neither part could be a valid
            # month OR day under any interpretation -- not usable
            # evidence, just skipped.)
            if n1 > 12:
                dayfirst_evidence.add(True)   # part1 can only be a day -> DD-MM-YYYY
            elif n2 > 12:
                dayfirst_evidence.add(False)  # part2 can only be a day -> MM-DD-YYYY
            if dayfirst_evidence_example is None:
                dayfirst_evidence_example = value

    inferred_dayfirst = None
    if len(dayfirst_evidence) == 1:
        inferred_dayfirst = dayfirst_evidence.pop()

    return {
        "date_like_values_checked": total_checked,
        "ambiguous_count": ambiguous_count,
        # Only actually ASK when there's no reliable way to resolve it
        # from the column's own data: either no ambiguity at all, or
        # ambiguity present but auto-resolved via unanimous evidence.
        "needs_dayfirst_question": ambiguous_count > 0 and inferred_dayfirst is None,
        "example": distinct_example or fallback_example,
        "inferred_dayfirst": inferred_dayfirst,
        "inferred_dayfirst_example": dayfirst_evidence_example if inferred_dayfirst is not None else None,
    }


# ---------------------------------------------------------------------------
# Updated profile_column / profile_dataframe -- now includes semantic role
# ---------------------------------------------------------------------------

def _get_sample_values(series: pd.Series, n: int = 3) -> list:
    """
    Grab a few real, distinct non-null example values from the column
    -- used to make clarifying questions concrete ("we think this is
    an ID column, e.g. 'A1023', 'A1024' -- is that right?") instead of
    asking about a column abstractly. Each value is truncated if long,
    so a free-text-ish value can't blow up a question's length.
    """
    non_null = series.dropna()
    if non_null.empty:
        return []
    examples = non_null.astype(str).drop_duplicates().head(n).tolist()
    return [v if len(v) <= 40 else v[:37] + "..." for v in examples]


# ---------------------------------------------------------------------------
# 6. Statistical/ML variable type (continuous / binary / categorical / ordinal)
# ---------------------------------------------------------------------------
# This is a DIFFERENT question from semantic role (role asks "what does
# this column REPRESENT" -- email, country, a date; variable type asks
# "how should its VALUES be treated statistically" -- a real-valued
# measurement, a two-category flag, an unordered set of categories, or
# an ordered set of categories). Confirmed as a real, consequential gap
# by a real client dataset (UCI-style hepatitis data): "Sex" stored as
# 1/2 and "Steroid" stored as 1/2 (yes/no) both get correctly typed
# "numeric" by pandas and land in role "numeric_measure" -- but they
# are NOT continuous measurements. KNN/mean/median imputation on a
# binary 1/2 column can produce a nonsense value like 1.4, and outlier
# statistics computed on it are meaningless.

def guess_variable_type(series: pd.Series, role: str, basic_dtype: str) -> dict:
    """
    Suggests a column's variable type for MISSING-VALUE HANDLING and
    (opt-in) machine-learning encoding purposes: "continuous", "binary",
    "categorical" (3+ unordered categories), or "ordinal" (3+ categories
    with a genuine, meaningful order, e.g. a severity scale).

    DELIBERATELY CONSERVATIVE: only "binary" (exactly 2 distinct
    non-null values) is confident enough to suggest automatically.
    "Ordinal" genuinely can't be told apart from "categorical" (or even
    "continuous", for something like a small-range count of children)
    using the data's VALUES alone -- telling them apart needs to know
    whether the categories have a real order, which is domain
    knowledge, not something inferable from the numbers themselves. So
    this always returns a SUGGESTION with an honest confidence level --
    meant to be reviewed and corrected by the user (in the opt-in
    Machine Learning Ready section), never silently acted on for
    something this consequential.
    """
    non_null = series.dropna()
    if non_null.empty:
        return {"variable_type": "continuous", "variable_type_confidence": 0.0}

    n_unique = non_null.nunique()

    if n_unique == 2:
        # The one case confident enough to flag strongly regardless of
        # dtype -- a column with only two distinct values (1/2, Y/N,
        # True/False) is a binary flag, full stop, whether it's stored
        # as numbers or text.
        return {"variable_type": "binary", "variable_type_confidence": 0.95}

    if role == "categorical":
        return {"variable_type": "categorical", "variable_type_confidence": 0.6}

    if basic_dtype == "numeric":
        # Could be a genuine continuous measure (age, salary) OR an
        # unordered/ordered small-integer code (a 1-5 severity score,
        # a region code) -- content alone can't safely tell these
        # apart. Stays "continuous" (the safe, current default
        # behaviour) but with LOW confidence whenever there are few
        # enough distinct values that it's clearly worth a second look
        # -- e.g. a column with only 4 distinct integer values is far
        # more likely to be a code than a genuinely continuous measure.
        confidence = 0.9 if n_unique > 10 else 0.5
        return {"variable_type": "continuous", "variable_type_confidence": confidence}

    return {"variable_type": "categorical", "variable_type_confidence": 0.6}


def profile_column(series: pd.Series, column_name: str = None) -> dict:
    """
    Build the full profile for a single column: basic dtype, missing-
    value stats, disguised-missing markers, semantic role, AND a few
    real sample values.

    column_name is now required for meaningful role detection (the
    name-hint half of guess_semantic_role() needs it) -- kept optional
    with a fallback so this function still works if ever called on a
    bare Series with no name context, but profile_dataframe() below
    always supplies it properly.
    """
    n = len(series)
    missing_count = int(series.isna().sum())
    name = column_name if column_name is not None else (series.name or "")

    role_info = guess_semantic_role(series, name)

    # Date-ambiguity detection only makes sense for columns we've
    # already decided look date-like -- running it on an arbitrary
    # text or numeric column would be meaningless (and wasteful).
    date_ambiguity = None
    if role_info["role"] == "date_or_datetime":
        date_ambiguity = detect_date_ambiguity(series)

    basic_dtype = classify_basic_dtype(series)
    variable_type_info = guess_variable_type(series, role_info["role"], basic_dtype)

    return {
        "dtype": basic_dtype,
        "missing_count": missing_count,
        "missing_percent": round((missing_count / n) * 100, 2) if n else 0.0,
        "disguised_missing_markers": detect_disguised_missing(series),
        "semantic_role": role_info["role"],
        "role_confidence": role_info["confidence"],
        "variable_type": variable_type_info["variable_type"],
        "variable_type_confidence": variable_type_info["variable_type_confidence"],
        "date_ambiguity": date_ambiguity,
        "sample_values": _get_sample_values(series),
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Run profile_column() across every column in the DataFrame.

    Returns {column_name: profile_dict}. This is the object the rest
    of the app (question generation, advanced-settings defaults) will
    read from -- kept as a plain dict (not a custom class) so it's
    trivially JSON-serialisable for a Streamlit session or a future
    API layer.
    """
    return {col: profile_column(df[col], column_name=col) for col in df.columns}


def print_profile(profile: dict) -> None:
    """
    Human-readable dump, for testing/debugging this module on its own
    before the Streamlit UI exists to display it properly.
    """
    for col, info in profile.items():
        print(f"\n{col}")
        print(f"  dtype: {info['dtype']}  |  role: {info['semantic_role']} (confidence {info['role_confidence']})")
        print(f"  missing: {info['missing_count']} ({info['missing_percent']}%)")
        if info["disguised_missing_markers"]:
            markers = ", ".join(
                f"'{m}' x{c}" for m, c in info["disguised_missing_markers"].items()
            )
            print(f"  possible disguised-missing markers found: {markers}")
        if info.get("date_ambiguity") and info["date_ambiguity"]["needs_dayfirst_question"]:
            amb = info["date_ambiguity"]
            ex = amb["example"]
            print(
                f"  DAYFIRST QUESTION NEEDED: {amb['ambiguous_count']}/{amb['date_like_values_checked']} "
                f"values are ambiguous. Example: '{ex['original_value']}' -> "
                f"{ex['if_dayfirst']} (dayfirst) or {ex['if_monthfirst']} (monthfirst)?"
            )


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")

    # Colab-friendly hardcoded path, same convention as every other module.
    filepath = "dirty_customer_data.xlsx"

    df = ingestion.load_data(filepath)
    profile = profile_dataframe(df)
    print_profile(profile)
