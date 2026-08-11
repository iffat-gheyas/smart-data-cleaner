"""
Module J: Dynamic Question Generation (Phase 2 — new)
---------------------------------------------------------
Takes the output of Module I's profile_dataframe() and decides which
clarifying questions, if any, are actually worth asking the user for
THIS specific uploaded file. This is the piece that keeps the app from
becoming a generic 15-item form -- most of the time, most questions
simply won't be generated, because most files won't trigger them.

Scope note (being upfront, per the project's working style): this
module builds the THREE question types the Phase 2 spec explicitly
names as examples:
  1. Dayfirst / monthfirst confirmation (only if genuinely ambiguous
     numeric dates were found -- Module I, sub-step 3)
  2. Disguised-missing-value confirmation (only if a suspicious
     repeated marker like "?" was found -- Module I, sub-step 1)
  3. Duplicate-detection column choice (always relevant when a
     sensible candidate column exists; pre-filled with a best guess,
     never forced)

A natural FOURTH type -- asking the user to confirm a column's
semantic role when Module I's confidence score is low (e.g. "we think
'ref_no' is an ID column, is that right?") -- is NOT built here. It's
a reasonable extension, but it widens scope beyond what the spec
named, so it's left as a clean follow-on rather than folded in now.

ADDENDUM (generic-mode measurement feature): two more question types
added on top of the above -- weight/length measurement-role
confirmation and target-unit selection (Sections 5 below), built on
Module 45's detection. Both follow the exact same "only ask if the
data genuinely needs it" principle as everything above.

ADDENDUM (generic-mode currency feature): a currency-standardisation
question chain (Section 7 below), built on Module 47's detection.
Mirrors the weight/length chain's shape closely, with ONE necessary
difference: exchange rates aren't fixed constants the way unit ratios
are, so there's a genuinely NEW question type here, "numeric_input" --
the app.py rendering layer needs one small addition to actually show
it (a number entry box), which doesn't exist for any other question
type in this module yet.

DESIGN: every question is returned as a plain dict, not tied to any
specific UI widget. That keeps this module UI-framework-agnostic --
the Streamlit layer (built later) just reads these dicts and decides
how to render "type": "single_select" / "yes_no" / "multi_select" /
"numeric_input" as actual widgets. This mirrors the same "logic
separate from presentation" principle Module A used for
diagnostic_summary() vs print_summary().
"""

from importlib import import_module

import pandas as pd

unit_std = import_module("45_unit_standardisation")
curr_std = import_module("47_currency_standardisation")
text_num = import_module("49_text_number_parsing")
duration_std = import_module("50_duration_standardisation")
phone_std = import_module("52_phone_standardisation")
email_std = import_module("53_email_standardisation")

# ---------------------------------------------------------------------------
# 0. Low-confidence role confirmation
# ---------------------------------------------------------------------------
# The follow-on question type flagged (but deliberately deferred) when
# Module I and this module were first built. Only applies to roles
# where getting it wrong actually changes what the pipeline DOES --
# full_name/address get split, date_or_datetime gets parsed, email/
# phone get validated, id_key gets excluded from imputation/outlier
# handling. categorical / free_text / numeric_measure are deliberately
# EXCLUDED: their confidence scores (from Module I) are closer to
# fixed baseline values than genuine per-column uncertainty, so
# thresholding on them would generate a confirmation question for
# almost every ordinary column in a file -- exactly the "generic
# 15-item form" the whole profiling system exists to avoid.

STRUCTURAL_ROLES_TO_CONFIRM = {
    "full_name", "address", "date_or_datetime", "email", "phone", "id_key", "numeric_text", "state",
}
ROLE_CONFIRMATION_THRESHOLD = 0.75

ROLE_DISPLAY_NAMES = {
    "email": "an email address",
    "phone": "a phone number",
    "full_name": "a full name",
    "address": "an address",
    "date_or_datetime": "a date/time",
    "id_key": "an ID/reference column",
    "numeric_text": "numbers (stored as text)",
    "state": "a US state name",
}


def generate_role_confirmation_questions(profile: dict) -> list:
    """
    One yes/no question per column whose guessed role is BOTH one of
    the structural roles above AND below-threshold confidence.

    Kept as a simple yes/no (confirm the guess, or reject it) rather
    than offering the full role taxonomy as a picklist -- consistent
    with the project's closed-question-only design: the user doesn't
    need to know our internal role vocabulary, just whether our guess
    matches what they know about their own data. Rejecting a role
    demotes that column to "no special processing" (Module K skips
    every role-based step for it) rather than trying to guess a
    replacement role, which would just be swapping one guess for
    another.

    Defaults to "Yes" -- even a low-confidence guess is still usually
    the best available guess, so it's a reasonable starting point, not
    a coin flip being silently resolved.
    """
    questions = []

    for col, info in profile.items():
        role = info["semantic_role"]
        confidence = info["role_confidence"]

        if role not in STRUCTURAL_ROLES_TO_CONFIRM or confidence >= ROLE_CONFIRMATION_THRESHOLD:
            continue

        role_label = ROLE_DISPLAY_NAMES.get(role, role)
        samples = info.get("sample_values") or []
        example_text = f" (e.g. {', '.join(repr(s) for s in samples)})" if samples else ""

        questions.append({
            "id": f"confirm_role__{col}",
            "type": "yes_no",
            "question_text": f"We think '{col}' contains {role_label}{example_text} — is that right?",
            "options": [
                {"label": f"Yes, treat it as {role_label}", "value": True},
                {"label": "No, treat it as plain text (skip special processing)", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
            "guessed_role": role,
        })

    return questions


# ---------------------------------------------------------------------------
# 1. Dayfirst / monthfirst confirmation
# ---------------------------------------------------------------------------

def generate_dayfirst_question(profile: dict) -> dict:
    """
    ONE job-level question (not one per column), because Sprint 1's
    pipeline applies a single DAYFIRST setting across every date
    column (see main.py's CONFIGURATION section) -- date columns in
    the same file almost always come from the same source system and
    share the same convention.

    Picks the column with the MOST ambiguous values to supply the
    example shown to the user (the most representative case), but
    lists every affected column so the user knows the setting will
    apply to all of them.

    Returns None if no date column in the profile actually needs this
    question (Module I already worked that out per-column).
    """
    ambiguous_cols = {
        col: info["date_ambiguity"]
        for col, info in profile.items()
        if info.get("date_ambiguity") and info["date_ambiguity"]["needs_dayfirst_question"]
    }

    if not ambiguous_cols:
        return None

    best_col = max(ambiguous_cols, key=lambda c: ambiguous_cols[c]["ambiguous_count"])
    example = ambiguous_cols[best_col]["example"]

    return {
        "id": "dayfirst",
        "type": "single_select",
        # Combines the plain-language interpretation ("3 February 2025")
        # WITH the format label ("DD-MM-YYYY") and the original raw
        # value, in both the question text and each option -- covers
        # someone who knows date-format terminology, someone who
        # doesn't, and someone who just wants to see it worked through
        # on a real example. Revised after real-user testing showed
        # the plain-interpretation-only version ("is that 3 February
        # or 2 March?") caused a moment of genuine confusion before
        # the reader connected it back to which format their file
        # actually uses.
        "question_text": (
            f"Your date column(s) ({', '.join(ambiguous_cols.keys())}) contain ambiguous "
            f"dates like '{example['original_value']}'. Please choose the format used in "
            f"your file:"
        ),
        "options": [
            {
                "label": f"DD-MM-YYYY  ({example['original_value']} = {example['if_dayfirst']})",
                "value": True,
            },
            {
                "label": f"MM-DD-YYYY  ({example['original_value']} = {example['if_monthfirst']})",
                "value": False,
            },
        ],
        # Deliberately NO default pre-selected. Sprint 1's date module
        # spec is explicit that dayfirst is "never auto-guessed, since
        # a wrong guess silently corrupts data" -- pre-selecting an
        # option (even a "sensible" one like UK-style dayfirst=True)
        # would undermine that principle by nudging the user toward an
        # unverified assumption instead of an active, informed choice.
        "default": None,
        "affected_columns": list(ambiguous_cols.keys()),
    }


# ---------------------------------------------------------------------------
# 2. Disguised-missing-value confirmation
# ---------------------------------------------------------------------------

def generate_disguised_missing_questions(profile: dict) -> list:
    """
    ONE question PER affected column, not combined into one -- unlike
    dayfirst, there's no reason to assume "?" means the same thing in
    'country' as it does in 'notes'.

    Presented as a MULTI-SELECT of the specific detected marker
    values, not a blanket yes/no -- revised after real-user testing
    highlighted a real problem with the yes/no version: it forces an
    all-or-nothing choice across every detected marker in a column,
    but a value like "999" might genuinely be a placeholder in one
    dataset and a real, legitimate value in another. A per-value
    multi-select lets the user keep some markers as missing and
    reject others in the SAME column, rather than being forced to
    accept or reject the whole set together. Every detected marker
    starts pre-checked (Module I only flagged values that are already
    repeated and placeholder-looking, so "yes" remains a sensible
    starting point) but each is independently uncheckable.

    Returns an empty list if no column has any disguised-missing
    markers.
    """
    questions = []

    for col, info in profile.items():
        markers = info.get("disguised_missing_markers") or {}
        if not markers:
            continue

        options = [
            {"label": f"'{marker}'  ({count} occurrence{'s' if count != 1 else ''})", "value": marker}
            for marker, count in markers.items()
        ]

        questions.append({
            "id": f"disguised_missing__{col}",
            "type": "multi_select",
            "question_text": (
                f"'{col}' contains these repeated values that might mean \"missing\" rather than "
                f"a real value. Select any that should be treated as missing:"
            ),
            "options": options,
            "default": [marker for marker in markers.keys()],
            "affected_columns": [col],
            "markers_found": markers,
        })

    return questions


# ---------------------------------------------------------------------------
# 3. Duplicate-detection column choice
# ---------------------------------------------------------------------------

# Preference order for the pre-filled best guess. Email is the
# strongest real-world signal for "same customer" (per the spec's own
# example), followed by phone, then a genuine ID column, then full
# name (weakest -- two different people can share a name, unlike an
# email or phone number).
# Sentinel option value for "match on every column, not a specific
# subset" -- its own visible entry in the multi-select rather than
# requiring the client to manually tick every single column one by one
# to express the same thing. Selecting it still means "yes, run
# duplicate detection" (unlike leaving the question empty, which now
# skips detection entirely -- see generate_duplicate_subset_question's
# own docstring) -- Module 12 treats it as "no subset filter", relying
# on the exact-duplicate baseline alone.
ALL_COLUMNS_DUPLICATE_MATCH = "__all_columns_exact_match__"


def generate_duplicate_subset_question(profile: dict) -> dict:
    """
    "Which columns make two rows the same real-world record" is a
    BUSINESS-RULE decision that depends entirely on what the dataset
    IS about -- not on any column's detected role. Real user feedback
    corrected an earlier version of this function that only offered
    email/phone/id/full_name as choices: that's guessing at "likely
    identifiers" using role alone, which misses genuine identifiers
    the role-detector doesn't specifically flag (e.g. address), and
    more fundamentally can't know whether the SAME person appearing
    multiple times is a duplicate at all -- in an orders dataset, one
    customer legitimately placing five orders is not a data-quality
    problem, and only someone who understands the dataset's context
    can tell the difference.

    So every non-empty column is offered as a selectable option here
    -- no role-based filtering.

    DEFAULT CHANGED per explicit client request: this used to pre-fill
    a "best guess" column (using the role-priority order below) so
    duplicate detection ran automatically unless the client opted out.
    It now defaults to NOTHING selected at all, and an empty selection
    means "skip duplicate detection entirely" (Module 12's
    duplicate_detection_enabled flag) -- duplicate removal is no longer
    something that happens unless you notice and disable it; it's
    something that happens only once you've actively told the tool
    which columns make two rows the same real-world record. Selecting
    "Exact match — all columns" is the explicit way to say "yes, detect
    duplicates, using every column" without ticking each one by hand.

    Returns None only if there are literally no usable columns at all
    (e.g. every column is "empty" and will be dropped anyway).
    """
    candidates = [
        col for col, info in profile.items()
        if info["semantic_role"] != "empty"
    ]

    if not candidates:
        return None

    return {
        "id": "duplicate_subset_columns",
        "type": "multi_select",
        "question_text": (
            "Which column(s) should we use to detect duplicate records? "
            "Select any column(s) that should match for two rows to count as the same "
            "real-world record (e.g. the same customer, order, or product) — not every "
            "repeat is necessarily a duplicate, so pick based on what makes sense for "
            "this specific data. Or pick 'Exact match — all columns' if two rows should "
            "only count as duplicates when every single column is identical. "
            "Leave this empty if you don't want duplicate detection at all."
        ),
        "options": [
            {"label": "Exact match — all columns", "value": ALL_COLUMNS_DUPLICATE_MATCH}
        ] + [{"label": col, "value": col} for col in candidates],
        "default": [],
        "affected_columns": candidates,
    }


# ---------------------------------------------------------------------------
# 4. Text-number confirmation questions
# ---------------------------------------------------------------------------
# Deliberately placed FIRST among the unit-related question sections
# (this file's Sections 4/5/6), even though it's numbered after
# duplicate-detection -- these questions are answered and applied
# BEFORE the weight/length and currency questions make sense of the
# result, exactly mirroring Module 12's own required execution order
# (text-number conversion must run before Modules 45-48, since those
# expect a digit at the start of a value). Display order in
# generate_questions() below follows this same logic.

def generate_text_number_confirmation_questions(detected_text_number_columns: dict) -> list:
    """
    One yes/no confirmation per flagged column -- "we found values
    written as words instead of digits here, convert them?" Always
    asked (no confidence threshold to skip it) since, unlike a
    role guess, there's no harm in asking even when the signal is
    strong -- accepting the default costs nothing, and this is exactly
    the kind of change (rewriting cell values) worth an explicit nod
    rather than a silent auto-apply.

    Carries an "info_text" field -- rendered as the small (i) tooltip
    the client asked for specifically, explaining the ONE thing this
    step promises: if a value has a unit attached ("fifty pounds"),
    only the number part changes ("50 pounds") -- the unit is kept
    exactly as written, never reinterpreted or converted itself (that
    happens later, in the weight/length and currency features, and
    only if THIS conversion has already turned the words into digits).
    """
    questions = []

    for col, info in detected_text_number_columns.items():
        examples = info["example_conversions"]
        example_str = ", ".join(f"{repr(raw)} -> {conv!r}" for raw, conv in examples[:3])

        questions.append({
            "id": f"convert_text_numbers__{col}",
            "type": "yes_no",
            "question_text": (
                f"'{col}' has some values written as words instead of numbers "
                f"(e.g. {example_str}). Convert these to numeric form?"
            ),
            "info_text": (
                "If a value includes a unit or currency (e.g. 'fifty pounds'), only the "
                "NUMBER part is converted (-> '50 pounds') -- the unit itself is left exactly "
                "as written and is not interpreted here. Unit and currency standardisation, if "
                "needed, happens as a separate step after this one."
            ),
            "options": [
                {"label": "Yes, convert word-form numbers to digits", "value": True},
                {"label": "No, leave these values as written", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def _build_text_number_preview(df: pd.DataFrame, detected_text_number_columns: dict) -> pd.DataFrame:
    """
    Builds a PREVIEW copy of df with every flagged text-number column's
    convertible values already turned into digits -- used ONLY so that
    the weight/length and currency detection below sees the same
    post-conversion picture Module 12 will actually be working with,
    never returned or persisted anywhere as real data.

    WHY THIS MATTERS: Module 45/47's detection relies on parse_quantity
    / parse_currency_value, both of which require a leading DIGIT --
    "fifty kg" contributes NOTHING to their weight/currency signal,
    same as any other unparseable text. Without this preview, a column
    that's mostly word-form values (e.g. "fifty kg", "sixty kg", with
    only a couple of already-digit "45kg" entries) would show Module 45
    a much weaker, less complete picture than what will actually exist
    once Module 49 converts the words -- potentially even missing a
    genuinely mixed-unit situation entirely if the only digit-based
    samples all happen to share one unit.

    KNOWN EDGE CASE, same shape as others already documented in this
    file: this preview OPTIMISTICALLY assumes every text-number
    confirmation will be answered "Yes" (the default). If the client
    answers "No" for a column, Module 46/48 will correctly detect
    against the ACTUAL (still word-form) data at apply time -- so the
    worst case is a measurement/currency question here having been
    generated against a slightly rosier picture than what ends up being
    converted, never a wrong conversion applied.
    """
    if not detected_text_number_columns:
        return df

    preview = df.copy()
    for col in detected_text_number_columns:
        preview[col] = preview[col].apply(
            lambda v: text_num.convert_leading_word_number(v)[1] if pd.notna(v) else v
        )
    return preview


# ---------------------------------------------------------------------------
# 5. Weight/length measurement questions
# ---------------------------------------------------------------------------
# Two SEPARATE questions, deliberately: "is this column a measurement at
# all" is a different decision from "which unit do you want it in" -- a
# column can need the first without the second (e.g. a weight column that's
# ALREADY entirely in kg needs no conversion, just confirmation of what it
# is, for the report), and in principle could need the second without ever
# asking the first (see the confidence threshold below).

# Re-uses the SAME threshold and reasoning as ROLE_CONFIRMATION_THRESHOLD
# above: Module 45's content-based detection (real "5kg"/"10cm"-style unit
# tokens actually found in the values) produces confidence >= 0.75, which
# is strong enough evidence to skip asking. Only the NAME-only detections
# (bare numbers, guessed purely from a column called "Weight" or, weaker
# still, "Size") fall below 0.75 and get a confirmation question -- which
# is exactly the case that needs one, since a name like "Size" is
# genuinely ambiguous (shoe size? package size? family size?) in a way
# actual embedded unit text never is.
MEASUREMENT_CONFIRMATION_THRESHOLD = 0.75

MEASUREMENT_TYPE_LABELS = {"weight": "weight", "length": "length"}

UNIT_DISPLAY_LABELS = {
    "kg": "Kilograms (kg)", "g": "Grams (g)", "lb": "Pounds (lb)",
    "oz": "Ounces (oz)", "tonne": "Tonnes (t)", "stone": "Stone (st)",
    "mm": "Millimetres (mm)", "cm": "Centimetres (cm)", "m": "Metres (m)",
    "km": "Kilometres (km)", "in": "Inches (in)", "ft": "Feet (ft)",
    "yd": "Yards (yd)", "mi": "Miles (mi)",
}

# The full picklist offered for each measurement type -- NOT limited to
# only the units actually found in the column, since the client may want
# to standardise on a unit that isn't even present yet (e.g. a dataset
# that's all in lbs, but the client wants everything in kg for a
# different downstream system).
WEIGHT_TARGET_UNIT_OPTIONS = ["kg", "g", "lb", "oz", "tonne", "stone"]
LENGTH_TARGET_UNIT_OPTIONS = ["mm", "cm", "m", "km", "in", "ft", "yd", "mi"]


def generate_measurement_role_questions(detected_measurements: dict) -> list:
    """
    One yes/no question per column Module 45 flagged, but ONLY where
    confidence is below MEASUREMENT_CONFIRMATION_THRESHOLD -- see the
    reasoning above. Mirrors generate_role_confirmation_questions()'s
    shape exactly (same yes/no-with-default-True pattern) so the
    Streamlit layer renders both identically without special-casing.
    """
    questions = []

    for col, info in detected_measurements.items():
        if info["confidence"] >= MEASUREMENT_CONFIRMATION_THRESHOLD:
            continue

        type_label = MEASUREMENT_TYPE_LABELS[info["measurement_type"]]

        questions.append({
            "id": f"confirm_measurement__{col}",
            "type": "yes_no",
            "question_text": (
                f"We think '{col}' contains {type_label} measurements — is that right?"
            ),
            "options": [
                {"label": f"Yes, treat it as a {type_label} measurement", "value": True},
                {"label": "No, treat it as a plain number (skip unit conversion)", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
            "guessed_measurement_type": info["measurement_type"],
        })

    return questions


def generate_measurement_standardize_questions(detected_measurements: dict) -> list:
    """
    The gate the client asked for explicitly: "do you want to convert
    this column into a single unit?" -- asked SEPARATELY from, and
    BEFORE, both the target-unit and unresolved-unit questions, since
    those two only matter if the answer here is yes.

    Only generated when there's actually a real decision to make (see
    _needs_standardization) -- a column that's already clean numeric
    values gets no gate, since there's nothing to opt in or out of.

    If the client answers "No": Module 46 leaves the column COMPLETELY
    untouched -- original text values like "5kg", "10 lbs" stay exactly
    as typed, no numeric conversion, no companion column added. This is
    the direct fix for "keep number and unit in the same column."

    Default is True (yes, standardize) -- the whole point of this
    feature is producing a usable numeric column for a data-cleaning
    service, so that's the expected common case; "No" is an explicit,
    deliberate opt-out for a client who wants to keep the raw text.

    KNOWN EDGE CASE, same shape as elsewhere in this file: this and the
    target-unit / unresolved-unit questions are all generated together
    in one pass. If the client says "No" here, Module 46 is what
    actually respects that and skips conversion -- so the worst case is
    the target-unit question having been shown unnecessarily, never a
    conversion applied against the client's wishes.
    """
    questions = []

    for col, info in detected_measurements.items():
        if not unit_std.needs_standardization(info):
            continue

        type_label = MEASUREMENT_TYPE_LABELS[info["measurement_type"]]

        questions.append({
            "id": f"standardize_units__{col}",
            "type": "yes_no",
            "question_text": (
                f"'{col}' has {type_label} values written in different formats/units. "
                f"Do you want to convert all values in this column into a single unit?"
            ),
            "options": [
                {"label": "Yes, standardise this column", "value": True},
                {"label": "No, leave the original values exactly as they are", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_measurement_target_unit_questions(detected_measurements: dict) -> list:
    """
    Only generated for a column where Module 45 found GENUINELY MIXED
    units in the actual values (2+ distinct units) -- this is the real
    "different units in different rows" data-quality problem the
    client asked to have fixed. A column that's already entirely in one
    unit doesn't need this question at all; it's already consistent.

    KNOWN EDGE CASE (same shape as the one documented in
    generate_questions() below, for the plain role-confirmation
    question): this question is generated even for a column whose
    measurement role hasn't been confirmed yet in this same batch, if
    that column's confidence was below threshold. If the client answers
    "No, not a measurement" to the role question, Module 12's config
    builder is what actually respects that rejection and skips
    conversion -- so the worst case here is one redundant question
    shown, never a wrong conversion applied.

    Default pre-fill is the MOST COMMON unit already present in that
    column -- converting everything else to match the majority, rather
    than to an arbitrary unit, is the smallest-footprint sensible
    starting point, fully overridable.
    """
    questions = []

    for col, info in detected_measurements.items():
        detected_units = info["detected_units"]
        if len(detected_units) < 2:
            continue

        measurement_type = info["measurement_type"]
        type_label = MEASUREMENT_TYPE_LABELS[measurement_type]
        target_options = (
            WEIGHT_TARGET_UNIT_OPTIONS if measurement_type == "weight" else LENGTH_TARGET_UNIT_OPTIONS
        )
        units_found_str = ", ".join(sorted(detected_units.keys()))
        default_unit = max(detected_units, key=detected_units.get)

        questions.append({
            "id": f"target_unit__{col}",
            "type": "single_select",
            "question_text": (
                f"'{col}' contains {type_label} values in multiple units ({units_found_str}). "
                f"Which unit would you like all values expressed in?"
            ),
            "options": [
                {"label": UNIT_DISPLAY_LABELS[u], "value": u} for u in target_options
            ],
            "default": default_unit,
            "affected_columns": [col],
            "detected_units": detected_units,
        })

    return questions


def generate_measurement_default_gate_questions(df: pd.DataFrame, detected_measurements: dict) -> list:
    """
    Mirrors generate_currency_default_gate_questions() exactly, added
    for consistency after the currency feature got this treatment
    (client feedback: a per-value dropdown for EVERY distinct bare
    number is more friction than needed when the client has one
    consistent answer, e.g. "any unlabelled weight in this file is
    always in kg").

    Scoped to is_bare=True values ONLY (see Module 45's
    find_unresolved_unit_details) -- a value with an unrecognised UNIT
    TOKEN (e.g. "8 kgz", a likely typo) is never covered by a default,
    since a blanket "no unit mentioned" answer can't resolve what a
    garbled mention actually meant; those always go through
    generate_measurement_unresolved_unit_questions unconditionally.
    """
    questions = []

    for col, info in detected_measurements.items():
        details = unit_std.find_unresolved_unit_details(df, col, info)
        bare_details = [d for d in details if d["is_bare"]]
        if not bare_details:
            continue

        measurement_type = info["measurement_type"]
        type_label = MEASUREMENT_TYPE_LABELS[measurement_type]
        target_options = (
            WEIGHT_TARGET_UNIT_OPTIONS if measurement_type == "weight" else LENGTH_TARGET_UNIT_OPTIONS
        )
        detected_units = info["detected_units"]
        # Prefer the magnitude-aware suggestion (see Module 45's own
        # docstring for the real bug this fixes -- a bare "1.7" in a
        # Height column defaulting to "cm" instead of the obviously-
        # correct "m") -- falls back to the plain frequency-based pick
        # only when there isn't enough data to compare magnitudes at
        # all (e.g. no bare values, or no recognised values elsewhere
        # in the column to compare against).
        default_unit = unit_std.suggest_default_unit_by_magnitude(df, col, info)
        if default_unit is None:
            default_unit = max(detected_units, key=detected_units.get) if detected_units else None
        gate_id = f"has_default_unit__{col}"

        questions.append({
            "id": gate_id,
            "type": "yes_no",
            "question_text": (
                f"Is there a DEFAULT unit that should be used whenever no unit is mentioned at "
                f"all in '{col}' (e.g. a bare value like '{bare_details[0]['raw_value']}')?"
            ),
            "info_text": (
                "This only covers values with NO unit written at all. Values with an unclear or "
                "misspelled unit (e.g. '8 kgz') are always handled one-by-one below, regardless "
                "of your answer here."
            ),
            "options": [
                {"label": "Yes, there's a consistent default", "value": True},
                {"label": "No, handle these case-by-case instead", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

        questions.append({
            "id": f"default_unit__{col}",
            "type": "single_select",
            "question_text": f"Which {type_label} unit should be the default for '{col}' when none is mentioned?",
            "options": [
                {"label": UNIT_DISPLAY_LABELS[u], "value": u} for u in target_options
            ],
            "default": default_unit,
            "affected_columns": [col],
            "depends_on": {"question_id": gate_id, "hide_if_value": False},
        })

    return questions


def generate_measurement_unresolved_unit_questions(df: pd.DataFrame, detected_measurements: dict) -> list:
    """
    REDESIGNED for consistency with the currency feature's per-value
    rework (client feedback there: one shared question per column is
    wrong when different unresolved values plausibly need different
    answers). Now asks ONE question PER DISTINCT unresolved value (via
    Module 45's find_unresolved_unit_details), each showing the raw
    value, occurrence count, the full example row (context clue), a
    genuine dropdown, and a fuzzy-match suggestion where a likely typo
    exists -- identical shape to
    generate_currency_value_resolution_questions().

    A BARE (is_bare=True) value's question carries a "depends_on"
    marker hiding it whenever has_default_unit__<col> is answered
    "Yes" -- see generate_measurement_default_gate_questions above. An
    unrecognised-unit-token value's question has no such marker and
    always shows, since a blanket default can't resolve a typo'd unit.

    Question IDs are positional (f"unit_value_resolution__{col}__{i}"),
    same reasoning as the currency version: Module 46 recomputes the
    same detail list, in the same order, at apply time.
    """
    questions = []

    for col, info in detected_measurements.items():
        details = unit_std.find_unresolved_unit_details(df, col, info)

        measurement_type = info["measurement_type"]
        type_label = MEASUREMENT_TYPE_LABELS[measurement_type]
        target_options = (
            WEIGHT_TARGET_UNIT_OPTIONS if measurement_type == "weight" else LENGTH_TARGET_UNIT_OPTIONS
        )

        for i, detail in enumerate(details):
            raw_value = detail["raw_value"]
            count = detail["count"]
            example_row = detail["example_row"]
            suggested = detail["suggested_unit"]
            is_bare = detail["is_bare"]

            occurrence_note = f" (found in {count} row(s))" if count > 1 else ""

            question = {
                "id": f"unit_value_resolution__{col}__{i}",
                "type": "single_select",
                "render_as_dropdown": True,
                "question_text": (
                    f"In column '{col}', the value {raw_value!r}{occurrence_note} has no {type_label} "
                    f"unit we recognise. What unit should it be treated as?"
                ),
                "options": [
                    {"label": UNIT_DISPLAY_LABELS[u], "value": u} for u in target_options
                ] + [
                    {"label": "I don't know — leave the original value as it is", "value": unit_std.LEAVE_AS_IS},
                    {"label": "I don't know — remove the value (treat as missing)", "value": unit_std.TREAT_AS_MISSING},
                ],
                "default": suggested,
                "affected_columns": [col],
                "raw_value": raw_value,
                "occurrence_count": count,
                "example_row": example_row,
            }
            if is_bare:
                question["depends_on"] = {"question_id": f"has_default_unit__{col}", "hide_if_value": True}
            questions.append(question)

    return questions


# ---------------------------------------------------------------------------
# 6. Currency standardisation questions
# ---------------------------------------------------------------------------
# Mirrors Section 5's shape closely -- same confirm/standardize/target/
# unresolved four-question chain -- with one addition unique to money:
# CURRENCY_RATE_QUESTIONS (numeric_input), since exchange rates can't be
# hardcoded the way Module 45's unit ratios can.

CURRENCY_CONFIRMATION_THRESHOLD = 0.75  # same reasoning as MEASUREMENT_CONFIRMATION_THRESHOLD

CURRENCY_DISPLAY_LABELS = {
    "USD": "US Dollar (USD)", "GBP": "British Pound (GBP)", "EUR": "Euro (EUR)",
    "JPY": "Japanese Yen (JPY)", "INR": "Indian Rupee (INR)", "KRW": "South Korean Won (KRW)",
    "RUB": "Russian Ruble (RUB)", "TRY": "Turkish Lira (TRY)", "CAD": "Canadian Dollar (CAD)",
    "AUD": "Australian Dollar (AUD)", "NZD": "New Zealand Dollar (NZD)", "HKD": "Hong Kong Dollar (HKD)",
    "SGD": "Singapore Dollar (SGD)", "CHF": "Swiss Franc (CHF)", "CNY": "Chinese Yuan (CNY)",
    "ZAR": "South African Rand (ZAR)", "BRL": "Brazilian Real (BRL)", "MXN": "Mexican Peso (MXN)",
    "SEK": "Swedish Krona (SEK)", "NOK": "Norwegian Krone (NOK)", "DKK": "Danish Krone (DKK)",
    "PLN": "Polish Zloty (PLN)", "AED": "UAE Dirham (AED)", "SAR": "Saudi Riyal (SAR)",
    "NGN": "Nigerian Naira (NGN)", "VND": "Vietnamese Dong (VND)",
}

# Full picklist offered for target currency -- same "not limited to
# what's already present" reasoning as Module 11's weight/length target
# unit options: a client may want to standardise into a currency that
# isn't even in the data yet (e.g. converting everything to USD for a
# US-based system, even though the file is all in EUR/GBP).
CURRENCY_TARGET_OPTIONS = sorted(curr_std.ALL_CURRENCY_CODES)


def generate_currency_role_questions(detected_currencies: dict) -> list:
    """
    One yes/no confirmation per column Module 47 flagged, below
    CURRENCY_CONFIRMATION_THRESHOLD -- identical shape and reasoning to
    generate_measurement_role_questions().
    """
    questions = []

    for col, info in detected_currencies.items():
        if info["confidence"] >= CURRENCY_CONFIRMATION_THRESHOLD:
            continue

        questions.append({
            "id": f"confirm_currency__{col}",
            "type": "yes_no",
            "question_text": f"We think '{col}' contains monetary values — is that right?",
            "options": [
                {"label": "Yes, treat it as currency/money", "value": True},
                {"label": "No, treat it as a plain number", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_currency_standardize_questions(detected_currencies: dict) -> list:
    """
    The standardize gate, identical reasoning to
    generate_measurement_standardize_questions(): "No" leaves the
    column completely untouched (e.g. "$100" stays "$100"), "Yes"
    proceeds to target-currency selection and, if needed, exchange
    rates.
    """
    questions = []

    for col, info in detected_currencies.items():
        if not curr_std.needs_standardization(info):
            continue

        questions.append({
            "id": f"standardize_currency__{col}",
            "type": "yes_no",
            "question_text": (
                f"'{col}' has monetary values written in different formats/currencies. "
                f"Do you want to convert all values in this column into a single currency?"
            ),
            "options": [
                {"label": "Yes, standardise this column", "value": True},
                {"label": "No, leave the original values exactly as they are", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_currency_target_questions(detected_currencies: dict) -> list:
    """
    Which currency to standardise into -- only asked when the column
    has genuinely mixed currencies (2+ distinct codes detected), same
    trigger as generate_measurement_target_unit_questions(). Default
    pre-fill is the column's own most common currency.
    """
    questions = []

    for col, info in detected_currencies.items():
        detected = info["detected_currencies"]
        if len(detected) < 2:
            continue

        default_currency = max(detected, key=detected.get)
        currencies_found_str = ", ".join(sorted(detected.keys()))

        questions.append({
            "id": f"target_currency__{col}",
            "type": "single_select",
            "question_text": (
                f"'{col}' contains monetary values in multiple currencies ({currencies_found_str}). "
                f"Which currency would you like all values expressed in?"
            ),
            "options": [
                {"label": CURRENCY_DISPLAY_LABELS.get(c, c), "value": c} for c in CURRENCY_TARGET_OPTIONS
            ],
            "default": default_currency,
            "affected_columns": [col],
            "detected_currencies": detected,
        })

    return questions


def generate_currency_default_gate_questions(df: pd.DataFrame, detected_currencies: dict) -> list:
    """
    ADDED after client feedback on the per-value redesign: asking a
    dropdown for EVERY distinct bare number ("50", "120", ...) is more
    friction than necessary when the client actually has one consistent
    answer -- e.g. a business that always omits its home currency's
    symbol. This asks that directly, ONCE per column, rather than
    forcing case-by-case for something that might have one simple
    answer.

    Deliberately scoped to is_bare=True values ONLY (see Module 47's
    find_unresolved_currency_details docstring) -- values with an
    unrecognised/ambiguous TOKEN (e.g. "60 GPB", a likely typo) are NOT
    covered by a default, since a blanket "no currency mentioned"
    answer doesn't resolve what a garbled mention actually meant; those
    always go through generate_currency_value_resolution_questions
    unconditionally, regardless of this gate's answer.

    Two questions per qualifying column, generated TOGETHER (both
    always exist so Module 48 can pick the right one at apply time --
    same "known edge case" pattern used throughout this file), but
    Module K/app.py hides the second one whenever the first is answered
    "No", and hides the PER-VALUE bare-number questions (from
    generate_currency_value_resolution_questions) whenever it's
    answered "Yes" -- see each question's "depends_on" field, a
    generic mechanism app.py's rendering loop understands (hide this
    question if the referenced question's CURRENT answer equals the
    given value), not something specific to currency.
    """
    questions = []

    for col, info in detected_currencies.items():
        details = curr_std.find_unresolved_currency_details(df, col, info)
        bare_details = [d for d in details if d["is_bare"]]
        if not bare_details:
            continue

        detected = info["detected_currencies"]
        default_currency = max(detected, key=detected.get) if detected else None
        gate_id = f"has_default_currency__{col}"

        questions.append({
            "id": gate_id,
            "type": "yes_no",
            "question_text": (
                f"Is there a DEFAULT currency that should be used whenever no currency is "
                f"mentioned at all in '{col}' (e.g. a bare value like '{bare_details[0]['raw_value']}')?"
            ),
            "info_text": (
                "This only covers values with NO currency written at all. Values with an "
                "unclear or misspelled currency (e.g. '60 GPB') are always handled one-by-one "
                "below, regardless of your answer here."
            ),
            "options": [
                {"label": "Yes, there's a consistent default", "value": True},
                {"label": "No, handle these case-by-case instead", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

        questions.append({
            "id": f"default_currency__{col}",
            "type": "single_select",
            "question_text": f"Which currency should be the default for '{col}' when none is mentioned?",
            "options": [
                {"label": CURRENCY_DISPLAY_LABELS.get(c, c), "value": c} for c in CURRENCY_TARGET_OPTIONS
            ],
            "default": default_currency,
            "affected_columns": [col],
            "depends_on": {"question_id": gate_id, "hide_if_value": False},
        })

    return questions


def generate_currency_value_resolution_questions(df: pd.DataFrame, detected_currencies: dict) -> list:
    """
    REDESIGNED after real-file testing (client-reported issue): the
    previous version asked ONE shared question per column -- "what
    currency should ALL unrecognised values be treated as" -- which is
    simply wrong whenever a column has more than one DIFFERENT
    unresolved value, since each one plausibly needs its own answer
    (a typo'd code isn't the same problem as a genuinely bare number).

    Now asks ONE question PER DISTINCT unresolved value (via Module
    47's find_unresolved_currency_details), each showing:
      - the raw value itself and how many rows contain it
      - the FULL EXAMPLE ROW it came from, since other columns in that
        row can be the actual clue (e.g. a Country column saying "UK"
        strongly suggests GBP) -- "the clue can be in the cell",
        exactly as requested
      - a genuine DROPDOWN (render_as_dropdown=True; app.py renders
        this with st.selectbox rather than the st.radio used for every
        other single_select question), since a page of one radio-list
        per distinct value would be unusably long otherwise
      - a pre-filled SUGGESTION from Module 47's fuzzy matching (e.g.
        "GPB" -> GBP) where a likely typo exists -- still just a
        default, never auto-applied; the client sees and can override
        it like any other default in this project

    Question IDs are positional (f"currency_value_resolution__
    {col}__{i}", i = enumeration index within that column's distinct-
    value list) rather than built from the raw value itself, since a
    raw value can contain characters unsafe for a dict key / Streamlit
    widget key. This relies on Module 48 recomputing Module 47's
    detail list in the SAME order at apply time, against the SAME
    (unchanged) column -- true here since nothing between question
    generation and this conversion step touches these columns.

    A BARE (is_bare=True) value's question carries a "depends_on"
    marker hiding it whenever has_default_currency__<col> is answered
    "Yes" -- see generate_currency_default_gate_questions. An
    AMBIGUOUS-TOKEN value's question has no such marker and always
    shows, since a blanket default can't resolve a typo'd currency.

    SCALE NOTE, worth being upfront about: this asks one question per
    DISTINCT value, which is fine for a handful of oddities but would
    become unwieldy on a real file with e.g. 200 different unrecognised
    spellings in one column. Not built here -- if a real client file
    ever hits that scale, grouping similar values together first would
    be the natural next refinement.
    """
    questions = []

    for col, info in detected_currencies.items():
        details = curr_std.find_unresolved_currency_details(df, col, info)

        for i, detail in enumerate(details):
            raw_value = detail["raw_value"]
            count = detail["count"]
            example_row = detail["example_row"]
            suggested = detail["suggested_currency"]
            is_bare = detail["is_bare"]

            occurrence_note = f" (found in {count} row(s))" if count > 1 else ""

            question = {
                "id": f"currency_value_resolution__{col}__{i}",
                "type": "single_select",
                "render_as_dropdown": True,
                "question_text": (
                    f"In column '{col}', the value {raw_value!r}{occurrence_note} has no currency "
                    f"we recognise. What currency should it be treated as?"
                ),
                "options": [
                    {"label": CURRENCY_DISPLAY_LABELS.get(c, c), "value": c} for c in CURRENCY_TARGET_OPTIONS
                ] + [
                    {"label": "I don't know — leave the original value as it is", "value": curr_std.LEAVE_AS_IS},
                    {"label": "I don't know — remove the value (treat as missing)", "value": curr_std.TREAT_AS_MISSING},
                ],
                "default": suggested,
                "affected_columns": [col],
                "raw_value": raw_value,
                "occurrence_count": count,
                "example_row": example_row,
            }
            if is_bare:
                question["depends_on"] = {"question_id": f"has_default_currency__{col}", "hide_if_value": True}
            questions.append(question)

    return questions


def generate_currency_rate_questions(detected_currencies: dict) -> list:
    """
    THE new question type for this feature: exchange rates can't be
    hardcoded (unlike Module 45's unit ratios), so the client has to
    supply one for every non-default currency actually found in a
    mixed-currency column.

    ONE question per (column, non-default currency) pair -- e.g. a
    column with USD/EUR/GBP and a default target of USD gets two rate
    questions: "1 EUR = ? USD" and "1 GBP = ? USD". Only generated for
    columns with genuinely mixed currencies (same trigger as
    generate_currency_target_questions -- a single-currency column
    needs no rate at all, just unit-stripping).

    A REAL BUG, FIXED after client feedback, worth explaining since the
    fix lives partly outside this function: the question text below is
    phrased against the column's DEFAULT target currency (its most
    common detected currency) purely as a FALLBACK for contexts where
    no live answer exists yet (e.g. this file's own __main__ block, or
    any other caller that doesn't go through app.py's rendering loop).
    In the real app, app.py's question-rendering loop rewrites this
    text live, every rerun, using whatever the client has ACTUALLY
    selected in target_currency__<col> by that point (that question
    always renders first, so its answer is already available) -- so a
    client who picks GBP instead of the assumed default sees "convert
    into GBP" immediately, never a stale "convert into USD" left over
    from generation time. Module 48's own former "rate mismatch ->
    treat as unresolved" fallback logic was removed as part of this
    same fix, since it's no longer possible for a collected rate to be
    aimed at the wrong target when the question itself always tracks
    the live one.

    No numeric default is offered -- unlike a unit ratio, there's no
    safe placeholder value for an exchange rate; forcing an explicit
    answer here follows the same "don't guess" principle as the
    dayfirst question having no default either.
    """
    questions = []

    for col, info in detected_currencies.items():
        detected = info["detected_currencies"]
        if len(detected) < 2:
            continue

        default_target = max(detected, key=detected.get)
        target_label = CURRENCY_DISPLAY_LABELS.get(default_target, default_target)

        # BUG FIXED alongside app.py's dynamic-rewrite fix: this used
        # to skip generating a rate question for whichever currency
        # happened to be the DEFAULT target at generation time (no
        # rate needed to convert a currency into itself). But if the
        # client later picks a DIFFERENT target, that original default
        # currency now DOES need a rate -- and since it was never
        # generated as a question at all, there was nothing for
        # app.py's dynamic rewrite to show. Now generates a question
        # for EVERY detected currency; app.py's rendering loop is what
        # decides, live, which ONE to skip (whichever currently
        # matches the client's ACTUAL chosen target) -- so the skip
        # always tracks the current target, not a frozen assumption.
        for source_currency in sorted(detected.keys()):
            source_label = CURRENCY_DISPLAY_LABELS.get(source_currency, source_currency)

            questions.append({
                "id": f"exchange_rate__{col}__{source_currency}",
                "type": "numeric_input",
                "question_text": (
                    f"What is the exchange rate to convert 1 {source_currency} into "
                    f"{default_target}? (i.e. 1 {source_currency} = ? {default_target}) -- "
                    f"used for '{col}' only if '{target_label}' remains your chosen target "
                    f"currency above; if you pick a different target, this value won't apply "
                    f"and those rows will be flagged as unresolved instead of converted."
                ),
                "default": None,
                "min_value": 0.0,
                "affected_columns": [col],
                "source_currency": source_currency,
                "assumed_target_currency": default_target,
            })

    return questions


# ---------------------------------------------------------------------------
# 7. Duration/Age standardisation questions
# ---------------------------------------------------------------------------
# Sibling to Sections 5/6, built WITH the default-gate + per-value
# design from the start (see Module 50's docstring) rather than
# retrofitted the way weight/length and currency both were. Deliberately
# NOT about datetime timestamps -- Module 50 only ever fires on values
# that parse as a NUMBER (optionally with a duration unit like "days"
# or "years"), never on date-formatted text.

DURATION_UNIT_DISPLAY_LABELS = {
    "s": "Seconds", "min": "Minutes", "hr": "Hours", "day": "Days",
    "week": "Weeks", "month": "Months", "year": "Years",
}
DURATION_TARGET_UNIT_OPTIONS = ["s", "min", "hr", "day", "week", "month", "year"]


def generate_duration_role_questions(detected_durations: dict) -> list:
    """Mirrors generate_measurement_role_questions() exactly."""
    questions = []

    for col, info in detected_durations.items():
        if info["confidence"] >= MEASUREMENT_CONFIRMATION_THRESHOLD:
            continue

        questions.append({
            "id": f"confirm_duration__{col}",
            "type": "yes_no",
            "question_text": f"We think '{col}' contains a duration (an amount of time, e.g. '5 days', '3 years') — is that right?",
            "options": [
                {"label": "Yes, treat it as a duration", "value": True},
                {"label": "No, treat it as a plain number", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_duration_standardize_questions(detected_durations: dict) -> list:
    """Mirrors generate_measurement_standardize_questions() exactly."""
    questions = []

    for col, info in detected_durations.items():
        if not duration_std.needs_standardization(info):
            continue

        questions.append({
            "id": f"standardize_duration__{col}",
            "type": "yes_no",
            "question_text": (
                f"'{col}' has duration values written in different formats/units. "
                f"Do you want to convert all values in this column into a single unit?"
            ),
            "options": [
                {"label": "Yes, standardise this column", "value": True},
                {"label": "No, leave the original values exactly as they are", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_duration_target_unit_questions(detected_durations: dict) -> list:
    """Mirrors generate_measurement_target_unit_questions() exactly."""
    questions = []

    for col, info in detected_durations.items():
        detected = info["detected_units"]
        if len(detected) < 2:
            continue

        default_unit = max(detected, key=detected.get)
        units_found_str = ", ".join(sorted(detected.keys()))

        questions.append({
            "id": f"target_duration_unit__{col}",
            "type": "single_select",
            "question_text": (
                f"'{col}' contains duration values in multiple units ({units_found_str}). "
                f"Which unit would you like all values expressed in?"
            ),
            "options": [
                {"label": DURATION_UNIT_DISPLAY_LABELS[u], "value": u} for u in DURATION_TARGET_UNIT_OPTIONS
            ],
            "default": default_unit,
            "affected_columns": [col],
            "detected_units": detected,
        })

    return questions


def generate_duration_default_gate_questions(df: pd.DataFrame, detected_durations: dict) -> list:
    """Mirrors generate_measurement_default_gate_questions() exactly."""
    questions = []

    for col, info in detected_durations.items():
        details = duration_std.find_unresolved_duration_details(df, col, info)
        bare_details = [d for d in details if d["is_bare"]]
        if not bare_details:
            continue

        detected = info["detected_units"]
        default_unit = max(detected, key=detected.get) if detected else None
        gate_id = f"has_default_duration_unit__{col}"

        questions.append({
            "id": gate_id,
            "type": "yes_no",
            "question_text": (
                f"Is there a DEFAULT unit that should be used whenever no unit is mentioned at "
                f"all in '{col}' (e.g. a bare value like '{bare_details[0]['raw_value']}')?"
            ),
            "info_text": (
                "This only covers values with NO unit written at all. Values with an unclear or "
                "misspelled unit (e.g. '5 yeras') are always handled one-by-one below, regardless "
                "of your answer here."
            ),
            "options": [
                {"label": "Yes, there's a consistent default", "value": True},
                {"label": "No, handle these case-by-case instead", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

        questions.append({
            "id": f"default_duration_unit__{col}",
            "type": "single_select",
            "question_text": f"Which unit should be the default for '{col}' when none is mentioned?",
            "options": [
                {"label": DURATION_UNIT_DISPLAY_LABELS[u], "value": u} for u in DURATION_TARGET_UNIT_OPTIONS
            ],
            "default": default_unit,
            "affected_columns": [col],
            "depends_on": {"question_id": gate_id, "hide_if_value": False},
        })

    return questions


def generate_duration_value_resolution_questions(df: pd.DataFrame, detected_durations: dict) -> list:
    """
    Per-value dropdown for genuinely BARE duration values (a number
    with no unit text at all, e.g. "42") -- mirrors generate_
    measurement_unresolved_unit_questions() exactly. Every entry here
    is is_bare=True now (see Module 50's find_unresolved_duration_
    details, redesigned to only return bare values) -- values with an
    unrecognised unit TOKEN are handled separately, at the token
    level, by generate_duration_token_resolution_questions below.
    """
    questions = []

    for col, info in detected_durations.items():
        details = duration_std.find_unresolved_duration_details(df, col, info)

        for i, detail in enumerate(details):
            raw_value = detail["raw_value"]
            count = detail["count"]
            example_row = detail["example_row"]
            suggested = detail["suggested_unit"]
            is_bare = detail["is_bare"]

            occurrence_note = f" (found in {count} row(s))" if count > 1 else ""

            question = {
                "id": f"duration_value_resolution__{col}__{i}",
                "type": "single_select",
                "render_as_dropdown": True,
                "question_text": (
                    f"In column '{col}', the value {raw_value!r}{occurrence_note} has no duration "
                    f"unit we recognise. What unit should it be treated as?"
                ),
                "options": [
                    {"label": DURATION_UNIT_DISPLAY_LABELS[u], "value": u} for u in DURATION_TARGET_UNIT_OPTIONS
                ] + [
                    {"label": "I don't know — leave the original value as it is", "value": duration_std.LEAVE_AS_IS},
                    {"label": "I don't know — remove the value (treat as missing)", "value": duration_std.TREAT_AS_MISSING},
                ],
                "default": suggested,
                "affected_columns": [col],
                "raw_value": raw_value,
                "occurrence_count": count,
                "example_row": example_row,
            }
            if is_bare:
                question["depends_on"] = {"question_id": f"has_default_duration_unit__{col}", "hide_if_value": True}
            questions.append(question)

    return questions


def generate_duration_token_resolution_questions(df: pd.DataFrame, detected_durations: dict) -> list:
    """
    THE client's requested redesign: one dropdown question PER
    DISTINCT UNRECOGNISED UNIT TOKEN (e.g. "m"), not per whole value.
    A column can have many different compound values that all share
    the same ambiguous abbreviation ("5 m 6 d", "10 m 3 yr", "2 m") --
    asking once "what does 'm' mean" resolves every one of them,
    rather than asking about each full value separately.

    Shows a few example VALUES the token appears in (for context --
    seeing "5 m 6 d" alongside "m" makes it obvious this is likely
    "months", for instance), and a fuzzy-match suggestion where a
    likely typo exists (e.g. "yeras" -> year), same principle as every
    other per-value/per-token question in this file: a suggestion is
    only ever a pre-filled default, never applied silently.

    Not scoped by is_bare/depends_on the way the bare-value questions
    are -- an unrecognised TOKEN is a fundamentally different kind of
    ambiguity from a missing unit, and always needs its own answer
    regardless of whether a default-unit gate exists for this column.
    """
    questions = []

    for col, info in detected_durations.items():
        tokens = duration_std.find_unresolved_duration_tokens(df, col, info)

        for i, detail in enumerate(tokens):
            token = detail["token"]
            count = detail["count"]
            examples = detail["example_values"]
            suggested = detail["suggested_unit"]

            examples_str = ", ".join(repr(e) for e in examples)
            occurrence_note = f" (appears in {count} value(s), e.g. {examples_str})"

            questions.append({
                "id": f"duration_token_resolution__{col}__{i}",
                "type": "single_select",
                "render_as_dropdown": True,
                "question_text": (
                    f"In column '{col}', the unit '{token}'{occurrence_note} isn't one we recognise. "
                    f"What does '{token}' mean here?"
                ),
                "options": [
                    {"label": DURATION_UNIT_DISPLAY_LABELS[u], "value": u} for u in DURATION_TARGET_UNIT_OPTIONS
                ] + [
                    {"label": "I don't know — leave the original value as it is", "value": duration_std.LEAVE_AS_IS},
                    {"label": "I don't know — remove the value (treat as missing)", "value": duration_std.TREAT_AS_MISSING},
                ],
                "default": suggested,
                "affected_columns": [col],
                "token": token,
                "occurrence_count": count,
                "example_values": examples,
            })

    return questions


# ---------------------------------------------------------------------------
# 8. Telephone & email standardisation questions
# ---------------------------------------------------------------------------
# Sibling to Sections 5-7, but with real structural differences from
# every measurement/currency/duration question above:
#   - Phone validity genuinely depends on EXTERNAL CONTEXT (which
#     country) that Modules 45/47/50 never needed -- hence the
#     default-REGION question below, asked once per column, not once
#     per possible country (see Module 52's own docstring).
#   - There's no "target unit to convert numbers into" here -- instead
#     a TARGET FORMAT (how the same information should be displayed).
#   - Per the client's own spec, unresolved values are NOT handled
#     with individual dropdown questions the way currency/duration
#     values are -- they're deliberately collected into ONE batch
#     review table instead (spec Section 5: "do not interrupt the
#     user with one question per bad value"). That table is a
#     genuinely different UI element (an editable grid, not a single
#     radio/dropdown) and is a separate build, not included in this
#     question-generation pass.

PHONE_CONFIRMATION_THRESHOLD = 0.75  # same reasoning as every other *_CONFIRMATION_THRESHOLD in this file
EMAIL_CONFIRMATION_THRESHOLD = 0.75

PHONE_FORMAT_DISPLAY_LABELS = {
    "E164": "E.164 -- fully machine-normalised (e.g. +442079460958)",
    "INTERNATIONAL": "International, natural spacing (e.g. +44 20 7946 0958)",
    "NATIONAL": "National/local convention, no country code (e.g. 020 7946 0958)",
}
PHONE_FORMAT_OPTIONS = ["INTERNATIONAL", "E164", "NATIONAL"]


def generate_phone_role_questions(detected_phones: dict) -> list:
    """One yes/no confirmation per flagged column, mirroring generate_measurement_role_questions()."""
    questions = []

    for col, info in detected_phones.items():
        if info["confidence"] >= PHONE_CONFIRMATION_THRESHOLD:
            continue

        questions.append({
            "id": f"confirm_phone__{col}",
            "type": "yes_no",
            "question_text": f"We think '{col}' contains telephone numbers — is that right?",
            "options": [
                {"label": "Yes, treat it as phone numbers", "value": True},
                {"label": "No, treat it as plain text", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_phone_standardize_questions(detected_phones: dict) -> list:
    """
    The standardize gate -- "Do you want to standardise telephone-
    number formatting?", straight from the client's own spec wording.
    """
    questions = []

    for col in detected_phones:
        questions.append({
            "id": f"standardize_phone__{col}",
            "type": "yes_no",
            "question_text": f"Do you want to standardise telephone-number formatting in '{col}'?",
            "options": [
                {"label": "Yes", "value": True},
                {"label": "No, leave the original values exactly as they are", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_phone_format_questions(detected_phones: dict) -> list:
    """
    Which format to standardise into. CUSTOM templates (the client's
    spec explicitly allows a free-form template) are NOT offered yet
    -- deliberately deferred, see Module 52's own scope note -- only
    the three library-backed formats are offered at this stage.
    """
    questions = []

    for col in detected_phones:
        questions.append({
            "id": f"target_phone_format__{col}",
            "type": "single_select",
            "question_text": f"Which format would you like phone numbers in '{col}' expressed in?",
            "options": [
                {"label": PHONE_FORMAT_DISPLAY_LABELS[f], "value": f} for f in PHONE_FORMAT_OPTIONS
            ],
            "default": "INTERNATIONAL",
            "affected_columns": [col],
            "depends_on": {"question_id": f"standardize_phone__{col}", "hide_if_value": False},
        })

    return questions


def generate_phone_default_region_questions(detected_phones: dict) -> list:
    """
    The ONE piece of country information ever asked for -- a single
    default region for numbers with NO country code at all (see
    Module 52's docstring for why this fully replaces "list every
    possible country/area code"). Pre-filled with whatever region was
    already detected among the column's OWN internationally-formatted
    values, if any -- purely a starting suggestion, always overridable.
    """
    questions = []

    for col, info in detected_phones.items():
        options = [
            {"label": name, "value": code} for code, name in sorted(
                phone_std.REGION_DISPLAY_NAMES.items(), key=lambda kv: kv[1]
            )
        ]
        questions.append({
            "id": f"default_phone_region__{col}",
            "type": "single_select",
            "render_as_dropdown": True,
            "question_text": (
                f"What is the default country for phone numbers in '{col}' that don't include "
                f"a country code (e.g. a bare local number written without a leading '+')?"
            ),
            "info_text": (
                "Only used for numbers with NO country code at all. Any number that already "
                "includes one (e.g. '+44...') is identified automatically and ignores this setting."
            ),
            "options": options,
            "default": info.get("guessed_region"),
            "affected_columns": [col],
            "depends_on": {"question_id": f"standardize_phone__{col}", "hide_if_value": False},
        })

    return questions


EMAIL_CASING_OPTIONS_LABELS = {
    "domain_only": "Lowercase the domain only (e.g. Name@GMAIL.com -> Name@gmail.com) -- always safe",
    "full_address": "Lowercase the whole address (e.g. Name@GMAIL.com -> name@gmail.com) -- matches real-world provider behaviour",
}


def generate_email_role_questions(detected_emails: dict) -> list:
    """Mirrors generate_phone_role_questions() exactly."""
    questions = []

    for col, info in detected_emails.items():
        if info["confidence"] >= EMAIL_CONFIRMATION_THRESHOLD:
            continue

        questions.append({
            "id": f"confirm_email__{col}",
            "type": "yes_no",
            "question_text": f"We think '{col}' contains email addresses — is that right?",
            "options": [
                {"label": "Yes, treat it as email addresses", "value": True},
                {"label": "No, treat it as plain text", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_email_standardize_questions(detected_emails: dict) -> list:
    """The standardize gate, straight from the client's spec wording."""
    questions = []

    for col in detected_emails:
        questions.append({
            "id": f"standardize_email__{col}",
            "type": "yes_no",
            "question_text": f"Do you want to standardise and validate email addresses in '{col}'?",
            "options": [
                {"label": "Yes", "value": True},
                {"label": "No, leave the original values exactly as they are", "value": False},
            ],
            "default": True,
            "affected_columns": [col],
        })

    return questions


def generate_email_casing_questions(detected_emails: dict) -> list:
    """
    The one real choice with a genuine technical nuance (see Module
    53's docstring): lowercase only the domain (always technically
    correct) vs. the whole address (matches real-world provider
    behaviour, but not strictly spec-guaranteed) -- an explicit choice
    rather than a silent default either way.
    """
    questions = []

    for col in detected_emails:
        questions.append({
            "id": f"email_casing__{col}",
            "type": "single_select",
            "question_text": f"How should casing be standardised for '{col}'?",
            "options": [
                {"label": EMAIL_CASING_OPTIONS_LABELS[k], "value": k} for k in ["domain_only", "full_address"]
            ],
            "default": "domain_only",
            "affected_columns": [col],
            "depends_on": {"question_id": f"standardize_email__{col}", "hide_if_value": False},
        })

    return questions


# ---------------------------------------------------------------------------
# 9. Basic Package orchestration
# ---------------------------------------------------------------------------
# A DELIBERATELY SMALL question set for the Basic Package tier --
# reuses only the two generators actually relevant to what the Basic
# Package promises (missing-value treatment, duplicate detection).
# Every other section above (role confirmation, dayfirst, text-number,
# weight/length, currency, duration, phone/email) exists to support
# STANDARD-Package-only features (unit/currency/duration standardis-
# ation, phone/email validation, column splitting) -- asking about
# them here would both confuse a Basic Package client with irrelevant
# questions AND misleadingly imply those features are included.

def generate_basic_mode_questions(profile: dict, df=None) -> list:
    """
    The full question set for a Basic Package run: disguised-missing-
    value confirmation (part of "missing value treatment" -- deciding
    what counts as missing has to happen before it can be treated) and
    duplicate-detection column choice. Nothing else.
    """
    questions = []
    questions.extend(generate_disguised_missing_questions(profile))

    duplicate_q = generate_duplicate_subset_question(profile)
    if duplicate_q:
        questions.append(duplicate_q)

    return questions


# ---------------------------------------------------------------------------
# 10. Standard Package orchestration
# ---------------------------------------------------------------------------

def generate_questions(profile: dict, df=None) -> list:
    """
    Run every question-generation check and return only the questions
    that actually apply to this file, in a sensible display order:
    role confirmation FIRST (everything else -- the dayfirst question,
    the duplicate-column guess -- implicitly assumes the roles it's
    reasoning about are correct, so this is the most foundational
    check), then dayfirst, then disguised-missing confirmations (one
    per column), then the duplicate-detection column choice last (a
    business-rule choice, not a data-quality flag).

    KNOWN EDGE CASE, worth being upfront about: if a user rejects a
    column's guessed role here (e.g. says a column ISN'T really a
    date), any OTHER question already generated based on that guess
    (e.g. a dayfirst question naming that same column) won't
    retroactively update in this same batch -- all questions for a
    file are generated together in one pass, not as a back-and-forth
    conversation. This only matters on the (rare) combination of a
    low-confidence structural guess that's ALSO genuinely ambiguous in
    another way, and Module K's config builder still respects a role
    rejection correctly when the pipeline actually runs -- so the
    worst case is a slightly redundant question shown, never a wrong
    action taken.


    `df`: the raw uploaded DataFrame, needed ONLY for the weight/length
    measurement questions (Module 45's detection works on actual cell
    values, not the profile dict). Optional and defaults to None so
    any existing caller that only ever passed `profile` keeps working
    unchanged; measurement questions are simply skipped when df isn't
    supplied. app.py's call site needs one line updated (passing df
    through) to actually get these questions in the live app.
    """
    questions = []

    questions.extend(generate_role_confirmation_questions(profile))

    detected_text_numbers = text_num.detect_text_number_columns(df) if df is not None else {}
    questions.extend(generate_text_number_confirmation_questions(detected_text_numbers))

    # Measurement/currency detection runs against a PREVIEW of df with
    # text-numbers already converted (see _build_text_number_preview),
    # not the raw df -- otherwise a column that's mostly word-form
    # values would show these two detectors a much weaker signal than
    # what will actually exist once Module 49 runs for real.
    detection_preview_df = _build_text_number_preview(df, detected_text_numbers) if df is not None else None

    detected_measurements = unit_std.detect_measurement_columns(detection_preview_df) if detection_preview_df is not None else {}
    # All FOUR measurement question types kept contiguous, in the order
    # the client actually wants to answer them: is this a measurement ->
    # do you want it standardised -> which unit -> what to do about
    # values we can't place. Moved together during the standardize-gate
    # addition, rather than left interleaved with dayfirst/disguised-
    # missing as before, since they're all one coherent decision chain
    # about the same columns.
    questions.extend(generate_measurement_role_questions(detected_measurements))
    questions.extend(generate_measurement_standardize_questions(detected_measurements))
    questions.extend(generate_measurement_target_unit_questions(detected_measurements))
    questions.extend(generate_measurement_default_gate_questions(detection_preview_df, detected_measurements))
    questions.extend(generate_measurement_unresolved_unit_questions(detection_preview_df, detected_measurements))

    detected_currencies = curr_std.detect_currency_columns(detection_preview_df) if detection_preview_df is not None else {}
    # Same contiguous grouping as the measurement chain above, plus the
    # rate questions at the end of the chain (they depend on a target
    # currency existing conceptually, even if not yet answered -- see
    # generate_currency_rate_questions()'s own docstring).
    questions.extend(generate_currency_role_questions(detected_currencies))
    questions.extend(generate_currency_standardize_questions(detected_currencies))
    questions.extend(generate_currency_target_questions(detected_currencies))
    questions.extend(generate_currency_default_gate_questions(detection_preview_df, detected_currencies))
    questions.extend(generate_currency_value_resolution_questions(detection_preview_df, detected_currencies))
    questions.extend(generate_currency_rate_questions(detected_currencies))

    detected_durations = duration_std.detect_duration_columns(detection_preview_df) if detection_preview_df is not None else {}
    # Same contiguous grouping and reasoning as the measurement/currency
    # chains above -- built with the default-gate design from the
    # start rather than retrofitted (see Module 50's docstring).
    questions.extend(generate_duration_role_questions(detected_durations))
    questions.extend(generate_duration_standardize_questions(detected_durations))
    questions.extend(generate_duration_target_unit_questions(detected_durations))
    questions.extend(generate_duration_default_gate_questions(detection_preview_df, detected_durations))
    questions.extend(generate_duration_value_resolution_questions(detection_preview_df, detected_durations))
    questions.extend(generate_duration_token_resolution_questions(detection_preview_df, detected_durations))

    # Phone/email detection does NOT use the text-number preview --
    # word-form numbers have no bearing on whether something looks
    # like a phone number or email address, unlike the measurement/
    # currency/duration detectors.
    detected_phones = phone_std.detect_phone_columns(df) if df is not None else {}
    questions.extend(generate_phone_role_questions(detected_phones))
    questions.extend(generate_phone_standardize_questions(detected_phones))
    questions.extend(generate_phone_format_questions(detected_phones))
    questions.extend(generate_phone_default_region_questions(detected_phones))

    detected_emails = email_std.detect_email_columns(df) if df is not None else {}
    questions.extend(generate_email_role_questions(detected_emails))
    questions.extend(generate_email_standardize_questions(detected_emails))
    questions.extend(generate_email_casing_questions(detected_emails))

    dayfirst_q = generate_dayfirst_question(profile)
    if dayfirst_q:
        questions.append(dayfirst_q)

    questions.extend(generate_disguised_missing_questions(profile))

    dup_q = generate_duplicate_subset_question(profile)
    if dup_q:
        questions.append(dup_q)

    return questions


def print_questions(questions: list) -> None:
    """Human-readable dump, for testing this module before the Streamlit UI exists."""
    if not questions:
        print("No clarifying questions needed for this file.")
        return

    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['type']}] {q['question_text']}")
        if q["type"] == "numeric_input":
            # No options list for a free numeric entry -- just show
            # whatever bounds/default exist, if any.
            default_str = q["default"] if q["default"] is not None else "(no default -- must be entered)"
            print(f"     enter a number, default: {default_str}")
            continue
        for opt in q["options"]:
            marker = " (default)" if q["default"] == opt["value"] or (
                isinstance(q["default"], list) and opt["value"] in q["default"]
            ) else ""
            print(f"     - {opt['label']}{marker}")


if __name__ == "__main__":
    ingestion = import_module("01_ingestion")
    profiling = import_module("10_column_profiling")

    # Colab-friendly hardcoded path, same convention as every other module.
    filepath = "dirty_customer_data.xlsx"

    df = ingestion.load_data(filepath)
    profile = profiling.profile_dataframe(df)
    questions = generate_questions(profile, df)
    print_questions(questions)
