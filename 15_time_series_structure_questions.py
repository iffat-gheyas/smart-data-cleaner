"""
Module O: Time Series Structure Identification (Questions)
------------------------------------------------------------
Section 1 of the Time-Series Smart Cleaner spec: before ANY cleaning
can happen, the pipeline needs to know which column is time, which
column(s) are the values, how the time stamp is structured, and
whether it represents real calendar dates or relative periods.

DESIGN NOTE -- why this module looks different from Module K's
(11_question_generation.py) batch-style generate_all_questions():
those cross-sectional questions are mostly INDEPENDENT of each other
(disguised-missing markers for column A don't depend on the answer
for column B). Time-structure questions are NOT independent -- you
can't sensibly ask "what does the first component mean?" until you
know the separator, and you can't detect the separator until you know
WHICH column is the time column. So instead of one big batch
function, this module exposes small, composable question-generator
functions that the caller (app.py) invokes IN SEQUENCE, feeding each
answer into the next question -- Streamlit's own script-rerun model
handles this naturally, the same way "prepare_for_ml" already
determines what renders further down the page.

Every function here only ever proposes a DEFAULT to pre-select in the
UI -- per the spec's explicit rule, meaning is never inferred and
silently applied without the client confirming it. This mirrors the
existing disguised-missing-marker pattern (Module K): pre-selected,
always overridable, never silent.
"""

import re

import pandas as pd

COMPONENT_MEANING_OPTIONS = [
    "Year (real calendar year)",
    "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
    "Quarter",
    "Month",
    "Week",
    "Day",
    "Hour",
    "Minute",
    "Second",
    "Relative period index (e.g. period 1, 2, 3 ...)",
    "Identifier (not part of the time ordering)",
    "Component to ignore",
    "Custom / other",
]

SEPARATOR_OPTIONS = ["-", "/", ".", "_", " ", ":", "Other / no separator (single value)"]


def detect_time_column_candidates(df: pd.DataFrame) -> list:
    """
    Scores every column on how likely it is to BE the time column --
    used only to pick a sensible DEFAULT for the "which column is
    time?" question, never to decide it outright. Three independent
    signals, any of which is enough to nominate a column:
      - the column NAME hints at time ("date", "month", "period"...)
      - most sampled values contain a separator character (a strong
        signal for structured stamps like "1-01", unlike a plain
        numeric ID column)
      - most sampled values parse as a real date on their own

    Returns columns ordered by score, ties broken by original column
    order (via Python's stable sort) so results are deterministic.
    """
    scored = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue
        # A column pandas already parsed as numeric (int/float) can
        # never genuinely be a structured time stamp like "1-01" --
        # pandas would have been unable to read that as a number in
        # the first place. Without this check, a float column's own
        # decimal points (e.g. "266.0") falsely match the separator
        # regex below and wrongly nominate it as a time candidate.
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        name_hint = any(
            k in col.lower() for k in ["date", "time", "period", "month", "year", "week", "day", "quarter"]
        )
        has_separator_pattern = sample.str.contains(r"[-/._: ]", regex=True).mean() > 0.5
        looks_like_real_date = pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.5
        score = int(name_hint) + int(has_separator_pattern) + int(looks_like_real_date)
        if score > 0:
            scored.append((col, score))
    scored.sort(key=lambda pair: -pair[1])
    return [col for col, _ in scored]


def detect_component_separator(series: pd.Series) -> str:
    """
    Finds which single-character separator splits EVERY sampled value
    into the SAME number of parts -- e.g. "1-01" / "2-06" both split
    into exactly 2 parts on "-". Returns None if no separator is
    consistent across the sample (e.g. genuinely single-valued
    stamps, or inconsistent formatting the client needs to resolve
    manually via the question UI rather than a guess).
    """
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return None
    for sep in ["-", "/", ".", "_", " ", ":"]:
        counts = sample.str.count(re.escape(sep))
        if (counts > 0).all() and counts.nunique() == 1:
            return sep
    return None


def preview_time_values(series: pd.Series, n_samples: int = 5) -> list:
    """A few real sample values to show the client, e.g. ['1-01', '1-02', '1-03']."""
    return series.dropna().astype(str).head(n_samples).tolist()


def generate_time_column_question(df: pd.DataFrame, candidate_time_cols: list) -> dict:
    return {
        "id": "ts_time_column",
        "type": "single_select",
        "text": "Which column contains the time information for this series?",
        "options": list(df.columns),
        "default": candidate_time_cols[0] if candidate_time_cols else df.columns[0],
    }


def generate_time_columns_question(df: pd.DataFrame, candidate_time_cols: list) -> dict:
    """
    Multi-select version -- lets a client mark SEVERAL columns as
    jointly forming the time stamp (e.g. a separate Date column and
    Time column, or fully pre-split Year/Month/Day columns). Defaults
    to just the single best-scoring candidate, since most files still
    have exactly one time column -- multi-column setups are the less
    common case, opted into rather than assumed.
    """
    return {
        "id": "ts_time_columns",
        "type": "multi_select",
        "text": (
            "Which column(s) together contain the time information for this series? "
            "(select more than one if, e.g., Date and Time are in separate columns)"
        ),
        "options": list(df.columns),
        "default": candidate_time_cols[:1] if candidate_time_cols else [],
    }


def generate_value_columns_question(df: pd.DataFrame, time_columns) -> dict:
    """
    Multi-select since a dataset can have several time-series
    variables sharing one time column (this becomes relevant once
    Multivariate mode is built) -- for a univariate file with exactly
    one remaining column, that column is pre-selected as the default
    since there's nothing else it could sensibly be.

    time_columns: a single column name (str) or a list -- every
    configured time column is excluded from the candidate value
    columns, not just one.
    """
    if isinstance(time_columns, str):
        time_columns = [time_columns]
    remaining = [c for c in df.columns if c not in time_columns]
    return {
        "id": "ts_value_columns",
        "type": "multi_select",
        "text": "Which column(s) contain the time-series value(s) you want cleaned?",
        "options": remaining,
        "default": remaining if len(remaining) == 1 else [],
    }


def generate_calendar_type_question(time_column: str, sample_values: list) -> dict:
    """
    THE critical question the spec calls out by name: never assume
    '1-01' means January of some real year. Default leans toward
    "relative" only when the values don't already look like genuine
    dates (preview_time_values + this default together still require
    explicit confirmation either way -- this just picks a sensible
    starting radio position).
    """
    return {
        "id": "ts_calendar_type",
        "type": "single_select",
        "text": (
            f"'{time_column}' contains values like {sample_values[:3]}. Are these real "
            f"calendar dates, or relative time periods (e.g. 'Year 1' isn't an actual "
            f"calendar year — just the first year covered by this series)?"
        ),
        "options": ["Real calendar dates", "Relative time periods (not real calendar years)"],
        "default": "Relative time periods (not real calendar years)",
    }


def generate_separator_question(time_column: str, sample_values: list, detected_separator: str) -> dict:
    default = detected_separator if detected_separator else "Other / no separator (single value)"
    return {
        "id": "ts_separator",
        "type": "single_select",
        "text": (
            f"'{time_column}' values like {sample_values[:3]} appear to use "
            f"'{detected_separator}' as a separator. Confirm, or choose the correct one:"
            if detected_separator
            else f"'{time_column}' values like {sample_values[:3]} don't show a clear, "
            f"consistent separator. What separates the components (if anything)?"
        ),
        "options": SEPARATOR_OPTIONS,
        "default": default,
    }


def generate_single_element_meaning_question(time_column: str, sample_values: list) -> dict:
    """
    For a time column with NO separator -- a single value per cell
    (e.g. just '2020', or just '9' meaning September). Same idea as
    generate_component_meaning_questions() above, but there's only one
    thing to ask about since there's only one component at all.

    "Component to ignore" is deliberately EXCLUDED from the options
    here (present in COMPONENT_MEANING_OPTIONS generally) -- ignoring
    the only value in the column would leave nothing to build a time
    index from at all, so offering it as a choice here would be
    actively nonsensical rather than just unlikely.

    Deliberately no default, same reasoning as the per-component
    version: meaning must be an active choice, never inferred silently.
    """
    options = [opt for opt in COMPONENT_MEANING_OPTIONS if opt != "Component to ignore"]
    example = sample_values[0] if sample_values else "?"
    return {
        "id": "ts_single_element_meaning",
        "type": "single_select",
        "text": f"'{time_column}' has one value per cell (e.g. '{example}'). What does it represent?",
        "options": options,
        "default": None,
    }


def generate_component_meaning_questions(
    time_column: str, sample_values: list, separator: str, n_components: int
) -> list:
    """
    One question PER component position -- e.g. for '1-01', a
    question for "the first part" and a separate one for "the second
    part". Deliberately NO default here (unlike every other question
    in this module): the spec is explicit that component meaning must
    never be inferred from name or formatting alone when several
    interpretations are genuinely plausible, so the UI should force
    an active choice rather than pre-select one.
    """
    questions = []
    parts_per_sample = [v.split(separator) for v in sample_values if len(v.split(separator)) == n_components]
    ordinal_words = {0: "first", 1: "second", 2: "third", 3: "fourth"}
    for i in range(n_components):
        example = parts_per_sample[0][i] if parts_per_sample else "?"
        position_word = ordinal_words.get(i, f"{i + 1}th")
        questions.append(
            {
                "id": f"ts_component_{i + 1}_meaning",
                "type": "single_select",
                "text": (
                    f"In '{time_column}', what does the {position_word} component "
                    f"(e.g. '{example}' in '{sample_values[0] if sample_values else ''}') represent?"
                ),
                "options": COMPONENT_MEANING_OPTIONS,
                "default": None,
            }
        )
    return questions


if __name__ == "__main__":
    # Colab-friendly manual test against the actual shampoo.csv, walking
    # through the sequence exactly as app.py will: ask time column,
    # THEN (using that answer) detect separator/components, THEN ask
    # calendar type + separator + per-component meaning.
    filepath = "shampoo.csv"
    df = pd.read_csv(filepath)

    print("=" * 60)
    print("STEP 1: detect time-column candidates")
    print("=" * 60)
    candidates = detect_time_column_candidates(df)
    print("Candidates (best first):", candidates)

    q1 = generate_time_column_question(df, candidates)
    print("\nQ1:", q1["text"])
    print("   options:", q1["options"], " default:", q1["default"])

    # Simulate the client accepting the default answer
    time_column = q1["default"]

    print("\n" + "=" * 60)
    print("STEP 2: value-column question (depends on Q1's answer)")
    print("=" * 60)
    q2 = generate_value_columns_question(df, time_column)
    print("Q2:", q2["text"])
    print("   options:", q2["options"], " default:", q2["default"])

    print("\n" + "=" * 60)
    print("STEP 3: separator + calendar-type + component questions")
    print("=" * 60)
    sample_values = preview_time_values(df[time_column])
    print("Sample values:", sample_values)

    detected_sep = detect_component_separator(df[time_column])
    print("Detected separator:", repr(detected_sep))

    q3 = generate_separator_question(time_column, sample_values, detected_sep)
    print("\nQ3:", q3["text"])
    print("   options:", q3["options"], " default:", q3["default"])

    q4 = generate_calendar_type_question(time_column, sample_values)
    print("\nQ4:", q4["text"])
    print("   options:", q4["options"], " default:", q4["default"])

    separator = q3["default"]
    n_components = max(len(v.split(separator)) for v in sample_values)
    print(f"\nn_components (using separator {separator!r}):", n_components)

    component_questions = generate_component_meaning_questions(
        time_column, sample_values, separator, n_components
    )
    for q in component_questions:
        print("\nQ:", q["text"])
        print("   options:", q["options"])
        print("   default:", q["default"], " (deliberately None -- must be actively chosen)")
