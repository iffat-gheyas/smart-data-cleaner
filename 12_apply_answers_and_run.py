"""
Module K: Apply User Answers & Run Pipeline (Phase 2 — new)
----------------------------------------------------------------
This is the orchestration layer the Phase 2 spec calls "the
interactive equivalent of main.py" -- it's what turns (a) Module I's
column profile, (b) Module J's generated questions, and (c) the
user's actual answers to those questions into real calls into the
Sprint 1 modules, replacing main.py's hardcoded CONFIGURATION section.

Built in three pieces:
  1. apply_disguised_missing_answers() -- a genuinely NEW transformation
     step. None of the Sprint 1 modules handle this: Module I only
     DETECTS disguised-missing markers (e.g. "?", "999"), it never
     touches the data. Once the user confirms "yes, treat as missing"
     for a column, something has to actually turn those markers into
     real NaN before the rest of the pipeline runs -- that's this
     function's job.
  2. build_pipeline_config() -- reads the profile and turns each
     column's semantic role into a concrete pipeline parameter (which
     columns get title-cased, which are date columns and whether they
     carry a time component, which are candidates for KNN numeric vs
     categorical imputation, etc.) -- the dynamic replacement for
     main.py's hardcoded CASING_RULES / NUMERIC_COLUMNS / etc.
  3. run_interactive_pipeline() -- calls the existing Sprint 1
     functions in the SAME validated order as main.py's run_pipeline()
     (text standardisation -> duplicate removal -> date/time ->
     outlier detection+capping -> missing-value imputation -> drop
     empty columns -> column splitting -> validation -> report), but
     every parameter now comes from build_pipeline_config() instead of
     a hardcoded constant.

Every Sprint 1 module is reused completely unchanged -- exactly as the
Phase 2 spec requires.

ADDENDUM (generic-mode measurement feature): a fourth piece,
apply_measurement_unit_conversion() (imported from Module 46), now
runs inside _run_pre_imputation_steps(), converting any confirmed AND
client-approved weight/length column to a single target unit BEFORE
numeric-text coercion -- see that function's own inline comment for
why the ordering matters. A client can decline standardisation per
column (standardize_units__<col> = False), in which case the column is
left completely untouched. For columns that ARE standardised, Module
46 no longer adds a `<col>_original_unit` companion column -- instead,
apply_final_column_renames() (also Module 46) runs as the LAST step of
both run_interactive_pipeline() and run_interactive_pipeline_multiple_
datasets(), renaming e.g. "Weight" to "Weight (kg)" once every other
step has finished using the original column name.

ADDENDUM (generic-mode currency feature): apply_currency_conversion()
(Module 48) runs immediately alongside apply_measurement_unit_
conversion(), same position and same reasoning. Unlike unit ratios,
exchange rates come from the client (Module 11's numeric_input
questions), never a hardcoded table. Reuses Module 46's
apply_final_column_renames() directly for the end-of-pipeline rename
(e.g. "Price" -> "Price (USD)") rather than keeping a separate copy.

ADDENDUM (generic-mode text-number feature): apply_text_number_
conversion() (Module 49) runs FIRST among these three, immediately
before apply_measurement_unit_conversion() -- "fifty kg" has to become
"50 kg" before Module 46's own parser (which requires a leading digit)
can recognise it at all. Only rewrites the number part of a value,
never a unit or currency itself, so it needs no awareness of what
Modules 46/48 do with its output.

ADDENDUM (generic-mode duration/age feature): apply_duration_
conversion() (Module 51) runs alongside measurement/currency, same
position and reasoning -- "5 days" needs its unit text stripped before
coerce_numeric_text_columns can do anything with it. Unlike weight/
length and currency (which both got the default-gate + per-value
redesign as a LATER addition), this feature was built with that design
from the start. A bug caught while testing this feature -- a column
that's entirely bare numbers with zero embedded unit tokens (e.g. an
"Age" column of just "42", "35", ...), flagged purely via a strong
name-keyword match, had no way to resolve a target unit even when the
client had explicitly confirmed a default -- turned out to affect
Modules 46 and 48 too (both fixed alongside this one): a client-
confirmed default now also serves as the column's target unit/currency
when nothing else can supply one.

ADDENDUM (telephone & email standardisation): apply_phone_
standardization() (Module 54) and apply_email_standardization()
(Module 55) run right after duration conversion. Structurally
different from every feature above: these columns stay TEXT the whole
way through (no profile numeric-patching, no end-of-pipeline rename),
and unresolved values are NEVER resolved with a per-value dropdown
question -- per the client's own spec, they're collected into a batch
review TABLE instead (app.py renders this, using Modules 54/55's
find_unresolved_phone_values() / find_unresolved_email_values(),
BEFORE this pipeline function ever runs), and the client's decisions
arrive here as f"phone_review__<col>__<idx>" / f"email_review__<col>__
<idx>" entries in `answers`. "Don't know" is treated as explicit
permission to set the value missing, exactly as specified.
"""

from importlib import import_module

import numpy as np
import pandas as pd

profiling = import_module("10_column_profiling")
text_std = import_module("02_text_standardisation")
dedup = import_module("03_duplicate_removal")
datetime_std = import_module("04_datetime_standardisation")
splitting = import_module("05_column_splitting")
missing_val = import_module("06_missing_value_single_imputation")
multiple_imputation = import_module("13_multiple_imputation")
ml_ready = import_module("14_ml_ready_encoding")
outliers = import_module("07_outlier_detection")
validation = import_module("08_validation")
ingestion = import_module("01_ingestion")
report = import_module("09_cleaning_report")
measurement_conv = import_module("46_unit_standardisation_conversion")
currency_conv = import_module("48_currency_conversion")
text_num = import_module("49_text_number_parsing")
duration_conv = import_module("51_duration_conversion")
phone_conv = import_module("54_phone_conversion")
phone_std = import_module("52_phone_standardisation")
qgen = import_module("11_question_generation")
email_conv = import_module("55_email_conversion")
email_std = import_module("53_email_standardisation")
value_corrections_mod = import_module("57_value_corrections")


# ---------------------------------------------------------------------------
# 1. Apply disguised-missing answers
# ---------------------------------------------------------------------------

def apply_user_specified_missing_values(df: pd.DataFrame, extra_missing_values: list) -> tuple:
    """
    Applies a GLOBAL, user-supplied list of extra "treat as missing"
    values across every column, regardless of whether Module I's
    automatic detection ever flagged anything for that specific
    column. Added after real-user testing highlighted a real
    structural limitation: if a placeholder convention (e.g. "99" as a
    numeric placeholder) isn't already on Module I's candidate list,
    no question gets generated for that column AT ALL -- so a
    per-column supplementary field would still miss it, since it would
    only ever appear on columns that already triggered a question.
    This global field works even when nothing was auto-detected.

    Deliberately simple: matches values case-insensitively for text
    columns, numerically for numeric columns, across ALL columns
    uniformly. Trades per-column precision (can't say "99 means
    missing in Quantity but not in Age") for a much simpler UI -- one
    field instead of one per column. A reasonable trade-off since this
    is meant as a targeted escape hatch for a specific known
    convention, not a replacement for the smarter per-column
    detection.
    """
    if not extra_missing_values:
        return df, []

    df = df.copy()
    log_entries = []

    text_markers = {str(v).strip().lower() for v in extra_missing_values}
    numeric_markers = set()
    for v in extra_missing_values:
        try:
            numeric_markers.add(float(str(v).strip()))
        except ValueError:
            continue

    for col in df.columns:
        if df[col].dtype == "bool":
            continue  # booleans can't meaningfully match a text/number marker

        if pd.api.types.is_numeric_dtype(df[col]):
            if not numeric_markers:
                continue
            mask = df[col].isin(numeric_markers)
        else:
            if not text_markers:
                continue
            normalised = df[col].astype(str).str.strip().str.lower()
            mask = normalised.isin(text_markers) & df[col].notna()

        n_replaced = int(mask.sum())
        if n_replaced > 0:
            df.loc[mask, col] = np.nan
            log_entries.append(f"Treated {n_replaced} user-specified missing value(s) in '{col}' as missing")

    return df, log_entries


def apply_column_specific_missing_values(df: pd.DataFrame, column_specific_missing_values: dict) -> tuple:
    """
    Like apply_user_specified_missing_values, but SCOPED to individual
    columns rather than applied globally -- e.g. "low" should be
    treated as missing in "Rating" but is a completely legitimate,
    real value in "Priority Level". The global version can't express
    that distinction (it's one setting for the whole file); this one
    can, at the cost of needing to be set per column instead of once.

    `column_specific_missing_values`: {column: [marker, marker, ...]}.
    Matches case-insensitively for text columns, numerically for
    numeric columns -- same matching approach as the global version,
    just scoped to one column's own list instead of one shared list
    applied everywhere.
    """
    if not column_specific_missing_values:
        return df, []

    df = df.copy()
    log_entries = []

    for col, markers in column_specific_missing_values.items():
        if col not in df.columns or not markers:
            continue

        if df[col].dtype == "bool":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_markers = set()
            for v in markers:
                try:
                    numeric_markers.add(float(str(v).strip()))
                except ValueError:
                    continue
            if not numeric_markers:
                continue
            mask = df[col].isin(numeric_markers)
        else:
            text_markers = {str(v).strip().lower() for v in markers}
            normalised = df[col].astype(str).str.strip().str.lower()
            mask = normalised.isin(text_markers) & df[col].notna()

        n_replaced = int(mask.sum())
        if n_replaced > 0:
            df.loc[mask, col] = np.nan
            log_entries.append(f"Treated {n_replaced} value(s) in '{col}' as missing (column-specific: {', '.join(str(m) for m in markers)})")

    return df, log_entries


def apply_disguised_missing_answers(df: pd.DataFrame, profile: dict, answers: dict) -> tuple:
    """
    For every column with a disguised_missing__<column> answer,
    convert ONLY the specific marker values the user actually confirmed
    (a list, from the multi-select question -- e.g. the user might
    confirm "?" and "unknown" but leave "999" unchecked because it's a
    real value in their data) into real NaN.

    Handles text and numeric columns differently (matching how Module
    I detected them in the first place): text markers are matched
    case-insensitively after stripping whitespace; numeric markers
    (e.g. 999) are matched as actual numbers.

    Returns (updated_df, log_entries) -- log_entries is a list of
    short strings describing what happened, ready to fold into the
    final cleaning report's actions log.
    """
    df = df.copy()
    log_entries = []

    for col, info in profile.items():
        answer_key = f"disguised_missing__{col}"
        confirmed_markers = answers.get(answer_key)
        if not confirmed_markers:
            continue

        available_markers = info.get("disguised_missing_markers") or {}
        # Only act on markers that are BOTH confirmed by the user AND
        # genuinely detected for this column -- guards against a
        # stale/tampered answer referencing a marker that isn't
        # actually present.
        confirmed_markers = [m for m in confirmed_markers if m in available_markers]
        if not confirmed_markers or col not in df.columns:
            continue

        if info["dtype"] == "numeric":
            numeric_markers = set()
            for marker in confirmed_markers:
                try:
                    numeric_markers.add(float(marker))
                except ValueError:
                    continue
            mask = df[col].isin(numeric_markers)
        else:
            normalised = df[col].astype(str).str.strip().str.lower()
            mask = normalised.isin(set(confirmed_markers)) & df[col].notna()

        n_replaced = int(mask.sum())
        if n_replaced > 0:
            df.loc[mask, col] = np.nan
            marker_list = ", ".join(f"'{m}'" for m in confirmed_markers)
            log_entries.append(
                f"Treated {n_replaced} value(s) in '{col}' as missing ({marker_list})"
            )

    return df, log_entries


# ---------------------------------------------------------------------------
# 1b. Coerce "numbers stored as text" columns to real numeric dtype
# ---------------------------------------------------------------------------

def coerce_numeric_text_columns(df: pd.DataFrame, profile: dict, answers: dict) -> tuple:
    """
    For every column Module I flagged as numeric_text (numbers Excel
    stored as text -- confirmed as a real, common problem by testing
    against an actual downloaded e-commerce dataset), convert it to a
    genuine numeric dtype using pd.to_numeric(errors="coerce").

    Any value that DOESN'T parse as a number (e.g. "abd", "four
    hundred") becomes a real NaN -- which is exactly correct: these
    are genuine data-entry errors, and turning them into proper
    missing values means they flow into the SAME outlier-detection and
    KNN-imputation steps as any other numeric column, rather than
    being silently left as uncleaned text or masked by a generic
    free-text placeholder fill.

    Respects a role rejection from a confirm_role__<col> answer, same
    as build_pipeline_config -- if the user said "no, this isn't
    really numeric", we don't coerce it.
    """
    df = df.copy()
    log_entries = []

    for col, info in profile.items():
        if info["semantic_role"] != "numeric_text":
            continue
        if answers.get(f"confirm_role__{col}") is False:
            continue
        if col not in df.columns:
            continue

        was_present = df[col].notna()
        coerced = pd.to_numeric(df[col], errors="coerce")
        newly_missing = int((coerced.isna() & was_present).sum())

        df[col] = coerced
        if newly_missing > 0:
            log_entries.append(
                f"Converted '{col}' from text to numeric; {newly_missing} non-numeric "
                f"value(s) treated as missing"
            )
        else:
            log_entries.append(f"Converted '{col}' from text to numeric")

    return df, log_entries


# ---------------------------------------------------------------------------
# 2. Build pipeline config from profile + answers
# ---------------------------------------------------------------------------

# Common legal-entity suffixes, grouped by equivalence class (Ltd ==
# Limited, PLC == Public Limited Company, etc). Used ONLY to decide
# whether two values are probably the SAME company written two ways
# (e.g. "Tesco PLC" / "TESCO", "IBM Ltd" / "IBM Limited") -- not a
# company-name dictionary. This is a deliberately narrower, safer
# approach than a company-name lookup table: two DIFFERENT companies
# would need to share the exact same core name (everything except a
# recognised legal suffix) to be merged, which is a much smaller false-
# merge risk than trying to maintain a master list of known company
# name variants (which has no universal reference the way country
# names do -- a related but legally distinct entity, e.g. a
# subsidiary, could easily be mismerged by a broader dictionary
# approach).
LEGAL_SUFFIXES = {
    "plc", "public limited company",
    "ltd", "limited",
    "inc", "incorporated",
    "corp", "corporation",
    "llc", "limited liability company",
}


def _canonicalization_group_key(value: str) -> str:
    """
    Grouping key for categorical casing canonicalization: the value's
    core text with any recognised trailing legal suffix stripped, so
    "Tesco PLC" and "TESCO" -- and "IBM Ltd" and "IBM Limited" -- land
    in the same group despite neither being a pure case-variant of the
    other. A generic word like "Holdings" is deliberately NOT in
    LEGAL_SUFFIXES (it's often part of a company's actual
    distinguishing name, not a generic entity-type suffix), so
    "HSBC Holdings" correctly stays its own group rather than being
    stripped down to just "HSBC".
    """
    # str() wrap is deliberate, not redundant: under pandas' newer
    # "str" extension dtype, a missing value survives .astype(str) as
    # an actual float NaN rather than being stringified to "nan" the
    # way legacy object-dtype columns behave -- confirmed as a real
    # crash by testing against the demo dataset (a bare .strip() call
    # on that raw NaN raised AttributeError). str(value) safely
    # handles both cases.
    lowered = str(value).strip().lower().rstrip(".")
    tokens = lowered.split()
    if tokens and tokens[-1] in LEGAL_SUFFIXES:
        return " ".join(tokens[:-1])
    return lowered


def _canonicalize_categorical_casing(series: pd.Series) -> pd.Series:
    """
    Normalise a low-cardinality categorical column's casing by
    grouping values that are the SAME once case is ignored (e.g.
    "Tesco PLC", "tesco plc", "TESCO PLC") and mapping every row in a
    group to that group's single MOST COMMON original casing.

    This deliberately does NOT use blanket str.title(). Testing
    surfaced a real problem with that approach: it mangles acronyms
    ("PLC" -> "Plc", "USA" -> "Usa", "HSBC" -> "Hsbc") -- but simply
    trying to detect and preserve "likely acronyms" by length/case
    turned out to be worse: it also caught ordinary short surnames
    entered in all-caps (e.g. "YOUNG", "SCOTT"), which are exactly as
    short and all-uppercase as a real acronym, with no reliable way to
    tell them apart from the text alone.

    Choosing the MOST COMMON real casing per group sidesteps that
    entirely -- it doesn't need to know what an acronym IS. If "PLC"
    genuinely appears more often than "Plc" or "plc" in the raw data,
    it wins on its own merits; if a column has no dominant form, one
    consistent variant still gets picked, which is at minimum no
    worse than before, and always internally consistent.

    Not used for full_name -- names are usually unique per row (no
    meaningful "group" to find a mode within), and plain title-casing
    genuinely is correct there (titles/names virtually never contain
    real acronyms).
    """
    non_null_mask = series.notna()
    trimmed = series.astype(str).str.strip()
    # Grouping key: case-insensitive AND legal-suffix-aware (see
    # _canonicalization_group_key) -- catches both pure case variants
    # ("tesco plc" / "TESCO PLC") and suffix variants of the same
    # entity ("Tesco PLC" / "TESCO", "IBM Ltd" / "IBM Limited").
    group_key = trimmed.apply(_canonicalization_group_key)

    # Most common exact original value within each group.
    mode_by_group = (
        trimmed[non_null_mask]
        .groupby(group_key[non_null_mask])
        .agg(lambda group: group.value_counts().idxmax())
    )

    canonical = group_key.map(mode_by_group)
    canonical[~non_null_mask] = np.nan
    return canonical


def _standardize_state_column(series: pd.Series) -> pd.Series:
    """
    Maps each value through the state-name canonical dictionary
    (profiling.STATE_CANONICAL_NAMES) to its correctly-cased official
    form -- e.g. "california" / "CALIFORNIA" / "California" all ->
    "California". Values that AREN'T a recognised US state (e.g. a
    genuinely unusual manual entry) fall back to the same suffix/
    case-aware canonicalization every other categorical column gets,
    so nothing outside the known state list is silently dropped --
    it's just normalised for consistent casing like any other
    category.

    Unlike _standardize_country_column, there's no merge/collapse
    business decision here. Module I's profiler now only assigns the
    "state" role to a column that's genuinely ASKING for a state (see
    _content_score_state's docstring for how it's told apart from a
    real country column), so preserving the actual state name is
    always the correct behaviour -- no per-column advanced-setting
    toggle needed for this one.
    """
    non_null_mask = series.notna()
    trimmed = series.astype(str).str.strip()
    normalized_key = trimmed.apply(profiling._normalize_country_text)

    resolved = normalized_key.map(profiling.STATE_CANONICAL_NAMES)
    unresolved_mask = resolved.isna() & non_null_mask
    if unresolved_mask.any():
        resolved.loc[unresolved_mask] = _canonicalize_categorical_casing(series[unresolved_mask])

    resolved[~non_null_mask] = np.nan
    return resolved


def _standardize_country_column(
    series: pd.Series,
    merge_uk_constituents: bool = False,
    merge_us_states: bool = True,
) -> pd.Series:
    """
    Maps each value through Module I's country reference dictionary to
    its canonical form (e.g. "U.K." / "UK" / "United Kingdom" all ->
    "United Kingdom"). Values the dictionary doesn't recognise fall
    back to the same suffix/case-aware canonicalization every other
    categorical column gets, so an unusual-but-valid country name
    still ends up internally consistent even without a dictionary
    entry for it.

    merge_uk_constituents controls the one genuine judgment call here:
    whether "England"/"Scotland"/"Wales"/"Northern Ireland" should be
    folded into "United Kingdom". Default False -- they're the UK's
    real constituent countries, not incorrect data, so some clients
    may want that regional detail preserved. Exposed as an advanced
    setting rather than decided silently.
    """
    non_null_mask = series.notna()
    trimmed = series.astype(str).str.strip()
    normalized_key = trimmed.apply(profiling._normalize_country_text)

    def resolve(key, original):
        if key in profiling.COUNTRY_VARIANT_LOOKUP:
            return profiling.COUNTRY_VARIANT_LOOKUP[key]
        if key in profiling.UK_CONSTITUENT_COUNTRIES:
            return "United Kingdom" if merge_uk_constituents else original.strip().title()
        if key in profiling.US_STATES_TO_COUNTRY:
            # Unlike the UK case, defaults to merging: a US state
            # appearing in a column literally called "country" is
            # essentially always a data-entry mistake, not a
            # deliberate regional distinction someone wants kept.
            return profiling.US_STATES_TO_COUNTRY[key] if merge_us_states else original.strip().title()
        return None  # not recognised -- resolved via fallback canonicalization below

    resolved = pd.Series(
        [resolve(k, o) for k, o in zip(normalized_key, trimmed)],
        index=series.index,
    )

    unresolved_mask = resolved.isna() & non_null_mask
    if unresolved_mask.any():
        resolved.loc[unresolved_mask] = _canonicalize_categorical_casing(series[unresolved_mask])

    resolved[~non_null_mask] = np.nan
    return resolved


def _column_has_time_component(series: pd.Series) -> bool:
    """
    A date column should be treated as a full datetime (parsed AND
    split with hour/minute/second) if ANY of its values actually carry
    a time component -- matching how Sprint 1's transaction_timestamp
    column was handled (some rows date-only, some rows date+time, per
    the spec's Section E.5). A single ':' character reliably signals a
    time component here, since none of our numeric/spelled-out DATE
    formats use a colon.
    """
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    return non_null.str.contains(":").any()


def _infer_global_dayfirst_default(profile: dict) -> bool:
    """
    Fallback dayfirst value to use when NO dayfirst question was asked
    (or answered) -- either because the file had no ambiguous dates at
    all, or because every date column's ambiguity was already
    auto-resolved from its own unambiguous rows (see Module I's
    detect_date_ambiguity). Previously this fallback was just a
    hardcoded True, which was silently WRONG whenever a file's real
    convention happened to be month-first and every date column had
    successfully self-resolved without ever asking -- confirmed as a
    real gap once auto-resolution was added, since a correct inference
    was being computed and then thrown away in favour of a constant.

    Only trusts the inference when every date column that resolved on
    its own AGREES on the same value -- if two different date columns
    in the same file genuinely imply different conventions, that's a
    real conflict, not something to silently pick a side on, so this
    falls back to True (matching the previous default) exactly as
    before in that case.
    """
    inferred_values = {
        info["date_ambiguity"]["inferred_dayfirst"]
        for info in profile.values()
        if info.get("date_ambiguity") and info["date_ambiguity"].get("inferred_dayfirst") is not None
    }
    if len(inferred_values) == 1:
        return inferred_values.pop()
    return True


def build_pipeline_config(df: pd.DataFrame, profile: dict, answers: dict) -> dict:
    """
    Translate the column profile into the concrete parameters the
    pipeline needs -- the dynamic, per-file replacement for main.py's
    hardcoded CONFIGURATION section (DAYFIRST, CASING_RULES,
    NUMERIC_COLUMNS, etc.).

    Role -> parameter mapping (the actual design decisions):
      full_name        -> title-cased, split into first/last name
      categorical       -> title-cased; candidate for KNN categorical
                           imputation if it has any missing values
      email             -> lower-cased (so duplicate detection and
                           validation aren't fooled by case differences),
                           validated for format
      phone             -> validated for format
      address           -> split into street/city/postcode
      date_or_datetime  -> standardised; time-component detected per
                           column rather than assumed
      numeric_measure   -> outlier detection (IQR) + KNN numeric
                           imputation, and supplies the numeric
                           features used by categorical KNN imputation
      free_text         -> filled with a constant placeholder if missing
      id_key            -> left alone by default (not cased, not
                           imputed, not outlier-checked) -- an ID isn't
                           a "measure" and shouldn't be averaged or
                           capped; it's only used elsewhere if the user
                           explicitly picked it for duplicate detection
      empty             -> not given a role at all; the existing
                           drop_columns_with_missing() step (Sprint 1's
                           M1) removes it same as main.py always did
    """
    _raw_duplicate_subset = answers.get("duplicate_subset_columns", [])
    config = {
        "dayfirst": answers.get("dayfirst", _infer_global_dayfirst_default(profile)),
        "casing_rules": {},
        # ADDED per explicit client request: leaving this question
        # completely unanswered (nothing selected at all) now means
        # "skip duplicate detection entirely" -- NOT the same as
        # selecting "Exact match — all columns", which still runs
        # detection, just without an additional subset-based pass. The
        # RAW (pre-sentinel-stripping) answer is what decides this, so
        # it must be checked before ALL_COLUMNS_DUPLICATE_MATCH gets
        # stripped out below.
        "duplicate_detection_enabled": bool(_raw_duplicate_subset),
        "duplicate_subset_columns": (
            [] if qgen.ALL_COLUMNS_DUPLICATE_MATCH in _raw_duplicate_subset else _raw_duplicate_subset
        ),
        "date_columns": {},          # {col: {"include_time": bool}}
        "numeric_columns": [],
        "categorical_knn_columns": [],
        "free_text_columns": [],
        "email_columns": [],
        "phone_columns": [],
        "phone_default_regions": {},  # {col: default_phone_region__<col> answer, if any} -- see _run_post_imputation_steps' validation fix
        "full_name_split_columns": [],
        "address_split_columns": [],
        "id_columns": [],
    }

    for col, info in profile.items():
        role = info["semantic_role"]

        # If the user rejected this column's guessed role via a
        # confirm_role__<col> question (Module J), it gets NO
        # role-based treatment at all -- not cased, not split, not
        # imputed by role, not validated. It's demoted to an ordinary
        # pass-through column (still gets whitespace-trimmed, same as
        # everything else, but nothing role-specific). We deliberately
        # don't try to guess a REPLACEMENT role -- that would just be
        # swapping one guess for another instead of respecting the
        # user's correction.
        if answers.get(f"confirm_role__{col}") is False:
            continue

        # A numeric-CODED binary/categorical column (e.g. "Sex" stored
        # as 1/2, "Steroid" stored as 1/2 for yes/no) -- confirmed as a
        # real bug by a real client dataset: pandas correctly types
        # these as numeric, and the semantic-role detector correctly
        # guesses "numeric_measure" (there's no reliable content signal
        # telling a genuine continuous measure apart from a numeric
        # code), but treating this as a continuous measure for
        # IMPUTATION is wrong regardless of whether the user ever wants
        # ML-ready output -- KNN/mean/median on a binary 1/2 column can
        # produce a nonsense value like 1.4. variable_type (auto-
        # suggested by Module I, correctable by the user in the app's
        # editable profile table) is what actually catches this, so it
        # takes priority over role here -- no casing rule applied
        # (these are numeric codes, not text needing case
        # normalisation), just categorical-style imputation, same as
        # any other categorical column.
        variable_type = info.get("variable_type")
        if variable_type in ("binary", "categorical") and info["dtype"] == "numeric" and role in ("numeric_measure", "numeric_text"):
            if df[col].isna().sum() > 0:  # fresh check -- info["missing_count"] is stale once user-specified missing markers have run
                config["categorical_knn_columns"].append(col)
            continue

        if role == "full_name":
            config["casing_rules"][col] = "title"
            config["full_name_split_columns"].append(col)

        elif role == "categorical":
            config["casing_rules"][col] = "canonicalize"
            if df[col].isna().sum() > 0:  # fresh check -- info["missing_count"] is stale once user-specified missing markers have run
                config["categorical_knn_columns"].append(col)

        elif role == "country":
            config["casing_rules"][col] = "country"
            if df[col].isna().sum() > 0:  # fresh check -- info["missing_count"] is stale once user-specified missing markers have run
                config["categorical_knn_columns"].append(col)

        elif role == "state":
            # A genuine "state" column (distinct from "country" -- see
            # Module I's _content_score_state) gets its own casing
            # rule: fix casing to the state's correctly-cased official
            # name, but never merge/collapse it into anything -- unlike
            # a country column, there's no "was this a data-entry
            # mistake" judgment call to make here. The whole point of
            # this role existing is that the user was ASKING for the
            # state, so the state is always the correct thing to keep.
            config["casing_rules"][col] = "state"
            if df[col].isna().sum() > 0:  # fresh check -- info["missing_count"] is stale once user-specified missing markers have run
                config["categorical_knn_columns"].append(col)

        elif role == "email":
            config["casing_rules"][col] = "lower"
            config["email_columns"].append(col)

        elif role == "phone":
            config["phone_columns"].append(col)
            config["phone_default_regions"][col] = answers.get(f"default_phone_region__{col}")

        elif role == "address":
            config["address_split_columns"].append(col)

        elif role == "date_or_datetime":
            config["date_columns"][col] = {
                "include_time": _column_has_time_component(df[col])
            }

        elif role == "numeric_measure":
            config["numeric_columns"].append(col)

        elif role == "numeric_text":
            # By the time this config is used, coerce_numeric_text_columns()
            # has already converted this column to a real numeric dtype
            # (see run_interactive_pipeline) -- it now flows through the
            # standard numeric pipeline (outlier detection + KNN
            # imputation) exactly like any naturally-numeric column.
            config["numeric_columns"].append(col)

        elif role == "free_text":
            config["free_text_columns"].append(col)

        elif role == "id_key":
            config["id_columns"].append(col)

        # "empty" columns and boolean-categorical columns intentionally
        # get no special treatment here -- empty ones are removed later
        # by the existing drop-mostly-empty-columns step.

    return config


# ---------------------------------------------------------------------------
# 3. Orchestration -- the interactive equivalent of main.py
# ---------------------------------------------------------------------------


# Columns Module 13/14 add for TRANSPARENCY (uncertainty disclosure,
# diagnostic flags) -- never genuine analytical features, so they're
# excluded from the Isolation Forest's input. Including them would
# mean "how confident the imputation was" or "was this row already
# flagged as a univariate outlier" quietly influencing which ROWS get
# flagged as multivariate outliers, which conflates two different
# concerns (data quality bookkeeping vs. the actual analytical signal).
_DIAGNOSTIC_COLUMN_SUFFIXES = (
    "_outlier_iqr", "_was_imputed", "_imputation_sd", "_imputation_lower_95",
    "_imputation_upper_95", "_top_probability", "_categorical_uncertainty",
    "_2nd_probability", "_3rd_probability",
)


def _select_multivariate_outlier_features(df: pd.DataFrame) -> list:
    """
    Picks the columns Isolation Forest should actually look at: every
    numeric, non-boolean, non-diagnostic column currently in `df`. In
    Machine-Learning-Ready mode this naturally ends up being almost
    every column (continuous stays numeric, binary is 0/1, categorical
    is one-hot 0/1, ordinal/integer are whole numbers) -- exactly the
    feature set a downstream model would actually see.
    """
    return [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and df[col].dtype != "bool"
        and not any(col.endswith(suf) for suf in _DIAGNOSTIC_COLUMN_SUFFIXES)
    ]


def _detect_multivariate_outliers(df: pd.DataFrame, actions_log: list, contamination=0.05) -> pd.DataFrame:
    """
    Runs Module G's Isolation Forest across every genuine analytical
    feature column (see _select_multivariate_outlier_features) --
    DETECTION ONLY, exactly like the existing IQR outlier step: adds
    a "multivariate_outlier" boolean flag column, never removes or
    modifies a row. What to actually DO with a flagged row (drop it,
    downweight it, investigate it) is a decision for whatever
    problem-specific app consumes this output next, not this cleaner.

    `contamination` deliberately defaults to an explicit 0.05 rather
    than Module G's own "auto" default -- confirmed as a genuine,
    reproducible problem by testing against a fully-categorical
    dataset (Isolation Forest's automatic threshold estimation badly
    degenerates once every feature is one-hot-encoded into a sparse
    binary space, flagging 99%+ of rows as "outliers" instead of a
    sensible minority). An explicit contamination value sidesteps
    that failure mode entirely and behaves sensibly regardless of how
    sparse/categorical the encoded feature space ends up being.

    Only runs in Machine-Learning-Ready mode, and only once every
    candidate feature column is already fully complete -- Isolation
    Forest can't run on data with missing values, and by this point
    (after imputation and ML-ready encoding) that should already be
    true for every real feature; if it isn't, this skips rather than
    crashing, with an honest note in the log about why.
    """
    feature_cols = _select_multivariate_outlier_features(df)
    if not feature_cols:
        return df

    if df[feature_cols].isna().sum().sum() > 0:
        actions_log.append(
            "Skipped multivariate outlier detection (Isolation Forest) -- some feature "
            "columns still have missing values"
        )
        return df

    df = outliers.detect_outliers_isolation_forest(df, columns=feature_cols, contamination=contamination)
    n_flagged = int(df["multivariate_outlier"].sum())
    actions_log.append(
        f"Detected {n_flagged} multivariate outlier row(s) via Isolation Forest across "
        f"{len(feature_cols)} feature column(s) -- flagged only, not modified or removed"
    )
    return df


def _build_settings(advanced_settings: dict = None) -> dict:
    """
    Applies every advanced-setting default in ONE place, shared by
    both run_interactive_pipeline() (consolidated output) and
    run_interactive_pipeline_multiple_datasets() (M complete datasets)
    -- previously this dict only existed inline inside the single
    consolidated function; factored out so a new entry point can't
    accidentally drift out of sync with these defaults over time.
    """
    settings = {
        "knn_k": 5,
        "drop_column_missing_threshold": 90,
        "merge_uk_constituents_by_column": {},
        "merge_us_states_by_column": {},
        "extra_missing_values": [],
        "column_specific_missing_values": {},
        "cap_outliers": False,
        "outlier_iqr_multiplier": 1.5,
        "numeric_imputation_default": "knn",
        "numeric_imputation_overrides": {},
        "categorical_imputation_default": "leave",
        "categorical_imputation_overrides": {},
        "multiple_imputation_m": 10,
        "multiple_imputation_numeric_consolidation": "mean",
        # Off by default -- gates Module N's final encoding pass
        # (binary -> 0/1, categorical -> one-hot, ordinal -> rounded
        # to nearest observed level). Deliberately opt-in: one-hot
        # encoding changes the SHAPE of the dataset (new columns,
        # originals removed), which isn't something a plain "clean my
        # Excel file" client delivery should ever do without being
        # explicitly asked for.
        "prepare_for_ml": False,
        "binary_positive_value_overrides": {},
        "multivariate_outlier_contamination": 0.05,
        "ordinal_category_order_overrides": {},
        "numeric_text_value_overrides": {},
        # Explicit, opt-in list of columns to REMOVE entirely from
        # ML-ready output (typically identifiers/personal info/
        # timestamps) -- never inferred, only ever what the client
        # explicitly confirmed in the app.
        "drop_non_analytical_columns": [],
    }
    if advanced_settings:
        settings.update(advanced_settings)
    return settings


def _run_pre_imputation_steps(df: pd.DataFrame, profile: dict, answers: dict, settings: dict) -> dict:
    """
    Everything in the pipeline that happens BEFORE missing-value
    imputation, and is fully DETERMINISTIC (no randomness involved) --
    disguised-missing markers, numeric-text coercion, text
    standardisation, duplicate removal, date/time standardisation, and
    outlier detection/capping.

    Factored out as its own step so it only ever runs ONCE regardless
    of which imputation output mode is requested afterwards
    (consolidated single dataset, or M complete datasets).

    Returns a dict bundling everything the imputation step and the
    post-imputation step need.
    """
    actions_log = []

    before_summary = ingestion.diagnostic_summary(df)

    # Runs FIRST of all, before every other step -- these are direct,
    # client-authored corrections to specific values they spotted
    # themselves while browsing "Explore unique values per column"
    # (e.g. "Scotlant" -> "Scotland"), so there's no reason for
    # anything else in the pipeline to see the uncorrected value at
    # all. Exact-match only, never a guess -- see Module 57's own
    # docstring.
    df, value_correction_log = value_corrections_mod.apply_value_corrections(
        df, answers.get("value_corrections", {})
    )
    actions_log.extend(value_correction_log)

    df, disguised_log = apply_disguised_missing_answers(df, profile, answers)
    actions_log.extend(disguised_log)

    df, extra_missing_log = apply_user_specified_missing_values(df, settings.get("extra_missing_values", []))
    actions_log.extend(extra_missing_log)

    df, column_specific_missing_log = apply_column_specific_missing_values(
        df, settings.get("column_specific_missing_values", {})
    )
    actions_log.extend(column_specific_missing_log)

    # Runs FIRST, before both ordinal ranking and numeric-text
    # coercion -- for an "integer"/"continuous" column that mixes real
    # numbers with a few non-numeric text labels (e.g. "doors" with
    # 2/3/4/"5more"), turns each client-mapped text value into its
    # real number (e.g. "5more" -> 5) before anything else gets a
    # chance to silently turn it into a missing value instead.
    df, numeric_text_value_log = ml_ready.resolve_non_numeric_text_values(
        df, settings.get("numeric_text_value_overrides", {})
    )
    actions_log.extend(numeric_text_value_log)

    # Runs BEFORE numeric-text coercion, deliberately -- confirmed as a
    # real bug by testing against a car-evaluation-style dataset: a
    # column like "doors" (values 2/3/4/"5more") gets auto-detected as
    # role "numeric_text" (most values genuinely look numeric), so
    # coerce_numeric_text_columns would run FIRST and silently turn
    # "5more" into a missing value before the ranking step ever saw
    # the real text -- corrupting exactly the special top-of-scale
    # label ordinal ranking exists to handle correctly. Running this
    # first and patching `profile[col]["semantic_role"]` to
    # "numeric_measure" means coerce_numeric_text_columns naturally
    # skips this column afterwards (it only acts on role
    # "numeric_text"), with no special-casing needed there.
    df, ordinal_text_log = ml_ready.encode_ordinal_text_columns(
        df, profile, settings.get("ordinal_category_order_overrides", {})
    )
    actions_log.extend(ordinal_text_log)

    # Runs FIRST among the three "text needs rewriting before it's
    # numeric" steps -- deliberately before measurement/currency
    # conversion, not just alongside them. "fifty kg" has to become
    # "50 kg" before apply_measurement_unit_conversion's own parser
    # (which requires a leading digit) can do anything useful with it
    # at all; running these in the wrong order would leave every
    # word-form value looking exactly like unparseable garbage to
    # Modules 46/48. Only rewrites the NUMBER part -- never touches or
    # interprets any unit/currency text itself, which is why this can
    # sit safely upstream of both without needing to know anything
    # about weights, lengths, or currencies.
    df, text_number_log = text_num.apply_text_number_conversion(df, profile, answers)
    actions_log.extend(text_number_log)

    # Runs BEFORE numeric-text coercion, same reasoning as
    # encode_ordinal_text_columns just above: a weight/length column
    # written as text ("5kg", "10 lbs") would otherwise never get
    # picked up by coerce_numeric_text_columns at all (it only acts on
    # role "numeric_text", and a column full of "5kg"-style values
    # scores LOW on that check, since pd.to_numeric can't parse "5kg"
    # directly -- it isn't numeric-text, it's unit-tagged text). This
    # step strips the unit text, converts to a single target unit, and
    # patches profile[col] to "numeric_measure" so every later step
    # (numeric-text coercion, outlier detection, imputation routing,
    # validation) sees the column as the clean numeric measurement it
    # now actually is, not the stale pre-conversion text. The third
    # return value, standardized_measurement_units, is carried forward
    # in this function's returned dict and used ONLY at the very end of
    # run_interactive_pipeline() / run_interactive_pipeline_multiple_
    # datasets() to rename these columns (e.g. "Weight" -> "Weight
    # (kg)") -- deliberately not renamed here, since every step between
    # here and there still needs to look this column up by its
    # ORIGINAL name.
    df, measurement_log, standardized_measurement_units = measurement_conv.apply_measurement_unit_conversion(
        df, profile, answers
    )
    actions_log.extend(measurement_log)

    # Currency conversion runs right alongside measurement conversion,
    # for the same reasons: a "$100"-style column needs its currency
    # text stripped before coerce_numeric_text_columns can do anything
    # useful with it, and profile needs patching so downstream steps
    # see it as numeric. Returns its own standardized_currencies map,
    # kept SEPARATE from standardized_measurement_units (not merged)
    # since a column could in principle be flagged as both -- unlikely
    # in practice, but keeping the two maps distinct avoids one
    # silently overwriting the other if it ever happened.
    df, currency_log, standardized_currencies = currency_conv.apply_currency_conversion(
        df, profile, answers
    )
    actions_log.extend(currency_log)

    # Duration/Age conversion runs alongside measurement/currency, same
    # position and reasoning: a "5 days"-style column needs its unit
    # text stripped before coerce_numeric_text_columns can do anything
    # with it. Returns its own standardized_duration_units map, kept
    # separate from the other two for the same reason.
    df, duration_log, standardized_duration_units = duration_conv.apply_duration_conversion(
        df, profile, answers
    )
    actions_log.extend(duration_log)

    # Phone and email standardisation, run alongside the other four
    # standardisation features. Unlike weight/currency/duration, these
    # columns stay TEXT the whole way through (a formatted phone
    # number or email address is not numeric) -- no profile role
    # patching, no coerce_numeric_text_columns interaction, and no
    # end-of-pipeline column rename (there's no unit ambiguity a
    # rename would resolve here, just a formatting choice). Both
    # respect the client's per-value batch-review decisions
    # (f"phone_review__<col>__<idx>" / f"email_review__<col>__<idx>"
    # in `answers`, collected by app.py's review-table step BEFORE
    # this pipeline runs) and the spec's explicit "Don't know ->
    # missing" rule.
    df, phone_log = phone_conv.apply_phone_standardization(df, profile, answers)
    actions_log.extend(phone_log)

    df, email_log = email_conv.apply_email_standardization(df, profile, answers)
    actions_log.extend(email_log)

    df, numeric_text_log = coerce_numeric_text_columns(df, profile, answers)
    actions_log.extend(numeric_text_log)

    integer_like_columns = set()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0 and (non_null % 1 == 0).all():
                integer_like_columns.add(col)

    # Captured HERE, before any imputation runs, for the opt-in
    # Machine-Learning-Ready encoding step (Module N) -- an ordinal
    # column gets fractionally-imputed by KNN/MI and later needs
    # rounding back to a REAL observed level, which requires knowing
    # what those real levels were BEFORE imputation contaminated the
    # column with its own fractional estimates (see Module N's
    # round_ordinal_column docstring for why this can't be inferred
    # after the fact).
    #
    # Only keeps values that actually parse as numbers -- confirmed as
    # a real crash by a client report (comparing a leftover string
    # against an already-ranked number during sorted()) that couldn't
    # be reproduced from several deliberate test scenarios, meaning
    # some combination of settings (e.g. an ordinal column ranked only
    # partially, or a global "extra missing value" colliding with one
    # of its own category names) can leave a column with a genuine
    # mix of real numbers and leftover text at this point. Rather than
    # crash trying to sort mixed types, any non-numeric leftover is
    # simply excluded from the observed-levels list -- it isn't a
    # valid level to round TO anyway.
    ordinal_observed_levels = {}
    for col, info in profile.items():
        if info.get("variable_type") != "ordinal" or col not in df.columns:
            continue
        numeric_values = []
        for v in df[col].dropna().unique().tolist():
            try:
                numeric_values.append(float(v))
            except (TypeError, ValueError):
                continue
        if numeric_values:
            ordinal_observed_levels[col] = sorted(numeric_values)

    config = build_pipeline_config(df, profile, answers)

    df = text_std.trim_whitespace(df)
    lower_case_cols = {col: style for col, style in config["casing_rules"].items() if style == "lower"}
    title_case_cols = [col for col, style in config["casing_rules"].items() if style == "title"]
    canonicalize_cols = [col for col, style in config["casing_rules"].items() if style == "canonicalize"]
    country_cols = [col for col, style in config["casing_rules"].items() if style == "country"]
    state_cols = [col for col, style in config["casing_rules"].items() if style == "state"]
    if lower_case_cols:
        df = text_std.standardise_casing(df, lower_case_cols)
    if title_case_cols:
        df = text_std.standardise_casing(df, {col: "title" for col in title_case_cols})
    for col in canonicalize_cols:
        if col in df.columns:
            df[col] = _canonicalize_categorical_casing(df[col])
    for col in country_cols:
        if col in df.columns:
            df[col] = _standardize_country_column(
                df[col],
                merge_uk_constituents=settings.get("merge_uk_constituents_by_column", {}).get(col, False),
                merge_us_states=settings.get("merge_us_states_by_column", {}).get(col, True),
            )
    for col in state_cols:
        if col in df.columns:
            df[col] = _standardize_state_column(df[col])
    if config["casing_rules"]:
        actions_log.append("Trimmed whitespace and standardised text casing")

    rows_before = len(df)
    if config["duplicate_detection_enabled"]:
        df = dedup.remove_exact_duplicates(df)
        if config["duplicate_subset_columns"]:
            df = dedup.remove_subset_duplicates(df, subset=config["duplicate_subset_columns"])
        actions_log.append(f"Removed {rows_before - len(df)} duplicate rows (exact + subset-based)")
    else:
        actions_log.append("Duplicate detection skipped -- no column(s) selected for it")

    total_date_failures = 0
    for col, date_info in config["date_columns"].items():
        if col not in df.columns:
            continue
        output_format = "%Y-%m-%d %H:%M:%S" if date_info["include_time"] else "%Y-%m-%d"
        df, n_failures = datetime_std.standardise_date_column(
            df, col, dayfirst=config["dayfirst"], output_format=output_format
        )
        total_date_failures += n_failures
    if config["date_columns"]:
        actions_log.append(
            f"Standardised date/time formats across {len(config['date_columns'])} column(s) "
            f"({total_date_failures} unparseable values flagged)"
        )

    for col in config["date_columns"]:
        da = profile.get(col, {}).get("date_ambiguity")
        if da and da.get("inferred_dayfirst") is not None and not da.get("needs_dayfirst_question", True):
            format_label = "DD-MM-YYYY" if da["inferred_dayfirst"] else "MM-DD-YYYY"
            actions_log.append(
                f"Inferred '{col}' date format as {format_label} from an unambiguous value in the "
                f"same column ('{da['inferred_dayfirst_example']}') -- no question needed"
            )

    numeric_cols_present = [c for c in config["numeric_columns"] if c in df.columns]

    # Defensive safety net: numeric_cols_present is built from
    # semantic-role classification, which should always mean a
    # genuinely numeric column by this point -- but confirmed as a
    # real crash risk by a client report (a TypeError doing string
    # arithmetic inside outlier detection) that couldn't be reproduced
    # from several realistic test scenarios, meaning some edge case in
    # real client data slips a still-text column past classification.
    # Rather than let that crash the entire pipeline, any column that
    # ISN'T actually numeric at this point is pulled out of numeric
    # processing entirely (not just skipped for outliers -- every
    # downstream numeric-only step, KNN/mean/median imputation
    # included, would hit the same crash otherwise) and flagged
    # honestly in the log rather than silently dropped.
    non_numeric_leaked = [c for c in numeric_cols_present if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_leaked:
        numeric_cols_present = [c for c in numeric_cols_present if c not in non_numeric_leaked]
        actions_log.append(
            f"Column(s) classified as numeric but still contain text, so numeric processing was "
            f"skipped for them (left as-is): {', '.join(non_numeric_leaked)}. This usually means the "
            f"column's role was misdetected -- worth a manual check."
        )

    n_outliers = {}
    outlier_multiplier = settings.get("outlier_iqr_multiplier", 1.5)
    cap_outliers_enabled = settings.get("cap_outliers", False)
    for col in numeric_cols_present:
        df = outliers.detect_outliers_iqr(df, col, multiplier=outlier_multiplier)
        n_outliers[col] = int(df[f"{col}_outlier_iqr"].sum())
        if cap_outliers_enabled:
            df = outliers.cap_outliers_iqr(df, col, multiplier=outlier_multiplier)
    if numeric_cols_present:
        outlier_summary = ", ".join(f"{v} {k}" for k, v in n_outliers.items())
        if cap_outliers_enabled:
            actions_log.append(f"Detected and capped outliers: {outlier_summary}")
        else:
            actions_log.append(
                f"Detected (but did NOT cap/modify) outliers: {outlier_summary} "
                f"-- enable 'Cap extreme outlier values' in Advanced Settings to change this"
            )

    return {
        "df": df,
        "config": config,
        "actions_log": actions_log,
        "before_summary": before_summary,
        "integer_like_columns": integer_like_columns,
        "numeric_cols_present": numeric_cols_present,
        "ordinal_observed_levels": ordinal_observed_levels,
        "standardized_measurement_units": standardized_measurement_units,
        "standardized_currencies": standardized_currencies,
        "standardized_duration_units": standardized_duration_units,
    }


def _run_deterministic_imputation(df: pd.DataFrame, config: dict, settings: dict, actions_log: list, numeric_cols_present: list) -> dict:
    """
    Runs every NON-stochastic imputation method (KNN, mean, median,
    zero, mode, leave) for numeric columns (and the numeric half of
    method assignment for categorical). Columns assigned "multiple"
    are deliberately SKIPPED here -- left exactly as they are (still
    missing) -- since Multiple Imputation is handled separately by the
    caller, either as one consolidated fill or as M distinct per-draw
    datasets.
    """
    numeric_default_method = settings.get("numeric_imputation_default", "knn")
    numeric_overrides = settings.get("numeric_imputation_overrides", {})
    categorical_default_method = settings.get("categorical_imputation_default", "knn")
    categorical_overrides = settings.get("categorical_imputation_overrides", {})

    numeric_method_for = {
        col: numeric_overrides.get(col, numeric_default_method) for col in numeric_cols_present
    }
    knn_numeric_cols = [c for c in numeric_cols_present if numeric_method_for[c] == "knn"]
    multiple_numeric_cols = [c for c in numeric_cols_present if numeric_method_for[c] == "multiple"]

    if knn_numeric_cols:
        n_missing_numeric = {col: int(df[col].isna().sum()) for col in knn_numeric_cols}
        df = missing_val.knn_impute_numeric(df, columns=knn_numeric_cols, k=settings["knn_k"])
        numeric_summary = ", ".join(f"{v} {k}" for k, v in n_missing_numeric.items())
        actions_log.append(f"Filled missing numeric values via KNN ({numeric_summary})")

    SIMPLE_NUMERIC_STRATEGIES = {"mean": "mean", "median": "median"}
    for col in numeric_cols_present:
        method = numeric_method_for[col]
        if method in ("knn", "multiple") or col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if method in SIMPLE_NUMERIC_STRATEGIES:
            df = missing_val.simple_impute(df, col, strategy=SIMPLE_NUMERIC_STRATEGIES[method])
            actions_log.append(f"Filled {n_missing} missing value(s) in '{col}' using {method} imputation")
        elif method == "zero":
            df = missing_val.simple_impute(df, col, strategy="constant", constant_value=0)
            actions_log.append(f"Filled {n_missing} missing value(s) in '{col}' with 0")
        elif method == "leave":
            actions_log.append(f"Left {n_missing} missing value(s) in '{col}' as-is (imputation skipped for this column)")

    categorical_method_for = {
        col: categorical_overrides.get(col, categorical_default_method)
        for col in config["categorical_knn_columns"]
    }
    multiple_categorical_cols = [
        c for c in config["categorical_knn_columns"] if categorical_method_for[c] == "multiple"
    ]

    return {
        "df": df,
        "numeric_method_for": numeric_method_for,
        "multiple_numeric_cols": multiple_numeric_cols,
        "categorical_method_for": categorical_method_for,
        "multiple_categorical_cols": multiple_categorical_cols,
    }


def _run_deterministic_categorical_imputation(
    df: pd.DataFrame, config: dict, settings: dict, actions_log: list,
    categorical_method_for: dict, complete_numeric_cols: list,
) -> pd.DataFrame:
    """
    The categorical half of deterministic imputation (KNN + mode) --
    separated from _run_deterministic_imputation because it needs
    complete_numeric_cols, which isn't known until numeric imputation
    (including any numeric Multiple Imputation) has actually finished.
    "multiple"-assigned categorical columns are still skipped here.
    """
    knn_categorical_cols = [c for c in config["categorical_knn_columns"] if categorical_method_for[c] == "knn"]

    if knn_categorical_cols and complete_numeric_cols:
        for cat_col in knn_categorical_cols:
            if cat_col not in df.columns:
                continue
            df = missing_val.knn_impute_categorical(
                df, categorical_column=cat_col, numeric_feature_columns=complete_numeric_cols, k=settings["knn_k"]
            )
        actions_log.append(f"Filled missing categorical values via KNN for: {', '.join(knn_categorical_cols)}")
    elif knn_categorical_cols:
        for cat_col in knn_categorical_cols:
            if cat_col in df.columns and df[cat_col].isna().any():
                df = missing_val.simple_impute(df, cat_col, strategy="mode")
        actions_log.append(
            f"No complete numeric columns available for KNN -- filled categorical column(s) "
            f"{', '.join(knn_categorical_cols)} using mode instead"
        )

    for col in config["categorical_knn_columns"]:
        method = categorical_method_for[col]
        if method in ("knn", "multiple") or col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if method == "mode":
            df = missing_val.simple_impute(df, col, strategy="mode")
            actions_log.append(f"Filled {n_missing} missing value(s) in '{col}' using mode imputation")
        elif method == "leave":
            actions_log.append(f"Left {n_missing} missing value(s) in '{col}' as-is (imputation skipped for this column)")

    return df


def _run_post_imputation_steps(
    df: pd.DataFrame, config: dict, settings: dict, actions_log: list,
    integer_like_columns: set, numeric_cols_present: list, profile: dict,
) -> pd.DataFrame:
    """
    Everything that happens AFTER every column's missing values are
    filled: free-text placeholder fill, rounding, dropping empty
    columns, column splitting, and validation. Applied identically
    regardless of imputation output mode -- including once per
    dataset when generating M complete datasets.
    """
    for col in config["free_text_columns"]:
        if col in df.columns and df[col].isna().any():
            df = missing_val.simple_impute(df, col, strategy="constant", constant_value="No information provided")
    if config["free_text_columns"]:
        actions_log.append(f"Filled missing free-text values with a placeholder: {', '.join(config['free_text_columns'])}")

    rounded_cols = []
    for col in numeric_cols_present:
        if col not in df.columns:
            continue
        # "integer" variable_type FORCES whole-number rounding even if
        # the ORIGINAL data wasn't 100% whole numbers (e.g. a count
        # column like "number of persons" with a stray data-entry typo
        # like 4.5) -- unlike integer_like_columns (which only catches
        # a column that HAPPENED to already be all-integer), this is
        # an explicit client statement that the variable conceptually
        # can never be fractional, regardless of what the raw data
        # looked like.
        if col in integer_like_columns or profile.get(col, {}).get("variable_type") == "integer":
            df[col] = df[col].round(0).astype("Int64")
        else:
            df[col] = df[col].round(2)
        rounded_cols.append(col)
    if rounded_cols:
        actions_log.append(
            f"Rounded numeric column(s) for a clean output (whole numbers where the "
            f"original data was whole-number or marked as an Integer variable, 2 decimal places otherwise): "
            f"{', '.join(rounded_cols)}"
        )

    cols_before_drop = set(df.columns)
    df = missing_val.drop_columns_with_missing(df, threshold_percent=settings["drop_column_missing_threshold"])
    dropped_cols = cols_before_drop - set(df.columns)
    if dropped_cols:
        actions_log.append(f"Dropped empty/near-empty column(s): {', '.join(dropped_cols)}")

    split_summary = []
    for col in config["full_name_split_columns"]:
        if col in df.columns:
            df = splitting.split_full_name(df, column=col)
            split_summary.append(col)
    for col in config["address_split_columns"]:
        if col in df.columns:
            df = splitting.split_address_column(df, column=col)
            split_summary.append(col)
    for col, date_info in config["date_columns"].items():
        if col in df.columns:
            df = splitting.split_datetime_column(df, col, include_time=date_info["include_time"])
            split_summary.append(col)
    if split_summary:
        actions_log.append(f"Split combined column(s) into separate fields: {', '.join(split_summary)}")

    n_invalid_email = n_invalid_phone = 0
    for col in config["email_columns"]:
        if col in df.columns:
            # BUG FIXED after client testing: this used to call
            # validation.validate_email_column(), a SEPARATE, much
            # simpler regex check completely disconnected from Module
            # 53's actual email_validator-based logic already used for
            # standardisation and the batch review table -- meaning a
            # value the review table correctly flagged as invalid
            # (and the client explicitly chose to KEEP as-is) could
            # still come back marked "valid" in the final Validation
            # Flags sheet, simply because it happened to have roughly
            # the right shape. Confirmed by a real client report: two
            # phone numbers correctly flagged and kept as-is in the
            # review table still showed as "valid" in the output.
            # Now uses the SAME parse_email_value() check as
            # everywhere else in this feature, so "valid" means the
            # same thing throughout the whole pipeline, not two
            # different definitions depending which step you look at.
            df[f"{col}_valid"] = df[col].apply(
                lambda v: np.nan if pd.isna(v) else email_std.parse_email_value(v)["is_valid"]
            )
            n_invalid_email += int((df[f"{col}_valid"] == False).sum())
    for col in config["phone_columns"]:
        if col in df.columns:
            # Same fix as email immediately above, same root cause --
            # uses the SAME phonenumbers-based parse_phone_value()
            # check Module 52/54 already use for standardisation and
            # the review table, with the SAME default_phone_region__
            # answer (if the client set one) so a bare national-format
            # number is judged against the correct country here too.
            default_region = config["phone_default_regions"].get(col)
            df[f"{col}_valid"] = df[col].apply(
                lambda v: np.nan if pd.isna(v) else phone_std.parse_phone_value(v, default_region)["is_valid"]
            )
            n_invalid_phone += int((df[f"{col}_valid"] == False).sum())
    n_invalid_postcode = 0
    for col in config["address_split_columns"]:
        # BUG FIXED alongside the column-splitting rename below: this
        # used to check for a column literally named "postcode",
        # which only ever existed when there was exactly one address
        # column in the whole file. Now that split_address_column()
        # names its output "{original column}: Postcode" (so multiple
        # address columns, e.g. "Customer Address" and "Company
        # Address", don't collide), this needs to check the SAME
        # per-column name split_address_column actually produced,
        # rather than a single fixed string -- otherwise postcode
        # validation would have silently stopped running for every
        # file, not just multi-address ones.
        postcode_col = f"{col}: Postcode"
        if postcode_col in df.columns:
            UK_POSTCODE_PATTERN = r"^[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}$"
            df = validation.validate_pattern(df, postcode_col, UK_POSTCODE_PATTERN)
            n_invalid_postcode += int((df[f"{postcode_col}_valid"] == False).sum())
    if config["email_columns"] or config["phone_columns"]:
        actions_log.append(f"Flagged {n_invalid_email} invalid email(s) and {n_invalid_phone} invalid phone number(s)")
    if config["address_split_columns"]:
        actions_log.append(f"Flagged {n_invalid_postcode} invalid postcode(s) across {len(config['address_split_columns'])} address column(s)")

    return df


def run_basic_cleaning_pipeline(
    df: pd.DataFrame,
    profile: dict,
    answers: dict,
    output_file,
    advanced_settings: dict = None,
) -> pd.DataFrame:
    """
    The BASIC PACKAGE pipeline -- a deliberately small, FIXED subset of
    the Standard Package's full pipeline (run_interactive_pipeline),
    covering only what the Basic tier promises:
      1. Basic text clean-up (case standardisation, whitespace trimming)
      2. Duplicate detection and removal
      3. Missing value treatment (single-value imputation only -- no
         Multiple Imputation, a Standard-only feature)
      4. Outlier detection, FLAGGING ONLY -- never capped/modified,
         regardless of any setting, since "detection only" is the
         explicit promise of this tier
      5. Data cleaning report

    DELIBERATELY DOES NOT call _run_pre_imputation_steps() /
    _run_post_imputation_steps() (the Standard pipeline's own
    orchestration) -- those unconditionally run weight/length,
    currency, duration, phone, email, and datetime standardisation,
    plus column splitting and ML-ready encoding, for ANY column their
    own detectors flag, with no tier awareness at all. Reusing them
    here would silently give Basic Package clients Standard-tier
    processing the moment their file happened to contain a phone,
    email, weight, or date column -- exactly what this separate,
    intentionally minimal function exists to prevent. What IS reused
    (build_pipeline_config, the deterministic imputation functions,
    Module B/C/G's own building blocks) is reused because those are
    genuinely generic -- they only reflect whatever `profile`/`answers`
    actually contain, and Basic Package's own question set
    (generate_basic_mode_questions) never produces the standardisation-
    trigger answers that would cause anything Standard-only to run.
    """
    settings = _build_settings(advanced_settings)
    actions_log = []
    df = df.copy()

    before_summary = ingestion.diagnostic_summary(df)

    df, value_correction_log = value_corrections_mod.apply_value_corrections(
        df, answers.get("value_corrections", {})
    )
    actions_log.extend(value_correction_log)

    df, disguised_log = apply_disguised_missing_answers(df, profile, answers)
    actions_log.extend(disguised_log)

    df, extra_missing_log = apply_user_specified_missing_values(df, settings.get("extra_missing_values", []))
    actions_log.extend(extra_missing_log)

    df, column_specific_missing_log = apply_column_specific_missing_values(
        df, settings.get("column_specific_missing_values", {})
    )
    actions_log.extend(column_specific_missing_log)

    config = build_pipeline_config(df, profile, answers)

    # 1. Basic text clean-up: whitespace trimming + casing
    #    standardisation -- the exact same Module B/C calls (and the
    #    same FIVE casing rule types -- lower/title/canonicalize/
    #    country/state) the Standard pipeline uses for this step.
    # BUG FIXED after testing: an earlier version of this only handled
    # "lower" and "title", missing "canonicalize" (unifies inconsistent
    # casing of the SAME category, e.g. "tesco"/"TESCO"/"Tesco" -> one
    # consistent form) plus the country/state standardisation rules --
    # confirmed by testing against a real Company column that stayed
    # inconsistently cased in the output despite "basic text clean-up"
    # being exactly what this step promises.
    df = text_std.trim_whitespace(df)
    lower_case_cols = {col: style for col, style in config["casing_rules"].items() if style == "lower"}
    title_case_cols = [col for col, style in config["casing_rules"].items() if style == "title"]
    canonicalize_cols = [col for col, style in config["casing_rules"].items() if style == "canonicalize"]
    country_cols = [col for col, style in config["casing_rules"].items() if style == "country"]
    state_cols = [col for col, style in config["casing_rules"].items() if style == "state"]
    if lower_case_cols:
        df = text_std.standardise_casing(df, lower_case_cols)
    if title_case_cols:
        df = text_std.standardise_casing(df, {col: "title" for col in title_case_cols})
    for col in canonicalize_cols:
        if col in df.columns:
            df[col] = _canonicalize_categorical_casing(df[col])
    for col in country_cols:
        if col in df.columns:
            df[col] = _standardize_country_column(
                df[col],
                merge_uk_constituents=settings.get("merge_uk_constituents_by_column", {}).get(col, False),
                merge_us_states=settings.get("merge_us_states_by_column", {}).get(col, True),
            )
    for col in state_cols:
        if col in df.columns:
            df[col] = _standardize_state_column(df[col])
    if config["casing_rules"]:
        actions_log.append("Applied basic text clean-up: trimmed whitespace and standardised casing")

    # 2. Duplicate detection and removal
    rows_before_dedup = len(df)
    if config["duplicate_detection_enabled"]:
        df = dedup.remove_exact_duplicates(df)
        if config["duplicate_subset_columns"]:
            df = dedup.remove_subset_duplicates(df, subset=config["duplicate_subset_columns"])
        n_duplicates_removed = rows_before_dedup - len(df)
        actions_log.append(f"Removed {n_duplicates_removed} duplicate row(s)")
    else:
        actions_log.append("Duplicate detection skipped -- no column(s) selected for it")

    # 3. Missing value treatment -- single-value imputation only (no
    #    "multiple" method is ever produced by Basic Package's own
    #    settings, so multiple_numeric_cols/multiple_categorical_cols
    #    are always empty here -- nothing gets silently skipped).
    numeric_cols_present = [c for c in config["numeric_columns"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    det = _run_deterministic_imputation(df, config, settings, actions_log, numeric_cols_present)
    df = det["df"]
    complete_numeric_cols = [c for c in numeric_cols_present if c in df.columns and df[c].isna().sum() == 0]
    df = _run_deterministic_categorical_imputation(
        df, config, settings, actions_log, det["categorical_method_for"], complete_numeric_cols
    )

    # 4. Outlier detection -- FLAGGING ONLY. Never calls
    #    outliers.cap_outliers_iqr() at all, regardless of settings --
    #    unlike the Standard pipeline, there is no "cap outliers"
    #    option exposed here, since detection-without-modification is
    #    exactly what the Basic Package promises.
    outlier_multiplier = settings.get("outlier_iqr_multiplier", 1.5)
    n_outliers = {}
    for col in numeric_cols_present:
        df = outliers.detect_outliers_iqr(df, col, multiplier=outlier_multiplier)
        n_outliers[col] = int(df[f"{col}_outlier_iqr"].sum())
    if numeric_cols_present:
        outlier_summary = ", ".join(f"{v} {k}" for k, v in n_outliers.items())
        actions_log.append(f"Detected (flagged only, not modified) outliers: {outlier_summary}")

    # 5. Report
    after_summary = ingestion.diagnostic_summary(df)
    report.export_cleaning_report_to_excel(before_summary, after_summary, actions_log, output_file, cleaned_df=df)

    return df


def run_interactive_pipeline(
    df: pd.DataFrame,
    profile: dict,
    answers: dict,
    output_file,
    advanced_settings: dict = None,
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline, producing ONE CONSOLIDATED cleaned
    dataset -- the same behaviour this function has always had. Any
    column set to Multiple Imputation gets one representative value
    (mean/median of its draws) plus full uncertainty-disclosure
    columns (see 13_multiple_imputation.py).

    For M SEPARATE complete datasets instead (no averaging, no
    uncertainty columns -- for formal pooled statistical inference),
    use run_interactive_pipeline_multiple_datasets() instead.

    output_file can be a path (string) OR an in-memory file-like
    object (e.g. io.BytesIO).

    Returns the cleaned DataFrame, and writes the "Cleaned Data +
    Summary + Actions Log (+ Outlier Flags)" Excel workbook.
    """
    settings = _build_settings(advanced_settings)

    pre = _run_pre_imputation_steps(df, profile, answers, settings)
    df, config, actions_log = pre["df"], pre["config"], pre["actions_log"]
    integer_like_columns, numeric_cols_present = pre["integer_like_columns"], pre["numeric_cols_present"]

    det = _run_deterministic_imputation(df, config, settings, actions_log, numeric_cols_present)
    df = det["df"]
    multiple_numeric_cols = det["multiple_numeric_cols"]
    categorical_method_for, multiple_categorical_cols = det["categorical_method_for"], det["multiple_categorical_cols"]

    if multiple_numeric_cols:
        n_missing_numeric = {col: int(df[col].isna().sum()) for col in multiple_numeric_cols}
        df = multiple_imputation.multiple_impute_numeric(
            df,
            columns=numeric_cols_present,
            target_columns=multiple_numeric_cols,
            m=settings.get("multiple_imputation_m", 10),
            consolidation=settings.get("multiple_imputation_numeric_consolidation", "mean"),
        )
        numeric_summary = ", ".join(f"{v} {k}" for k, v in n_missing_numeric.items())
        added_cols = ", ".join(
            f"{c}_was_imputed/_imputation_sd/_imputation_lower_95/_imputation_upper_95"
            for c in multiple_numeric_cols
        )
        actions_log.append(
            f"Filled missing numeric values via Multiple Imputation ({numeric_summary}) "
            f"-- added columns: {added_cols}"
        )

    complete_numeric_cols = [c for c in numeric_cols_present if c in df.columns and df[c].isna().sum() == 0]
    df = _run_deterministic_categorical_imputation(df, config, settings, actions_log, categorical_method_for, complete_numeric_cols)

    if multiple_categorical_cols and complete_numeric_cols:
        for cat_col in multiple_categorical_cols:
            if cat_col not in df.columns:
                continue
            n_missing = int(df[cat_col].isna().sum())
            df = multiple_imputation.multiple_impute_categorical(
                df,
                categorical_column=cat_col,
                numeric_feature_columns=complete_numeric_cols,
                k=settings["knn_k"],
                m=settings.get("multiple_imputation_m", 10),
            )
            actions_log.append(
                f"Filled {n_missing} missing value(s) in '{cat_col}' via Multiple Imputation "
                f"-- added columns: {cat_col}_was_imputed/_top_probability/_categorical_uncertainty (+ per-category probabilities)"
            )
    elif multiple_categorical_cols:
        for cat_col in multiple_categorical_cols:
            if cat_col in df.columns and df[cat_col].isna().any():
                df = missing_val.simple_impute(df, cat_col, strategy="mode")
        actions_log.append(
            f"No complete numeric columns available for Multiple Imputation -- filled categorical "
            f"column(s) {', '.join(multiple_categorical_cols)} using mode instead (no uncertainty column added)"
        )

    df = _run_post_imputation_steps(df, config, settings, actions_log, integer_like_columns, numeric_cols_present, profile)

    if settings.get("prepare_for_ml", False):
        df, ml_ready_log = ml_ready.apply_ml_ready_encoding(
            df, profile,
            ordinal_observed_levels=pre["ordinal_observed_levels"],
            binary_positive_value_overrides=settings.get("binary_positive_value_overrides", {}),
        )
        actions_log.extend(ml_ready_log)

        drop_cols = settings.get("drop_non_analytical_columns", [])
        if drop_cols:
            df, actually_dropped = ml_ready.drop_non_analytical_columns(df, drop_cols)
            if actually_dropped:
                actions_log.append(
                    f"Removed non-analytical column(s) from ML-ready output (client-confirmed): "
                    f"{', '.join(actually_dropped)}"
                )

        df = _detect_multivariate_outliers(df, actions_log, contamination=settings.get("multivariate_outlier_contamination", 0.05))

    # LAST step before export: rename any standardised weight/length
    # columns to include their unit (e.g. "Weight" -> "Weight (kg)").
    # Deliberately here and not earlier -- see apply_final_column_
    # renames()'s own docstring for why every step above this needs the
    # ORIGINAL column name to still work.
    df = measurement_conv.apply_final_column_renames(df, pre["standardized_measurement_units"])
    # Same LAST-step treatment for currency (e.g. "Price" -> "Price
    # (USD)") -- apply_final_column_renames is fully generic, reused
    # as-is from Module 46 rather than Module 48 keeping its own copy.
    df = measurement_conv.apply_final_column_renames(df, pre["standardized_currencies"])
    # Same treatment for duration (e.g. "Tenure" -> "Tenure (year)").
    df = measurement_conv.apply_final_column_renames(df, pre["standardized_duration_units"])

    after_summary = ingestion.diagnostic_summary(df)
    report.print_cleaning_report(pre["before_summary"], after_summary, actions_log)
    report.export_cleaning_report_to_excel(pre["before_summary"], after_summary, actions_log, output_file, cleaned_df=df)

    print(f"\nSaved: {output_file} (Cleaned Data + Summary + Actions Log sheets)")
    return df


def run_interactive_pipeline_multiple_datasets(
    df: pd.DataFrame,
    profile: dict,
    answers: dict,
    advanced_settings: dict = None,
) -> list:
    """
    Like run_interactive_pipeline(), but for any column set to
    Multiple Imputation, produces M SEPARATE complete datasets instead
    of one consolidated dataset -- spec Section 5.1: "each dataset
    contains one plausible draw for every selected missing cell...
    suitable for formal statistical inference, pooled analysis, and
    training one model per imputed dataset."

    Columns using any OTHER method (KNN, mean, median, zero, mode,
    leave) are deterministic, so they're computed ONCE and are
    IDENTICAL across every one of the M datasets -- only the columns
    actually assigned "multiple" differ between datasets. None of them
    carry the uncertainty-disclosure columns (_was_imputed,
    _imputation_sd, _top_probability, etc.) -- those describe spread
    ACROSS draws, which isn't meaningful for one single draw.

    HONEST SIMPLIFICATION, worth being upfront about: categorical
    Multiple Imputation's neighbour search uses the numeric columns'
    CONSOLIDATED (mean-imputed) values as its distance features for
    every one of the M datasets, rather than recomputing neighbour
    relationships separately against each numeric draw. Recomputing
    per-draw would be more perfectly correct in principle, but far
    more expensive for a marginal difference in practice -- a
    deliberate, documented trade-off, not an oversight.

    Returns a list of dicts, one per dataset:
      [{"index": 1, "df": <DataFrame>, "actions_log": [...],
        "before_summary": {...}, "after_summary": {...}}, ...]
    -- before_summary is identical across every dataset (the
    pre-imputation state doesn't vary by draw); after_summary is
    computed per dataset. Deliberately returns DataFrames and summary
    dicts rather than writing files directly, so the caller (e.g. the
    Streamlit app) decides HOW to package them (a zip of Excel files,
    etc.) without this module needing to know about that.
    """
    settings = _build_settings(advanced_settings)
    m = settings.get("multiple_imputation_m", 10)

    pre = _run_pre_imputation_steps(df, profile, answers, settings)
    base_df, config = pre["df"], pre["config"]
    integer_like_columns, numeric_cols_present = pre["integer_like_columns"], pre["numeric_cols_present"]
    base_actions_log = pre["actions_log"]

    det = _run_deterministic_imputation(base_df, config, settings, base_actions_log, numeric_cols_present)
    base_df = det["df"]
    categorical_method_for, multiple_categorical_cols = det["categorical_method_for"], det["multiple_categorical_cols"]
    multiple_numeric_cols = det["multiple_numeric_cols"]

    original_numeric_missing = {col: base_df[col].isna().to_numpy() for col in multiple_numeric_cols}

    numeric_draws = None
    if multiple_numeric_cols:
        numeric_draws = multiple_imputation.compute_numeric_mi_draws(
            base_df, columns=numeric_cols_present, m=m
        )
        base_df = multiple_imputation.multiple_impute_numeric(
            base_df, columns=numeric_cols_present, target_columns=multiple_numeric_cols,
            m=m, consolidation=settings.get("multiple_imputation_numeric_consolidation", "mean"),
            draws=numeric_draws,
        )
        for col in multiple_numeric_cols:
            base_df = base_df.drop(columns=[
                f"{col}_was_imputed", f"{col}_imputation_sd",
                f"{col}_imputation_lower_95", f"{col}_imputation_upper_95",
            ], errors="ignore")

    complete_numeric_cols = [c for c in numeric_cols_present if c in base_df.columns and base_df[c].isna().sum() == 0]
    base_df = _run_deterministic_categorical_imputation(
        base_df, config, settings, base_actions_log, categorical_method_for, complete_numeric_cols
    )

    categorical_draws = {}
    if multiple_categorical_cols and complete_numeric_cols:
        for cat_col in multiple_categorical_cols:
            categorical_draws[cat_col] = multiple_imputation.compute_categorical_mi_draws(
                base_df, categorical_column=cat_col, numeric_feature_columns=complete_numeric_cols,
                k=settings["knn_k"], m=m,
            )
    elif multiple_categorical_cols:
        for cat_col in multiple_categorical_cols:
            if cat_col in base_df.columns and base_df[cat_col].isna().any():
                base_df = missing_val.simple_impute(base_df, cat_col, strategy="mode")
        base_actions_log.append(
            f"No complete numeric columns available for Multiple Imputation -- filled categorical "
            f"column(s) {', '.join(multiple_categorical_cols)} using mode instead (all M datasets identical for these columns)"
        )

    results = []
    for i in range(m):
        variant = base_df.copy()
        variant_log = list(base_actions_log)

        for col in multiple_numeric_cols:
            j = numeric_cols_present.index(col)
            mask = original_numeric_missing[col]
            variant.loc[mask, col] = numeric_draws[i][mask, j]
        if multiple_numeric_cols:
            variant_log.append(
                f"Filled missing numeric values via Multiple Imputation, draw {i + 1}/{m}: "
                f"{', '.join(multiple_numeric_cols)}"
            )

        for cat_col, per_row_draws in categorical_draws.items():
            for row_index, draws in per_row_draws.items():
                variant.loc[row_index, cat_col] = draws[i]
        if categorical_draws:
            variant_log.append(
                f"Filled missing categorical values via Multiple Imputation, draw {i + 1}/{m}: "
                f"{', '.join(categorical_draws.keys())}"
            )

        variant = _run_post_imputation_steps(variant, config, settings, variant_log, integer_like_columns, numeric_cols_present, profile)

        if settings.get("prepare_for_ml", False):
            variant, ml_ready_log = ml_ready.apply_ml_ready_encoding(
                variant, profile,
                ordinal_observed_levels=pre["ordinal_observed_levels"],
                binary_positive_value_overrides=settings.get("binary_positive_value_overrides", {}),
            )
            variant_log.extend(ml_ready_log)

            drop_cols = settings.get("drop_non_analytical_columns", [])
            if drop_cols:
                variant, actually_dropped = ml_ready.drop_non_analytical_columns(variant, drop_cols)
                if actually_dropped:
                    variant_log.append(
                        f"Removed non-analytical column(s) from ML-ready output (client-confirmed): "
                        f"{', '.join(actually_dropped)}"
                    )

            variant = _detect_multivariate_outliers(variant, variant_log, contamination=settings.get("multivariate_outlier_contamination", 0.05))

        # Same LAST-step rename as run_interactive_pipeline() -- applied
        # per variant, since each one is otherwise an independent copy.
        variant = measurement_conv.apply_final_column_renames(variant, pre["standardized_measurement_units"])
        variant = measurement_conv.apply_final_column_renames(variant, pre["standardized_currencies"])
        variant = measurement_conv.apply_final_column_renames(variant, pre["standardized_duration_units"])

        variant_after_summary = ingestion.diagnostic_summary(variant)
        results.append({
            "index": i + 1,
            "df": variant,
            "actions_log": variant_log,
            "before_summary": pre["before_summary"],
            "after_summary": variant_after_summary,
        })

    return results

if __name__ == "__main__":
    profiling = import_module("10_column_profiling")
    qgen = import_module("11_question_generation")

    filepath = "dirty_customer_data.xlsx"
    df = ingestion.load_data(filepath)
    profile = profiling.profile_dataframe(df)
    questions = qgen.generate_questions(profile)

    print("--- Questions that would be shown to the user ---")
    qgen.print_questions(questions)

    # Simulated answers -- standing in for what the Streamlit UI will
    # eventually collect. Picks the dayfirst option (Module J supplies
    # no default for this one deliberately) and accepts the
    # duplicate-detection default guess.
    simulated_answers = {}
    for q in questions:
        if q["id"] == "dayfirst":
            simulated_answers["dayfirst"] = True  # simulating the user picking "dayfirst"
        elif q["default"] is not None:
            simulated_answers[q["id"]] = q["default"]

    print("\n--- Simulated answers ---")
    print(simulated_answers)

    print("\n--- Running pipeline ---")
    cleaned_df = run_interactive_pipeline(df, profile, simulated_answers, "interactive_cleaned_output.xlsx")
    print("\nFinal shape:", cleaned_df.shape)
    print("Final columns:", list(cleaned_df.columns))
