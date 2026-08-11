"""
Module N: Machine-Learning-Ready Encoding (Phase 2 -- new, opt-in)
---------------------------------------------------------------------
This module is what actually powers the "Machine Learning Ready —
Cross-Sectional Data" choice: it takes a fully-cleaned, fully-IMPUTED
dataframe (every
column already has zero missing values by the time this runs) and
applies a final ENCODING pass so the result is directly usable by a
model, based on each column's variable_type (see Module I's
guess_variable_type -- continuous / binary / categorical / ordinal,
reviewable and correctable by the user before this ever runs):

  - "binary"      -> recoded to 0/1 (whatever the original two values
                     were, e.g. 1/2, "Yes"/"No")
  - "categorical" -> one-hot encoded into one new 0/1 column PER
                     observed category
  - "ordinal"     -> rounded to the NEAREST OBSERVED category level
                     (not just the nearest integer -- ordinal levels
                     aren't guaranteed to be consecutive, e.g. 1, 2, 4,
                     5 with no 3, so "nearest integer" could invent a
                     level that never actually existed in the data)
  - "continuous"  -> left completely untouched

This is DELIBERATELY a separate, final, opt-in step -- never run as
part of "General cleaning" -- because one-hot encoding especially
changes the SHAPE of the dataset (new columns, original column
removed), which isn't something a client expects from a plain
"clean my Excel file" delivery. Missing-value IMPUTATION is a
completely separate, always-active concern (see Module K's
build_pipeline_config, which now uses variable_type -- not just
semantic role -- to decide whether a column gets numeric or
categorical treatment during imputation, regardless of whether this
encoding step ever runs).
"""

import pandas as pd


def recode_binary_column(series: pd.Series, positive_value=None) -> tuple:
    """
    Recodes a binary column's two observed values to 0/1.

    If `positive_value` is given, that raw value becomes 1 (and the
    other becomes 0) -- lets the caller say e.g. "Yes should be 1".
    Otherwise, a deterministic default: if the two values are already
    {0, 1}, they're left as-is; otherwise the value that sorts LATER
    (as a string) becomes 1 -- arbitrary but consistent, and always
    disclosed via the returned mapping so nothing is silently guessed
    without a paper trail.

    Returns (recoded_series, mapping) where mapping is
    {original_value: 0_or_1}, meant to go straight into an Actions Log
    entry for full transparency.
    """
    non_null = series.dropna()
    distinct = list(non_null.unique())
    if len(distinct) != 2:
        raise ValueError(f"recode_binary_column expects exactly 2 distinct values, found {len(distinct)}: {distinct}")

    if positive_value is not None and positive_value in distinct:
        pos = positive_value
        neg = [v for v in distinct if v != positive_value][0]
    elif set(distinct) == {0, 1}:
        pos, neg = 1, 0
    else:
        neg, pos = sorted(distinct, key=str)

    mapping = {neg: 0, pos: 1}
    recoded = series.map(mapping)
    return recoded, mapping


def one_hot_encode_column(df: pd.DataFrame, column: str) -> tuple:
    """
    One-hot / dummy-encodes a categorical column into one 0/1 column
    PER OBSERVED CATEGORY, named "<column>__<category>". Drops the
    original column.

    Assumes `column` has ALREADY been imputed (no missing values) --
    this is a final ENCODING step, not an imputation step. Raises if
    it still has missing values, rather than silently producing an
    all-zero row for a missing cell (which would look like a valid
    "none of the categories" observation instead of the truth: this
    value was never actually filled in).
    """
    if df[column].isna().any():
        raise ValueError(f"one_hot_encode_column: '{column}' still has missing values -- impute it first.")

    dummies = pd.get_dummies(df[column], prefix=column, prefix_sep="__").astype(int)
    new_df = df.drop(columns=[column])
    new_df = pd.concat([new_df, dummies], axis=1)
    return new_df, list(dummies.columns)


def round_ordinal_column(series: pd.Series, observed_levels: list) -> pd.Series:
    """
    Rounds a (possibly fractionally-imputed, e.g. via KNN or Multiple
    Imputation) ordinal column to the nearest value in
    `observed_levels`.

    `observed_levels` must be the set of levels that were ACTUALLY
    PRESENT in the ORIGINAL data, BEFORE imputation -- deliberately
    NOT inferred from `series` itself. Confirmed as a real bug by
    testing: by the time this function runs, `series` already
    contains the imputed (possibly fractional, possibly not a real
    level) estimates, so inferring "valid levels" from that same
    series would trivially treat any already-imputed value as valid
    just because it happens to already be there -- e.g. an imputed
    3.3 would count as its own "nearest observed level" and never get
    rounded to anything. The caller (Module K) captures the real
    pre-imputation levels before any imputation runs, and passes them
    in here explicitly.
    """
    if not observed_levels:
        return series

    def nearest_level(value):
        if pd.isna(value):
            return value
        return min(observed_levels, key=lambda lvl: abs(lvl - value))

    return series.apply(nearest_level)


def find_unresolved_text_values(series: pd.Series) -> list:
    """
    Returns the sorted list of DISTINCT non-null values in `series`
    that do NOT parse as a number -- the exact values a client needs
    to provide a numeric replacement for before an "integer" or
    "continuous" column can be treated as genuinely numeric, instead
    of silently becoming missing values.
    """
    non_null = series.dropna().astype(str).unique()
    unresolved = []
    for v in non_null:
        try:
            float(v)
        except ValueError:
            unresolved.append(v)
    return sorted(unresolved)


def resolve_non_numeric_text_values(df: pd.DataFrame, numeric_text_value_overrides: dict = None) -> tuple:
    """
    For an "integer" or "continuous" column that mixes genuinely
    numeric values with a few non-numeric text labels (e.g. a "doors"
    column with 2/3/4 alongside "5more"), replaces each non-numeric
    text value with a client-provided NUMBER (e.g. "5more" -> 5)
    rather than letting it silently become a missing value.

    Unlike ordinal ranking (encode_ordinal_text_columns), which only
    needs RELATIVE order, so mapping to a rank position 0..N-1 is
    fine -- an integer/continuous column's actual MAGNITUDE matters:
    "5more" needs to become the real number 5, not an arbitrary rank,
    because the column will be used as a genuine count or
    measurement, not just an ordered category.

    `numeric_text_value_overrides`: {column: {text_value: number}}.
    Matched case/whitespace-insensitively, since a client is thinking
    in terms of the value they SAW, not necessarily its exact
    internal representation. Only ever acts on values the client
    explicitly mapped -- anything left unmapped falls through to the
    normal numeric-text coercion afterwards (becomes a genuine missing
    value, handled by imputation like any other gap, exactly as
    before this feature existed).

    Returns (df, actions_log). Runs BEFORE numeric-text coercion, same
    timing reason as encode_ordinal_text_columns -- the real text
    needs to still be there to match against.
    """
    df = df.copy()
    actions_log = []

    for col, value_map in (numeric_text_value_overrides or {}).items():
        if col not in df.columns or not value_map:
            continue

        normalized_map = {str(k).strip().lower(): v for k, v in value_map.items()}

        def resolve(value, normalized_map=normalized_map):
            if pd.isna(value):
                return value
            key = str(value).strip().lower()
            return normalized_map.get(key, value)

        df[col] = df[col].apply(resolve)
        mapping_str = ", ".join(f"{k!r}->{v}" for k, v in value_map.items())
        actions_log.append(f"Replaced non-numeric text value(s) in '{col}': {mapping_str}")

    return df, actions_log


def encode_ordinal_text_columns(df: pd.DataFrame, profile: dict, ordinal_category_order_overrides: dict = None) -> tuple:
    """
    For every column with a USER-PROVIDED category order (lowest to
    highest -- e.g. ["none", "very low", "low", "moderate", "high",
    "very high"]), recodes its text categories to sequential integers
    (0 = lowest, N-1 = highest).

    Unlike everything else in this module, this runs BEFORE
    imputation, not after -- deliberately. Once a text ordinal column
    like "Very high"/"high"/... is recoded to plain integers, it can
    flow through the EXACT SAME numeric imputation machinery
    (KNN/mean/median/Multiple Imputation) as any other ordinal column
    that was already numeric to begin with, and the existing
    round_ordinal_column step (which runs LATE, after imputation)
    handles it correctly with no extra code path needed. There's no
    way to infer "high > moderate" from the text alone -- this is
    exactly the piece that genuinely needs the client's input, per the
    spec's own reasoning for why ordinal detection is never automatic.

    Also patches `profile[col]`'s dtype/semantic_role IN PLACE to
    reflect the new numeric reality -- profile was computed once, on
    the original text data, at upload time. Every downstream step
    (imputation-method routing, the integer-like-columns snapshot,
    etc.) needs to see a column that's ALREADY numeric, not the stale
    pre-recoding picture, or it would still try to canonicalise/case-
    standardise it as text.

    Returns (df, actions_log). A column whose provided order doesn't
    cover every value actually in the data leaves those specific cells
    unmapped (NaN) rather than guessing -- and says so in the log.
    """
    df = df.copy()
    actions_log = []

    for col, ordered_categories in (ordinal_category_order_overrides or {}).items():
        if col not in df.columns:
            continue

        rank_map = {cat: rank for rank, cat in enumerate(ordered_categories)}
        original_non_null = df[col].notna()
        recoded = df[col].map(rank_map)

        n_unmapped = int((original_non_null & recoded.isna()).sum())
        df[col] = recoded

        if col in profile:
            profile[col]["dtype"] = "numeric"
            profile[col]["semantic_role"] = "numeric_measure"
            # variable_type stays "ordinal" -- unchanged, and is what
            # the LATE encoding step still uses to round this column
            # to a valid level after imputation.

        rank_str = ", ".join(f"{cat!r}->{rank}" for cat, rank in rank_map.items())
        note = (
            f" ({n_unmapped} value(s) didn't match any ranked category and were left as missing)"
            if n_unmapped else ""
        )
        actions_log.append(f"Ranked ordinal column '{col}' as: {rank_str}{note}")

    return df, actions_log


def drop_non_analytical_columns(df: pd.DataFrame, columns_to_drop: list) -> tuple:
    """
    Removes columns the client EXPLICITLY chose to exclude entirely
    from the ML-ready output -- typically identifiers (patient ID),
    personal identifiers (name, email, phone, address), or timestamps.
    These already never get ENCODED by apply_ml_ready_encoding (see
    its EXCLUDED_ROLES guard), but staying in the output at all can
    still be undesirable for a model -- an ID column can leak
    information, and personal identifiers often shouldn't be fed into
    a model regardless of predictive value.

    Also drops any DERIVED column tied to something being removed --
    e.g. dropping "Email" without also dropping "Email_valid" (added
    earlier by Module H's validation step) would leave a validation
    flag pointing at a column that no longer exists in the output,
    which is confusing rather than useful. Confirmed as a real gap by
    testing.

    Purely opt-in and never inferred: unlike every encoding decision
    elsewhere in this module (which can be reasoned about after the
    fact, since the original value is still recoverable from context),
    removing a column is genuinely destructive -- there's no undo once
    it's gone from the output file. Only ever acts on columns the
    caller explicitly listed (plus their derived columns).

    Returns (df, actually_dropped) -- actually_dropped only includes
    columns that were genuinely present, for an accurate log entry.
    """
    df = df.copy()
    DERIVED_SUFFIXES = ("_valid", "_outlier_iqr")

    all_to_drop = set()
    for col in columns_to_drop:
        if col in df.columns:
            all_to_drop.add(col)
        for suffix in DERIVED_SUFFIXES:
            derived_col = f"{col}{suffix}"
            if derived_col in df.columns:
                all_to_drop.add(derived_col)

    actually_dropped = sorted(all_to_drop)
    if actually_dropped:
        df = df.drop(columns=actually_dropped)
    return df, actually_dropped


def encode_imputation_indicator_columns(df: pd.DataFrame) -> tuple:
    """
    Converts every "<column>_was_imputed" flag column from Python's
    True/False to 0/1.

    Multiple Imputation (Module M) adds these columns to disclose
    exactly which cells were originally missing -- genuinely useful
    information worth keeping, but boolean True/False isn't a
    machine-learning-ready encoding the way every other column in this
    mode's output is meant to be, since most ML libraries expect a
    numeric feature matrix rather than mixed bool/numeric dtypes.

    Deliberately scoped to columns ending in "_was_imputed" specifically
    (rather than sweeping every boolean-dtype column in the dataframe)
    so this never accidentally touches something else that happens to
    be boolean for an unrelated reason -- e.g. the outlier-flag columns
    ("_outlier_iqr", "multivariate_outlier") are also boolean but are
    diagnostic-only and get split into their own report sheet rather
    than shipped as a model-ready feature, so leaving them as True/False
    is intentional, not an oversight.

    Returns (df, actions_log).
    """
    df = df.copy()
    actions_log = []
    was_imputed_cols = [
        c for c in df.columns
        if c.endswith("_was_imputed") and pd.api.types.is_bool_dtype(df[c])
    ]
    for col in was_imputed_cols:
        df[col] = df[col].astype(int)
    if was_imputed_cols:
        actions_log.append(
            f"Encoded {len(was_imputed_cols)} imputation-indicator column(s) as 0/1 instead of "
            f"True/False: {', '.join(was_imputed_cols)}"
        )
    return df, actions_log


def apply_ml_ready_encoding(
    df: pd.DataFrame, profile: dict, ordinal_observed_levels: dict = None, binary_positive_value_overrides: dict = None
) -> tuple:
    """
    Runs the FINAL machine-learning-readiness encoding pass over every
    column still present in `df`, based on each column's variable_type
    in `profile` (already reflecting any user correction made in the
    app's editable profile table -- see app.py's variable_type_overrides).

    Meant to run AFTER all missing-value imputation is already
    complete, so every column processed here genuinely has zero
    missing values -- binary/categorical columns raise a clear error
    instead of silently mishandling a leftover missing cell (e.g. a
    column the user explicitly set to "Leave missing").

    `ordinal_observed_levels`: {column: [pre-imputation observed
    values]} -- REQUIRED for any column with variable_type "ordinal"
    to be rounded correctly (see round_ordinal_column's docstring for
    why this can't be inferred from the already-imputed data). A
    column missing from this dict is left untouched with a note in
    the log, rather than guessed at.

    `binary_positive_value_overrides`: optional {column: raw_value}
    dict letting the caller specify which of a binary column's two
    raw values should become 1 (e.g. "Yes" -> 1) instead of the
    default deterministic rule.

    Returns (encoded_df, actions_log).
    """
    df = df.copy()
    actions_log = []
    ordinal_observed_levels = ordinal_observed_levels or {}
    binary_positive_value_overrides = binary_positive_value_overrides or {}

    # Identifiers and structural fields are NEVER touched by this
    # step, regardless of what variable_type says. variable_type
    # detection (Module I's guess_variable_type) has no special case
    # for "this is an ID column" -- it happens to usually come out
    # "continuous" for a genuine ID (IDs are normally near-100% unique),
    # which is safe by coincidence, not by guarantee. Confirmed as a
    # real, if narrow, gap by testing: a small/unusual dataset can
    # produce an ID-like column with few enough distinct values that
    # it gets misread as "binary" or "categorical" -- this exclusion
    # makes "never encode an identifier" an explicit rule instead of
    # an accident of the numbers involved.
    EXCLUDED_ROLES = {
        "id_key", "full_name", "address", "email", "phone", "date_or_datetime", "free_text",
    }

    for col, info in list(profile.items()):
        if col not in df.columns:
            continue
        if info.get("semantic_role") in EXCLUDED_ROLES:
            continue
        variable_type = info.get("variable_type")

        if variable_type == "binary":
            if df[col].isna().any():
                actions_log.append(f"Skipped binary (0/1) encoding for '{col}' -- still has missing values")
                continue
            if df[col].nunique() != 2:
                actions_log.append(
                    f"Skipped binary (0/1) encoding for '{col}' -- expected exactly 2 distinct "
                    f"values, found {df[col].nunique()}"
                )
                continue
            recoded, mapping = recode_binary_column(df[col], positive_value=binary_positive_value_overrides.get(col))
            df[col] = recoded
            mapping_str = ", ".join(f"{k!r}->{v}" for k, v in mapping.items())
            actions_log.append(f"Recoded binary column '{col}' to 0/1 ({mapping_str})")

        elif variable_type == "categorical":
            if df[col].isna().any():
                actions_log.append(f"Skipped one-hot encoding for '{col}' -- still has missing values")
                continue
            n_categories = df[col].nunique()
            df, new_cols = one_hot_encode_column(df, col)
            actions_log.append(f"One-hot encoded '{col}' into {n_categories} column(s): {', '.join(new_cols)}")

        elif variable_type == "ordinal":
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Never got recoded by encode_ordinal_text_columns --
                # e.g. the client marked it ordinal but hasn't ranked
                # its categories yet. Rounding requires actual numbers
                # to do arithmetic on, so this has to skip rather than
                # crash on text.
                actions_log.append(
                    f"Skipped ordinal rounding for '{col}' -- still text-valued; provide a category "
                    f"ranking (lowest to highest) to encode this as ordinal"
                )
                continue
            levels = ordinal_observed_levels.get(col)
            if not levels:
                actions_log.append(
                    f"Skipped ordinal rounding for '{col}' -- no pre-imputation observed levels available"
                )
                continue
            df[col] = round_ordinal_column(df[col], levels)
            actions_log.append(f"Rounded ordinal column '{col}' to the nearest observed category level")

        # "continuous" -> deliberately no action.

    # Runs LAST, and independently of the variable_type loop above --
    # "_was_imputed" columns aren't in `profile` at all (they're
    # generated dynamically by Multiple Imputation, not part of the
    # original column set), so they'd never be reached by the loop.
    df, was_imputed_log = encode_imputation_indicator_columns(df)
    actions_log.extend(was_imputed_log)

    return df, actions_log
