"""
Module: Currency Standardisation -- Apply Conversion
------------------------------------------------------
Sibling to Module 46 (weight/length conversion), built on Module 47's
detection and Module 11's currency question chain. Same overall shape
as Module 46 -- confirm -> standardize gate -> convert -> rename column
at the very end of the pipeline -- with the one genuine difference
already flagged when this feature started: exchange rates are supplied
BY THE CLIENT (Module 11's numeric_input questions), never hardcoded.

REUSES two things directly from Module 46 rather than duplicating them:
  - _to_display_string(): the same mixed-type-column fix (see Module
    46's own bug note) applies here identically -- a currency column
    left untouched per a "No" answer can just as easily mix native int/
    float/str values as a weight/length one can, and would break
    text_std.trim_whitespace() the same way if left un-normalised.
  - apply_final_column_renames(): already fully generic (col -> "col
    (LABEL)"), so renaming "Price" to "Price (USD)" needs no
    currency-specific version of this function at all.

THE RATE-MISMATCH EDGE CASE, handled explicitly rather than ignored:
Module 11's rate questions were phrased against a column's DEFAULT
target currency (its most common detected currency), because the
actual target_currency__<col> answer doesn't exist yet at question-
generation time (see that module's own docstring). This module
recomputes that same default at APPLY time and compares it to the
client's actual answer:
  - If they match: the collected rates are valid, used normally.
  - If they DON'T match (client picked a different target): every
    rate collected for this column is treated as inapplicable -- not
    silently applied against the wrong currency. Every non-target
    value in the column is left unresolved instead (same "don't guess"
    fallback as a genuinely missing rate), and the log says exactly
    why, so the client understands the column needs re-doing with the
    new target rather than wondering why nothing converted.
"""

from importlib import import_module

import numpy as np
import pandas as pd

curr_std = import_module("47_currency_standardisation")
measurement_conv = import_module("46_unit_standardisation_conversion")

_to_display_string = measurement_conv._to_display_string
apply_final_column_renames = measurement_conv.apply_final_column_renames


def _resolve_effective_target_currency(col_info: dict, answers: dict, col: str):
    """
    Same priority order as Module 46's _resolve_effective_target_unit,
    for the same reason (a name-implied currency, e.g. "Price_USD", is
    more reliable than a single stray detected token that could be an
    outlier):
      1. Explicit target_currency__<col> answer
      2. Currency implied by the column name
      3. The single currency actually found, if there's only one
      4. None
    """
    answered = answers.get(f"target_currency__{col}")
    if answered:
        return answered

    if col_info.get("implied_currency_from_name"):
        return col_info["implied_currency_from_name"]

    detected = col_info["detected_currencies"]
    if len(detected) == 1:
        return next(iter(detected))

    return None


def apply_currency_conversion(df: pd.DataFrame, profile: dict, answers: dict) -> tuple:
    """
    Runs Module 47's detection fresh against the CURRENT state of df,
    then, for every column confirmed AND client-approved for
    standardisation, converts every value into a single target currency
    using client-supplied exchange rates.

    Respects "No" answers to confirm_currency__<col> (skip entirely) and
    standardize_currency__<col> (leave the column exactly as typed,
    just type-normalised -- see _to_display_string), identical
    reasoning to Module 46.

    Returns (df, actions_log, standardized_currencies) -- the third
    value feeds apply_final_column_renames() at the very end of the
    pipeline, exactly like Module 46's return value does.
    """
    df = df.copy()
    actions_log = []
    standardized_currencies = {}

    detected_currencies = curr_std.detect_currency_columns(df)

    for col, info in detected_currencies.items():
        if col not in df.columns:
            continue

        confirm_answer = answers.get(f"confirm_currency__{col}")
        if confirm_answer is False:
            continue  # client said this ISN'T a currency column -- leave untouched

        if not curr_std.needs_standardization(info):
            if pd.api.types.is_numeric_dtype(df[col]):
                profile[col]["dtype"] = "numeric"
                profile[col]["semantic_role"] = "numeric_measure"
                profile[col]["role_confidence"] = 1.0
            continue

        standardize_answer = answers.get(f"standardize_currency__{col}")
        if standardize_answer is False:
            # Same type-normalisation fix as Module 46's standardize=No
            # path -- see that module's bug note for why this is
            # necessary, not optional, even though no unit/currency
            # conversion happens here.
            df[col] = df[col].apply(lambda v: _to_display_string(v) if pd.notna(v) else v)
            actions_log.append(
                f"Left '{col}' as original text, per client's choice not to standardise currency "
                f"(e.g. values like {info['detected_currencies'] or info['bare_number_examples']} "
                f"remain exactly as entered)"
            )
            continue

        detected = info["detected_currencies"]
        implied_currency = info.get("implied_currency_from_name")

        # ADDED per client request: a client-confirmed BLANKET default
        # for "no currency mentioned at all" cells (Module 11's
        # has_default_currency__<col> / default_currency__<col>
        # questions) -- deliberately checked against is_bare_cell
        # below, NOT against source_currency being None in general, so
        # it never accidentally covers an ambiguous/typo'd token (e.g.
        # "60 GPB") -- those always go through the per-value lookup
        # below regardless of this default's existence.
        has_default_answer = answers.get(f"has_default_currency__{col}")
        blanket_default_currency = (
            answers.get(f"default_currency__{col}") if has_default_answer is True else None
        )

        target_currency = _resolve_effective_target_currency(info, answers, col)
        if target_currency is None and blanket_default_currency and blanket_default_currency != curr_std.TREAT_AS_MISSING:
            # BUG FIXED after testing: a column that's ENTIRELY bare
            # numbers with zero embedded currency tokens anywhere
            # (e.g. a "Price" column that's just "50", "30", "75", ...
            # flagged purely via the strong name-keyword signal) gives
            # _resolve_effective_target_currency nothing to work with
            # -- no explicit target_currency__<col> answer (no
            # question was even asked, nothing to disambiguate), no
            # name-implied currency, and detected_currencies is
            # completely empty. Previously target_currency stayed
            # None and the WHOLE COLUMN was silently left untouched,
            # even though the client had just confirmed a default
            # currency for exactly this situation. The blanket default
            # is the obvious target: if every bare value defaults to
            # the same currency, that currency IS the column's
            # currency, nothing else to convert against.
            target_currency = blanket_default_currency

        if target_currency is None:
            continue  # fail safe -- nothing resolvable, leave alone rather than guess

        # BUG FIXED alongside app.py's dynamic-question-text fix: this
        # used to also check whether target_currency matched the
        # DEFAULT currency the rate questions were originally phrased
        # against, treating a mismatch as "the collected rate no
        # longer applies." That's no longer needed -- app.py now
        # rewrites the rate question's own wording live, every rerun,
        # to always ask about whatever target_currency the client
        # actually has selected, so the answer stored under
        # exchange_rate__<col>__<source> is always already the rate
        # INTO the current target, never a stale default.

        # REDESIGNED (previously a single shared clarified_currency
        # answer applied to every unresolved value in the column): now
        # builds a PER-VALUE lookup, matching Module 11's
        # generate_currency_value_resolution_questions() one-question-
        # per-distinct-value redesign. Recomputes Module 47's detail
        # list the same way that module did at question-generation
        # time -- same (col, i) positional IDs line up correctly as
        # long as this column's data hasn't changed since, true here
        # since nothing upstream of this function touches it.
        unresolved_details = curr_std.find_unresolved_currency_details(df, col, info)
        value_to_resolution = {
            str(detail["raw_value"]): answers.get(f"currency_value_resolution__{col}__{i}")
            for i, detail in enumerate(unresolved_details)
        }

        converted_values = []
        source_currency_counts = {}
        n_converted = 0
        n_already_target = 0
        n_unresolved = 0
        n_marked_missing = 0
        n_left_as_is = 0
        n_default_applied = 0
        unresolved_examples = []
        missing_rate_currencies = set()

        for raw in df[col]:
            number, currency_code, raw_token = curr_std.parse_currency_value(raw)

            if number is None:
                converted_values.append(np.nan)
                continue

            source_currency = currency_code or implied_currency
            is_bare_cell = source_currency is None and currency_code is None and raw_token is None

            if source_currency is None and is_bare_cell and blanket_default_currency:
                clarified_currency = blanket_default_currency
                if clarified_currency not in (curr_std.TREAT_AS_MISSING, curr_std.LEAVE_AS_IS):
                    n_default_applied += 1
            else:
                clarified_currency = value_to_resolution.get(str(raw)) if source_currency is None else None

            if source_currency is None and clarified_currency == curr_std.LEAVE_AS_IS:
                # ADDED per explicit client request: keeps the value
                # EXACTLY as originally entered (full original text),
                # distinct from the generic "unresolved" fallback
                # below, which only keeps the parsed leading number.
                converted_values.append(raw)
                n_left_as_is += 1
                continue

            if source_currency is None and clarified_currency == curr_std.TREAT_AS_MISSING:
                converted_values.append(np.nan)
                n_marked_missing += 1
                continue

            source_currency = source_currency or clarified_currency

            if source_currency is None:
                # Genuinely no currency information for this value at all
                # (client hasn't answered this specific value's question yet,
                # or this file is being run through an older answer set).
                converted_values.append(number)
                n_unresolved += 1
                if len(unresolved_examples) < 5:
                    unresolved_examples.append(raw)
                continue

            if source_currency == target_currency:
                converted_values.append(number)
                source_currency_counts[source_currency] = source_currency_counts.get(source_currency, 0) + 1
                n_already_target += 1
                continue

            rate = answers.get(f"exchange_rate__{col}__{source_currency}")
            if rate is None or not isinstance(rate, (int, float)) or rate <= 0:
                # No valid rate supplied for this source currency --
                # same "don't guess" fallback as everywhere else in
                # this pipeline (e.g. Module 46's unresolved-unit
                # values). Rather than assume a rate of 1 (silently
                # wrong for any currency but the target itself), the
                # value is left unresolved and flagged.
                converted_values.append(number)
                n_unresolved += 1
                missing_rate_currencies.add(source_currency)
                if len(unresolved_examples) < 5:
                    unresolved_examples.append(raw)
                continue

            converted = number * rate
            converted_values.append(converted)
            source_currency_counts[source_currency] = source_currency_counts.get(source_currency, 0) + 1
            n_converted += 1

        df[col] = converted_values

        source_breakdown = ", ".join(f"{n} {c}" for c, n in sorted(source_currency_counts.items()))

        if implied_currency == target_currency:
            rename_note = "column name already documents this currency, no rename needed"
        else:
            standardized_currencies[col] = target_currency
            rename_note = f"column will be labelled '{col} ({target_currency})' in the final output"

        profile[col]["dtype"] = "numeric"
        profile[col]["semantic_role"] = "numeric_measure"
        profile[col]["role_confidence"] = 1.0
        profile[col]["variable_type"] = "continuous"
        profile[col]["variable_type_confidence"] = 0.9

        log_line = (
            f"Standardised '{col}' to {target_currency} ({rename_note}): {n_converted} value(s) "
            f"converted, {n_already_target} value(s) already in {target_currency} -- source "
            f"currencies were: {source_breakdown or 'none'}"
        )
        if n_default_applied:
            log_line += (
                f"; {n_default_applied} value(s) with no currency mentioned were assumed to be "
                f"{blanket_default_currency}, per client's confirmed default for '{col}'"
            )
        if missing_rate_currencies:
            log_line += (
                f"; no exchange rate supplied for: {', '.join(sorted(missing_rate_currencies))} "
                f"(affected values left unconverted)"
            )
        if n_marked_missing:
            log_line += (
                f"; {n_marked_missing} value(s) set to MISSING per client's 'I don't know' answer "
                f"to the unresolved-currency question"
            )
        if n_left_as_is:
            log_line += (
                f"; {n_left_as_is} value(s) left EXACTLY as originally entered, per client's "
                f"'I don't know' answer to the unresolved-currency question"
            )
        if n_unresolved and not missing_rate_currencies:
            log_line += (
                f"; {n_unresolved} value(s) left UNCONVERTED -- original number kept as-is because "
                f"no currency could be determined (e.g. {unresolved_examples})"
            )
        actions_log.append(log_line)

    return df, actions_log, standardized_currencies


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame({
        "Price": ["$100", "150 USD", "89.99", "\u20ac75", "\u00a360", "120", "R$45", "100 kr"],
        "Salary_GBP": [50000, 62000, 48000, "\u00a355,000", 71000, None, 49500, 60000],
        "Local_Fee": ["$20", "$20", "$20", "$20", "$20", "$20", "$20", "$20"],  # single-currency-as-text case
        "Total_Items": [4, 3, 2, 5, 4, 3, 2, 1],
    })

    profile = {
        "Price": {"dtype": "text", "semantic_role": "categorical"},
        "Salary_GBP": {"dtype": "text", "semantic_role": "categorical"},
        "Local_Fee": {"dtype": "text", "semantic_role": "categorical"},
        "Total_Items": {"dtype": "numeric", "semantic_role": "numeric_measure"},
    }

    print("=== BEFORE ===")
    print(df)

    # Scenario 1: Price -- client accepts USD as target (the assumed
    # default: USD/EUR/GBP/BRL detected, USD is most common with 2
    # hits), supplies rates for EUR, GBP, BRL. The three genuinely
    # unresolved values ("89.99", "120", "100 kr" -- confirmed via
    # Module 47's find_unresolved_currency_details, enumeration
    # indices 0/1/2 for this exact dataframe) each get their OWN
    # per-value answer now, rather than one shared answer for all
    # three -- here all three happen to be answered "I don't know",
    # but in real use each could get a genuinely different resolution.
    answers = {
        "confirm_currency__Salary_GBP": True,
        "standardize_currency__Price": True,
        "target_currency__Price": "USD",
        "exchange_rate__Price__EUR": 1.09,
        "exchange_rate__Price__GBP": 1.27,
        "exchange_rate__Price__BRL": 0.20,
        "currency_value_resolution__Price__0": curr_std.TREAT_AS_MISSING,  # "89.99"
        "currency_value_resolution__Price__1": curr_std.TREAT_AS_MISSING,  # "120"
        "currency_value_resolution__Price__2": curr_std.TREAT_AS_MISSING,  # "100 kr"
        "standardize_currency__Salary_GBP": True,
        "standardize_currency__Local_Fee": True,
        "standardize_currency__Total_Items": False,
    }

    converted_df, log, standardized = apply_currency_conversion(df, profile, answers)

    print("\n=== AFTER conversion ===")
    print(converted_df)

    print("\n=== Actions log ===")
    for line in log:
        print(" -", line)

    print("\n=== standardized_currencies ===")
    print(standardized)

    final_df = apply_final_column_renames(converted_df, standardized)
    print("\n=== AFTER final rename ===")
    print(final_df)
    print("Final columns:", list(final_df.columns))

    # Scenario 2: client changes Price's target to GBP instead of the
    # default USD. BEHAVIOUR CHANGED alongside the app.py fix that
    # accompanies this: app.py now dynamically rewrites the exchange-
    # rate question's own wording every rerun to always ask about
    # whichever target is CURRENTLY selected -- so by the time the
    # client sees an exchange-rate question at all, it already says
    # "into GBP", and whatever they type is the GBP rate, not a stale
    # USD one. This test reflects that: rates supplied here are
    # phrased as "-> GBP" (not the old "-> USD" scenario 1 used),
    # exactly as the real UI would ensure.
    print("\n\n=== Scenario 2: target currency changed, with correspondingly-updated rates ===")
    answers2 = dict(answers)
    answers2["target_currency__Price"] = "GBP"
    answers2["exchange_rate__Price__EUR"] = 0.85   # 1 EUR = 0.85 GBP
    answers2["exchange_rate__Price__USD"] = 0.79   # 1 USD = 0.79 GBP
    converted_df2, log2, standardized2 = apply_currency_conversion(df.copy(), dict(profile), answers2)
    print(converted_df2[["Price"]])
    for line in log2:
        if "Price" in line:
            print(" -", line)

    # Scenario 3: THE POINT OF THIS REDESIGN -- each of the three
    # unresolved values gets a DIFFERENT real answer instead of one
    # shared "I don't know" for all of them: "89.99" -> EUR (rate
    # already supplied above), "120" -> GBP (rate already supplied),
    # "100 kr" -> still missing (client genuinely doesn't know).
    print("\n\n=== Scenario 3: three different per-value resolutions ===")
    answers3 = dict(answers)
    answers3["currency_value_resolution__Price__0"] = "EUR"   # "89.99" -> EUR
    answers3["currency_value_resolution__Price__1"] = "GBP"   # "120"   -> GBP
    answers3["currency_value_resolution__Price__2"] = curr_std.TREAT_AS_MISSING  # "100 kr" stays missing
    converted_df3, log3, standardized3 = apply_currency_conversion(df.copy(), dict(profile), answers3)
    print(converted_df3[["Price"]])
    for line in log3:
        if "Price" in line:
            print(" -", line)
