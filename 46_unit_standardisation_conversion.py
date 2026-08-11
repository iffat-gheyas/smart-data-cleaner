"""
Module: Weight & Length Unit Standardisation -- Sub-step B (Apply Conversion)
------------------------------------------------------------------------------
Builds on Module 45 (detection + single-value parsing). This module is the
piece that actually REWRITES column values once the client has answered
Module 11's measurement question chain:
  - confirm_measurement__<col>   (yes/no -- is this really a weight/length column)
  - standardize_units__<col>     (yes/no -- does the client want it converted
                                   into one unit at all, or left as original text)
  - target_unit__<col>           (only asked when the column's values were
                                   genuinely mixed across 2+ units)
  - unresolved_unit__<col>       (only asked when some values had no
                                   determinable unit at all)

REDESIGNED after client feedback that the previous version's
`<col>_original_unit` companion column read as the pipeline "splitting"
or corrupting data, even when the underlying numbers were fine. Two
changes address that directly:
  1. The standardize_units__<col> gate: if the client says "No", the
     column is left COMPLETELY untouched -- "5kg" stays "5kg", no
     numeric conversion, no extra column, nothing added or removed.
  2. No more per-row audit column at all. A converted column's unit is
     now documented ONCE, in the column header itself (via
     apply_final_column_renames, below) -- e.g. "Weight" becomes
     "Weight (kg)" -- rather than repeated down an entire companion
     column. Per-row provenance is still captured, just as an aggregate
     count in the actions log ("source units were: 5 kg, 3 lb, ...")
     instead of a persistent column.

Three distinct situations for a column that HAS been confirmed +
standardized, all handled the same code path:
  1. MIXED units in one column (e.g. "5kg", "10 lbs") -- the client answered
     the target_unit question; every value gets converted to that unit.
  2. A SINGLE unit throughout, but written as text (e.g. every value is
     "5kg", "10kg", "7kg") -- no target_unit question was asked (nothing
     to choose between), but the column still needs its unit text STRIPPED
     to become a genuine numeric column. This case is easy to miss: "no
     question was generated" does NOT mean "no work to do" here.
  3. Bare numbers with the unit only implied by the column name (e.g.
     "Height_cm" already all plain numbers) -- already numeric, genuinely
     nothing to do, and no standardize-gate question was even generated
     for it (see Module 45's needs_standardization()).

HONEST LIMITATION, kept here rather than silently guessed around: a bare
number with NO recognisable unit ANYWHERE (no unit text in that value, no
unit implied by the column name), where the client HASN'T answered the
unresolved-unit question (or answered it with a real unit -- see below),
has no safe way to be converted. Rather than assume it's already in the
target unit (which could silently corrupt a real value), this module
leaves the NUMBER unchanged and counts it as "unresolved" in the actions
log, the same "don't guess, flag it" principle the project uses
everywhere else (e.g. dayfirst is never auto-guessed). If the client
answered the unresolved-unit question with unit_std.TREAT_AS_MISSING
("I don't know"), these values become genuine NaN instead -- see that
branch below.
"""

from importlib import import_module

import numpy as np
import pandas as pd

unit_std = import_module("45_unit_standardisation")


# ---------------------------------------------------------------------------
# 1. Fixed conversion ratios -- imported from Module 45, which now owns
#    these as the single source of truth (see that module's addendum:
#    keeping one copy avoids the two files silently drifting apart, e.g.
#    Module 45 gaining a "stone" unit that this module didn't know about).
# ---------------------------------------------------------------------------

CONVERSION_TABLES = {"weight": unit_std.WEIGHT_UNIT_TO_KG, "length": unit_std.LENGTH_UNIT_TO_M}


def convert_value(number: float, from_unit: str, to_unit: str, measurement_type: str) -> float:
    """
    Converts a single number between two units of the SAME measurement
    type, via the fixed base unit (kg for weight, m for length) -- e.g.
    convert_value(10, "lb", "kg", "weight") -> 4.5359237.

    Raises on an unrecognised unit or a type mismatch (e.g. asking to
    convert a length unit using measurement_type="weight") rather than
    silently returning a wrong number -- a caller passing bad units here
    is a bug worth surfacing immediately, not a data-quality issue worth
    softening.
    """
    table = CONVERSION_TABLES[measurement_type]
    if from_unit not in table:
        raise ValueError(f"Unrecognised {measurement_type} unit '{from_unit}'. Known: {sorted(table)}")
    if to_unit not in table:
        raise ValueError(f"Unrecognised {measurement_type} unit '{to_unit}'. Known: {sorted(table)}")
    base_value = number * table[from_unit]
    return base_value / table[to_unit]


# ---------------------------------------------------------------------------
# 2. Apply confirmed answers to the DataFrame
# ---------------------------------------------------------------------------

def _resolve_effective_target_unit(col_info: dict, answers: dict, col: str) -> str:
    """
    Works out what unit this column should end up in, in priority order:
      1. The client's explicit target_unit__<col> answer (mixed-unit case)
      2. The unit IMPLIED BY THE COLUMN NAME (e.g. "Weight_kg", "Height_cm")
         -- this is the column's own declared convention, and outranks a
         single stray embedded unit token found in the content (see #3).
      3. The single unit actually found in the values, if there's only
         one AND the name doesn't already imply a different one -- e.g.
         a column with no informative name where every value happens to
         say "kg".
      4. None -- genuinely nothing to convert (bare numeric column with
         no unit information anywhere; see module docstring)

    ORDER MATTERS, confirmed by a real bug caught in testing: a column
    named "Height_cm" that's almost entirely bare numbers, with a single
    stray "5.5 ft" value, has detected_units = {"ft": 1} -- just ONE
    distinct unit, satisfying the old priority-3-before-2 rule and
    wrongly making the whole column convert TO feet based on one
    outlier, instead of staying in the cm the name and the vast
    majority of values actually use. Checking the name first fixes this.
    """
    answered = answers.get(f"target_unit__{col}")
    if answered:
        return answered

    if col_info.get("implied_unit_from_name"):
        return col_info["implied_unit_from_name"]

    detected_units = col_info["detected_units"]
    if len(detected_units) == 1:
        return next(iter(detected_units))

    return None


def _to_display_string(value):
    """
    Converts a single cell to a clean string, WITHOUT the ".0" pandas/
    Excel tends to leave on whole-number floats (e.g. 83.0 -> "83", not
    "83.0") -- used only when "leaving a column untouched" still needs
    every cell to be a genuine string (see the standardize=False branch
    below and its accompanying bug note).
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def apply_measurement_unit_conversion(df: pd.DataFrame, profile: dict, answers: dict) -> tuple:
    """
    Runs Module 45's detection fresh against the CURRENT state of df,
    then, for every column confirmed as a weight/length measurement AND
    where the client said yes to standardizing it, converts every value
    to a single target unit using the fixed ratios above.

    Respects a "No" answer to confirm_measurement__<col> (this isn't
    really a measurement column) OR standardize_units__<col> (it is,
    but the client wants the original text left alone) by skipping that
    column ENTIRELY -- no numeric conversion, no columns added, values
    stay exactly as typed (e.g. "5kg" stays "5kg"). This is the direct
    fix for a client-reported issue: values and units used to always
    get split into a separate `<col>_original_unit` column, which read
    as the pipeline corrupting data even when the underlying numbers
    were fine -- the standardize gate now makes that splitting fully
    opt-in per column, and it no longer happens as a companion column
    at all (see next paragraph).

    REDESIGNED (previously added a `<col>_original_unit` audit column
    for every converted column): no more per-row audit column. Instead,
    returns a THIRD value, standardized_units -- {col: target_unit} for
    every column actually converted -- which the caller (Module 12)
    uses to rename the column itself at the very end of the pipeline
    (e.g. "Weight" -> "Weight (kg)"), once, so the unit is documented
    in exactly one place (the header) instead of duplicated down an
    entire extra column. Per-row provenance (what unit each value
    started in) is still captured, just as an AGGREGATE in the actions
    log rather than a persistent column, since the client only asked to
    see it once for transparency, not to keep it forever in the data.

    Runs BEFORE coerce_numeric_text_columns() in the pipeline and
    patches profile[col] in place (dtype -> "numeric", semantic_role ->
    "numeric_measure") ONLY for columns actually converted -- a column
    left alone per a "No" answer keeps its ORIGINAL profile entry
    untouched, since it's still genuinely text, and mislabelling it
    numeric would break every downstream step that trusts profile.

    Returns (df, actions_log, standardized_units).
    """
    df = df.copy()
    actions_log = []
    standardized_units = {}

    detected_measurements = unit_std.detect_measurement_columns(df)

    for col, info in detected_measurements.items():
        if col not in df.columns:
            continue

        confirm_answer = answers.get(f"confirm_measurement__{col}")
        if confirm_answer is False:
            continue  # client said this ISN'T a measurement column -- leave untouched

        if not unit_std.needs_standardization(info):
            # Already plain numeric with a name-implied unit (or nothing
            # detected at all) -- genuinely nothing to convert, and no
            # standardize-gate question was even generated for this
            # column (see Module 11), so there's no answer to check
            # here either. Still worth confirming the role in profile,
            # since the client DID confirm (or auto-accept) this as a
            # real measurement, just one that was already clean.
            if pd.api.types.is_numeric_dtype(df[col]):
                profile[col]["dtype"] = "numeric"
                profile[col]["semantic_role"] = "numeric_measure"
                profile[col]["role_confidence"] = 1.0
            continue

        standardize_answer = answers.get(f"standardize_units__{col}")
        if standardize_answer is False:
            # Client explicitly chose to keep the original VALUES
            # untouched -- "5kg" stays "5kg", "83" stays "83". No
            # dtype/profile change beyond this, since the column
            # genuinely isn't numeric yet; treating it as numeric here
            # would break outlier detection, imputation routing, and
            # validation downstream, all of which trust
            # profile[col]["dtype"].
            #
            # BUG FIXED after real-file testing: simply `continue`-ing
            # here (leaving the column exactly as pandas read it from
            # Excel) let a column keep a MIX of native types -- str for
            # "70 kg"-style cells, but genuine int/float for bare-number
            # cells like "83" (Excel stores those as numbers, not
            # text). That mix silently broke a LATER step,
            # text_std.trim_whitespace(): pandas' `.str.strip()`
            # accessor returns NaN (not an error, not the original
            # value) for any non-string entry in an object-dtype
            # column, so every bare-number cell quietly vanished into
            # a missing value despite this branch never touching them.
            # Previously EVERY column reaching that step had already
            # been forced into one consistent dtype (either fully
            # numeric via conversion, or naturally all-string) by
            # something upstream -- this "leave it alone" path was the
            # first case where a genuinely mixed-type column could
            # reach it. Fix: normalise every cell to a clean STRING
            # here (still no unit conversion, no numeric interpretation
            # -- purely a type consistency fix), which is what a
            # naturally-entered text/categorical column would have had
            # anyway.
            df[col] = df[col].apply(lambda v: _to_display_string(v) if pd.notna(v) else v)
            actions_log.append(
                f"Left '{col}' as original text, per client's choice not to standardise units "
                f"(e.g. values like {info['detected_units'] or info['bare_number_examples']} "
                f"remain exactly as entered)"
            )
            continue

        measurement_type = info["measurement_type"]
        implied_unit = info.get("implied_unit_from_name")

        # REDESIGNED (previously a single shared clarified_unit answer
        # applied to every unresolved value in the column): now mirrors
        # Module 48's currency redesign exactly -- a PER-VALUE lookup
        # (Module 11's generate_measurement_unresolved_unit_questions,
        # one dropdown per distinct value) PLUS an optional BLANKET
        # DEFAULT (has_default_unit__<col> / default_unit__<col>) that
        # only ever applies to genuinely BARE values (no unit text at
        # all) -- an unrecognised unit TOKEN (e.g. "8 kgz") always goes
        # through the per-value lookup regardless of any default,
        # since a blanket "no unit mentioned" answer can't resolve what
        # a garbled mention actually meant.
        unresolved_details = unit_std.find_unresolved_unit_details(df, col, info)
        value_to_resolution = {
            str(detail["raw_value"]): answers.get(f"unit_value_resolution__{col}__{i}")
            for i, detail in enumerate(unresolved_details)
        }
        has_default_answer = answers.get(f"has_default_unit__{col}")
        blanket_default_unit = (
            answers.get(f"default_unit__{col}") if has_default_answer is True else None
        )

        target_unit = _resolve_effective_target_unit(info, answers, col)
        if target_unit is None and blanket_default_unit and blanket_default_unit != unit_std.TREAT_AS_MISSING:
            # BUG FIXED after testing: a column that's ENTIRELY bare
            # numbers with zero embedded unit tokens anywhere (e.g. a
            # "Weight" column that's just "70", "82", "65", ... --
            # flagged purely via the strong name-keyword signal) has
            # NOTHING for _resolve_effective_target_unit's normal
            # checks to work with: no explicit target_unit__<col>
            # answer (no target-unit question was even asked, since
            # there's nothing to disambiguate), no name-implied unit,
            # and detected_units is completely empty (not just "one
            # unit", genuinely zero). Previously this made target_unit
            # stay None, and the WHOLE COLUMN was silently left
            # untouched -- even though the client HAD just confirmed a
            # default unit for exactly this situation. The blanket
            # default is the obvious target here: if every bare value
            # defaults to the same unit, that unit IS the column's
            # unit, nothing else to convert against.
            target_unit = blanket_default_unit

        if target_unit is None:
            # Shouldn't normally happen once needs_standardization() is
            # True (there's SOME unit evidence), but guards against the
            # pathological case of detected_units being non-empty yet
            # somehow unresolvable -- fail safe by leaving the column
            # alone rather than guessing.
            continue

        converted_values = []
        source_unit_counts = {}
        n_converted = 0
        n_already_target = 0
        n_unresolved = 0
        n_marked_missing = 0
        n_default_applied = 0
        unresolved_examples = []

        for raw in df[col]:
            number, canonical_unit, raw_token = unit_std.parse_quantity(raw)

            if number is None:
                # The cell itself was genuinely empty/non-numeric to begin
                # with -- this mirrors real missingness already present in
                # the source data, so NaN here is correct and expected,
                # NOT a new missing value being introduced.
                converted_values.append(np.nan)
                continue

            source_unit = canonical_unit or implied_unit
            is_bare_cell = source_unit is None and canonical_unit is None and raw_token is None

            if source_unit is None and is_bare_cell and blanket_default_unit:
                clarified_unit = blanket_default_unit
                if clarified_unit != unit_std.TREAT_AS_MISSING:
                    n_default_applied += 1
            else:
                clarified_unit = value_to_resolution.get(str(raw)) if source_unit is None else None

            if source_unit is None and clarified_unit == unit_std.TREAT_AS_MISSING:
                # The client explicitly said "I don't know" for values
                # like this one -- rather than force a guessed unit (which
                # would silently corrupt a real number) or leave a
                # confusing raw number sitting in an otherwise-standardised
                # column, this value becomes genuinely missing, exactly as
                # requested: a real number the client couldn't place a
                # unit on is less useful than an honestly-blank cell the
                # normal missing-value pipeline can then handle properly
                # (impute, or leave blank, per whatever the client decides
                # for missing values generally).
                converted_values.append(np.nan)
                n_marked_missing += 1
                continue

            source_unit = source_unit or clarified_unit

            if source_unit is None:
                # A REAL number is present, we just can't tell what unit
                # it's in -- leave the number untouched rather than guess
                # (see module docstring's honest limitation). If the
                # client answered the relevant per-value question, or a
                # blanket default applies, this branch is never reached
                # for THIS value at all.
                converted_values.append(number)
                n_unresolved += 1
                if len(unresolved_examples) < 5:
                    unresolved_examples.append(raw)
                continue

            converted = convert_value(number, source_unit, target_unit, measurement_type)
            converted_values.append(converted)
            source_unit_counts[source_unit] = source_unit_counts.get(source_unit, 0) + 1
            if source_unit == target_unit:
                n_already_target += 1
            else:
                n_converted += 1

        df[col] = converted_values

        source_breakdown = ", ".join(f"{n} {u}" for u, n in sorted(source_unit_counts.items()))

        if implied_unit == target_unit:
            # Column name already documents this exact unit (e.g.
            # "Height_cm" converted to cm) -- rename would just be
            # redundant ("Height_cm (cm)"), caught during testing.
            # Values are still fully converted/standardised, just no
            # entry added to standardized_units, so
            # apply_final_column_renames leaves the name as-is.
            rename_note = "column name already documents this unit, no rename needed"
        else:
            standardized_units[col] = target_unit
            rename_note = f"column will be labelled '{col} ({target_unit})' in the final output"

        profile[col]["dtype"] = "numeric"
        profile[col]["semantic_role"] = "numeric_measure"
        profile[col]["role_confidence"] = 1.0
        profile[col]["variable_type"] = "continuous"
        profile[col]["variable_type_confidence"] = 0.9

        log_line = (
            f"Standardised '{col}' to {target_unit} ({rename_note}): {n_converted} value(s) "
            f"converted, {n_already_target} value(s) already in {target_unit} -- source units "
            f"were: {source_breakdown}"
        )
        if n_default_applied:
            log_line += (
                f"; {n_default_applied} value(s) with no unit mentioned were assumed to be "
                f"{blanket_default_unit}, per client's confirmed default for '{col}'"
            )
        if n_marked_missing:
            log_line += (
                f"; {n_marked_missing} value(s) set to MISSING per client's 'I don't know' answer "
                f"to the unresolved-unit question"
            )
        if n_unresolved:
            log_line += (
                f"; {n_unresolved} value(s) left UNCONVERTED -- original number kept as-is because "
                f"no unit could be determined (e.g. {unresolved_examples})"
            )
        actions_log.append(log_line)

    return df, actions_log, standardized_units


def apply_final_column_renames(df: pd.DataFrame, standardized_units: dict) -> pd.DataFrame:
    """
    The LAST step for this feature, meant to be called at the very end
    of the pipeline (right before the final diagnostic summary and
    Excel export in Module 12) -- renames each standardised measurement
    column to include its unit, e.g. "Weight" -> "Weight (kg)".

    Deliberately NOT done at conversion time (inside
    apply_measurement_unit_conversion, above): every step between there
    and here -- outlier detection, imputation, validation, ML-ready
    encoding, the report -- looks columns up by their ORIGINAL name via
    `profile` and `answers`. Renaming early would silently break every
    one of those lookups. Renaming once, at the end, after all of that
    processing has already run under the stable original name, avoids
    that entirely.

    Trusts standardized_units as given -- apply_measurement_unit_
    conversion is responsible for NOT including a column here at all if
    its name already documents the unit (e.g. "Height_cm" converted to
    cm), so there's nothing to double-check on this end.
    """
    rename_map = {
        col: f"{col} ({unit})" for col, unit in standardized_units.items() if col in df.columns
    }
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame({
        "Weight": ["5kg", "10 lbs", "7", "3.2 kg", "8 lbs", "6.5kg", None, "12 stone"],
        "Height_cm": [170, 165, 180, 175, "5.5 ft", 160, 172, None],
        "Package_Weight": ["2kg", "2kg", "2kg", "2kg", "2kg", "2kg", "2kg", "2kg"],  # single-unit-as-text case
        "Family_Size": [4, 3, 2, 5, 4, 3, 2, 1],
    })

    profile = {
        "Weight": {"dtype": "text", "semantic_role": "free_text"},
        "Height_cm": {"dtype": "text", "semantic_role": "free_text"},
        "Package_Weight": {"dtype": "text", "semantic_role": "free_text"},
        "Family_Size": {"dtype": "numeric", "semantic_role": "numeric_measure"},
    }

    print("=== BEFORE ===")
    print(df)

    # Simulated answers: client confirms Height_cm is a real measurement
    # and wants it standardised (no target_unit__Height_cm answer exists
    # -- Module 11 never asked that question, since only ONE embedded
    # unit token was found in the content; this is the exact scenario
    # the priority-order bug was caught on). Client also confirms and
    # standardises Weight, picking kg. Family_Size (the decoy) is
    # rejected as NOT a measurement. Package_Weight is confirmed as a
    # real measurement but the client explicitly says NO to
    # standardising it -- exercising the new "leave it exactly as
    # typed" path.
    answers = {
        "confirm_measurement__Height_cm": True,
        "standardize_units__Height_cm": True,
        "confirm_measurement__Family_Size": False,
        "standardize_units__Weight": True,
        "target_unit__Weight": "kg",
        "standardize_units__Package_Weight": False,
    }

    converted_df, log, standardized_units = apply_measurement_unit_conversion(df, profile, answers)

    print("\n=== AFTER conversion (Package_Weight untouched, no new columns anywhere) ===")
    print(converted_df)
    print("\nColumns:", list(converted_df.columns))

    print("\n=== Actions log ===")
    for line in log:
        print(" -", line)

    print("\n=== standardized_units (feeds the final rename) ===")
    print(standardized_units)

    final_df = apply_final_column_renames(converted_df, standardized_units)
    print("\n=== AFTER final rename (simulating end-of-pipeline step) ===")
    print(final_df)
    print("\nFinal columns:", list(final_df.columns))

    print("\n=== Patched profile entries ===")
    for col in ["Weight", "Height_cm", "Package_Weight", "Family_Size"]:
        print(f"  {col}: {profile[col]}")
