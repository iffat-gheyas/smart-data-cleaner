"""
Module Y: Imputation Approach & Method Selection (Questions)
--------------------------------------------------------------
First step of Phase B. This module ONLY generates the questions --
same split as Module O (15) generating questions before Module P (16)
implemented the actual splitting logic.

REVISED after building the backtesting module (AA/27): the original
design asked "which single-value method do you want?" UPFRONT, before
any evidence existed -- which stopped making sense the moment
backtesting could test every candidate automatically and show real
comparative accuracy. Asking someone to guess blindly at something the
system can now tell them empirically would be worse, not more
helpful. So:
  - generate_imputation_questions_for_column() now only asks approach
    (single vs multiple) and, if multiple, which multiple-imputation
    method -- both genuinely need to be known BEFORE running anything.
  - generate_single_method_override_question() is NEW, and generated
    AFTER the backtest has already run: it shows the recommended
    method (backtest's own top rank) and lets the client override to
    any OTHER method the backtest actually tested, with options built
    directly from the backtest's own results table rather than a
    separately-maintained list, so this can never drift out of sync
    with Module AA's actual candidate set.

GATING: uses Module V's diagnose_missing_values() /
columns_needing_imputation() directly -- a column with zero missing
values gets NO question at all, exactly the behaviour established
from the very first instruction on this whole Time Series build.

NO "LEAVE MISSING" OPTION, unlike the cross-sectional General-cleaning
mode: Time Series mode here IS the machine-learning-ready mode (there
is no separate "general cleaning" flavour of it), the same reasoning
that excluded "leave missing" from cross-sectional ML mode applies
identically -- a gap left in the series breaks most forecasting/ML
uses downstream. Flagging this as a deliberate call, not an oversight,
in case it's wanted back.
"""

MULTIPLE_IMPUTATION_METHODS = {
    "Seasonal-trend residual bootstrap (fits trend + seasonal pattern, draws plausible residuals)": "seasonal_bootstrap",
}


def generate_global_imputation_approach_question(columns_needing_imputation: list) -> dict:
    """
    Asked ONCE for the whole dataset, not per column. Single-value vs
    multiple imputation is a decision about how the FINISHED file will
    be used (one best-estimate number per gap, or several plausible
    draws for uncertainty-aware analysis) -- that's a property of the
    client's downstream analysis workflow, not something that
    naturally varies from one variable to the next. Previously asked
    separately for every column needing imputation, which was
    repetitive and didn't reflect a genuine per-column decision --
    fixed directly in response to that being confirmed unnecessary.

    The actual imputation METHOD/model choice still varies correctly
    per column (via the backtest, or the multiple-imputation method
    question) -- only this single upfront approach choice is now global.
    """
    columns_list = ", ".join(f"'{c}'" for c in columns_needing_imputation)
    return {
        "id": "ts_global_imputation_approach",
        "type": "single_select",
        "text": f"Missing values were found in: {columns_list}. How should these be filled in?",
        "options": [
            "Single-value imputation (one best estimate per missing value)",
            "Multiple imputation (several plausible estimates, for uncertainty-aware analysis)",
        ],
        "default": "Single-value imputation (one best estimate per missing value)",
    }


def generate_imputation_questions_for_column(value_column: str, diagnosis_entry: dict) -> list:
    """
    diagnosis_entry: Module V's diagnosis dict for ONE column (i.e.
    diagnosis[value_column], not the whole diagnosis dict).

    Returns [] if this column doesn't need imputation at all -- the
    caller should loop this over every value column and simply get
    nothing back for the clean ones, same pattern as every other
    dynamic-question generator in this pipeline.

    Only asks what genuinely needs deciding BEFORE anything runs:
    approach (single vs multiple), and -- only if multiple was chosen
    -- which multiple-imputation method. The single-method choice is
    NOT asked here any more; see generate_single_method_override_question().
    """
    if not diagnosis_entry["imputation_required"]:
        return []

    questions = []

    questions.append(
        {
            "id": f"ts_imputation_approach__{value_column}",
            "type": "single_select",
            "text": (
                f"'{value_column}' has {diagnosis_entry['missing_values']} missing value(s) "
                f"({diagnosis_entry['missing_percentage']}%). How should these be filled in?"
            ),
            "options": [
                "Single-value imputation (one best estimate per missing value)",
                "Multiple imputation (several plausible estimates, for uncertainty-aware analysis)",
            ],
            "default": "Single-value imputation (one best estimate per missing value)",
        }
    )

    questions.append(
        {
            "id": f"ts_multiple_method__{value_column}",
            "type": "single_select",
            "text": f"Which multiple-imputation method for '{value_column}'?",
            "options": list(MULTIPLE_IMPUTATION_METHODS.keys()),
            "default": list(MULTIPLE_IMPUTATION_METHODS.keys())[0],
            "shown_when": {"question_id": f"ts_imputation_approach__{value_column}", "equals": "Multiple imputation (several plausible estimates, for uncertainty-aware analysis)"},
        }
    )

    return questions


def generate_imputation_questions(diagnosis: dict) -> dict:
    """
    Runs generate_imputation_questions_for_column() across every
    column in Module V's full diagnosis dict. Returns
    {column_name: [questions]} -- columns needing nothing are simply
    absent from the returned dict entirely (not present with an empty
    list), so a caller can do `if column in questions_by_column` as a
    clean existence check.
    """
    result = {}
    for column, entry in diagnosis.items():
        qs = generate_imputation_questions_for_column(column, entry)
        if qs:
            result[column] = qs
    return result


def generate_single_method_override_question(value_column: str, backtest_results_df) -> dict:
    """
    Generated AFTER the backtest has actually run (Module AA/27) --
    unlike every other question in this module, this one needs real
    evidence as an input, not just the diagnosis, so it genuinely
    can't be produced upfront the way the others are.

    Options are built directly from the backtest's own "method" column
    (whatever candidates it actually tested, in ITS OWN ranked order --
    best first), rather than a separately-maintained list here, so
    this can never drift out of sync with whatever Module AA actually
    tested. Default is the top-ranked (recommended) method; choosing
    anything else in the UI is an explicit, informed override, made
    AFTER seeing the comparison table, not a blind guess beforehand.
    """
    ranked_methods = backtest_results_df["method"].tolist()
    default = ranked_methods[0] if ranked_methods else None
    return {
        "id": f"ts_single_method_override__{value_column}",
        "type": "single_select",
        "text": (
            f"Recommended method for '{value_column}': '{default}' (see the comparison table above). "
            f"Use this, or choose a different method the backtest tested:"
        ),
        "options": ranked_methods,
        "default": default,
    }


if __name__ == "__main__":
    import importlib.util
    import sys

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    import pandas as pd

    split_mod = load("split_mod", "16_time_series_split_validate.py")
    sort_mod = load("sort_mod", "17_time_series_sort.py")
    dup_mod = load("dup_mod", "18_time_series_duplicates.py")
    gaps_mod = load("gaps_mod", "19_time_series_gaps_and_timestep.py")
    val_mod = load("val_mod", "21_time_series_value_validation.py")
    diag_mod = load("diag_mod", "22_time_series_missing_diagnosis.py")
    backtest_mod = load("backtest_mod", "27_time_series_backtesting.py")

    component_meanings = [
        "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
        "Month",
    ]

    print("=" * 60)
    print("TEST 1: upfront questions -- approach only, NO blind single-method question")
    print("=" * 60)
    df = pd.read_csv("shampoo_missing_reconstructed.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(df, "Month", "-", component_meanings)
    sort_columns = sort_mod.get_temporal_sort_columns(split_report["component_columns"])
    sorted_df, _ = sort_mod.sort_time_series(split_df, split_report["component_columns"])
    resolved_df = sorted_df.copy()
    resolved_df["_period_duplicate_count"] = 1
    complete_df, gap_report = gaps_mod.build_complete_sequence(resolved_df, split_report["component_columns"], "Month", ["Sales"])
    freq_info = gaps_mod.detect_original_frequency(split_report["component_columns"])
    after_ph, _ = val_mod.apply_confirmed_placeholders(complete_df, {})
    resolved_vals, excl, manual, _ = val_mod.apply_non_numeric_resolutions(after_ph, {})
    final_df, _ = val_mod.finalize_numeric_conversion(resolved_vals, ["Sales"], excl, manual)
    diagnosis = diag_mod.diagnose_missing_values(final_df, ["Sales"])

    questions_by_column = generate_imputation_questions(diagnosis)
    print("Columns with questions:", list(questions_by_column.keys()))
    ids = [q["id"] for q in questions_by_column["Sales"]]
    print("Question ids:", ids)
    print("'ts_single_method__Sales' present (should be False -- removed):", any("ts_single_method__Sales" == i for i in ids))
    print("'ts_imputation_approach__Sales' present (should be True):", any("ts_imputation_approach__Sales" == i for i in ids))

    print("\n" + "=" * 60)
    print("TEST 2: post-backtest override question, built from REAL backtest results")
    print("=" * 60)
    series = final_df.set_index("Time_Step")["Sales"]
    backtest_results, _ = backtest_mod.run_backtest(series, cycle_length=freq_info["cycle_length"])
    override_q = generate_single_method_override_question("Sales", backtest_results)
    print("text:", override_q["text"])
    print("options (should exactly match the backtest's own ranked method list):")
    print(" ", override_q["options"])
    print("default (should be the rank-1 method):", override_q["default"])
    print("matches backtest rank 1:", override_q["default"] == backtest_results.iloc[0]["method"])
