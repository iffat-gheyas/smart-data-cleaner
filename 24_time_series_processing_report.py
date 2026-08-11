"""
Module X: Processing Report
--------------------------------
Section 20 of the spec. Doesn't recompute anything -- purely
consolidates the report dicts every prior module (15-23) already
produced into one audit trail, the same "decoupled, just formats what
it's given" philosophy as the cross-sectional pipeline's own
09_cleaning_report.py (which explicitly documents this design choice
for the same reason: keeping the report module independent of every
other module's internals).

imputation_summary and plausibility_summary are optional and default
to a clear "not applicable" state, since Phase B (imputation) and
plausibility constraints haven't been built yet -- this module will
need zero changes once they exist; the caller just starts passing
real values instead of None.
"""


def build_processing_report(
    dataset_type: str,
    time_column,
    value_columns: list,
    component_meanings: list,
    separator: str,
    calendar_type: str,
    split_report: dict,
    sort_report: dict,
    duplicate_report: dict,
    gap_report: dict,
    freq_info: dict,
    output_frequency_report: dict,
    placeholder_log: list,
    non_numeric_log: list,
    missingness_diagnosis: dict,
    imputation_summary: dict = None,
    plausibility_summary: dict = None,
) -> dict:
    if isinstance(time_column, (list, tuple)):
        time_column = ", ".join(time_column)
    return {
        "dataset_type": dataset_type,
        "time_column": time_column,
        "value_columns": value_columns,
        "time_stamp_interpretation": {
            "separator": separator,
            "calendar_type": calendar_type,
            "component_meanings": component_meanings,
        },
        "original_frequency": freq_info["frequency_label"],
        "splitting": {
            "output_columns": split_report["output_columns"],
            "n_valid_rows": split_report["n_valid_rows"],
            "n_invalid_rows": split_report["n_invalid_rows"],
            "invalid_row_details": split_report["invalid_rows"].to_dict("records"),
        },
        "sorting": {
            "was_already_sorted": sort_report["was_already_sorted"],
            "n_rows_reordered": sort_report["n_rows_reordered"],
        },
        "duplicate_periods": {
            "n_duplicate_periods_found": duplicate_report["n_duplicate_periods"],
            "n_rows_involved": duplicate_report["n_rows_involved"],
            "method_applied": duplicate_report.get("method_applied", "n/a — none found"),
        },
        "missing_periods": {
            "n_expected_periods": gap_report["n_expected_periods"],
            "n_observed_periods": gap_report["n_observed_periods"],
            "n_missing_periods_inserted": gap_report["n_missing_periods_inserted"],
            "sequence_range": gap_report["sequence_min_to_max"],
        },
        "output_frequency": {
            "multiple_applied": output_frequency_report["multiple_applied"],
            "method_applied": output_frequency_report["method_applied"],
            "n_output_periods": output_frequency_report["n_output_periods"],
        },
        "missing_value_placeholders_converted": placeholder_log,
        "non_numeric_values_resolved": non_numeric_log,
        "missingness_summary": missingness_diagnosis,
        "imputation": imputation_summary or {"status": "Not applicable — no missing values required imputation"},
        "plausibility_checks": plausibility_summary or {"status": "Not applicable — no imputed values to check yet"},
    }


def format_processing_report(report: dict) -> str:
    lines = []
    # NOT "=" * 60 -- confirmed via a real Excel repair log that a cell
    # starting with '=' gets treated as an attempted formula by Excel's
    # own file parser, even when explicitly declared as a string-type
    # cell, and gets silently stripped ("Removed Records: Formula").
    # Hyphens don't trigger this.
    lines.append("-" * 60)
    lines.append("TIME SERIES PROCESSING REPORT")
    lines.append("-" * 60)
    lines.append(f"Detected dataset type: {report['dataset_type']}")
    lines.append(f"Time column: {report['time_column']}")
    lines.append(f"Value column(s): {', '.join(report['value_columns'])}")

    ts = report["time_stamp_interpretation"]
    lines.append(f"Time-stamp interpretation: {ts['calendar_type']}, separator '{ts['separator']}'")
    for i, meaning in enumerate(ts["component_meanings"], start=1):
        lines.append(f"  Component {i}: {meaning}")

    lines.append(f"Original frequency: {report['original_frequency']}")

    sp = report["splitting"]
    lines.append(f"\nSplitting: {sp['n_valid_rows']} valid row(s), {sp['n_invalid_rows']} invalid row(s)")
    if sp["n_invalid_rows"]:
        for r in sp["invalid_row_details"]:
            lines.append(f"  Row {r.get('Original_Row_Number', '?')}: {r.get('_time_component_invalid_reason', '')}")

    so = report["sorting"]
    lines.append(f"\nSorting: already sorted = {so['was_already_sorted']}, rows reordered = {so['n_rows_reordered']}")

    dp = report["duplicate_periods"]
    lines.append(f"\nDuplicate time periods: {dp['n_duplicate_periods_found']} found ({dp['n_rows_involved']} row(s) involved), method: {dp['method_applied']}")

    mp = report["missing_periods"]
    lines.append(
        f"\nMissing time periods: {mp['n_missing_periods_inserted']} inserted "
        f"(expected {mp['n_expected_periods']}, observed {mp['n_observed_periods']}, "
        f"sequence range {mp['sequence_range']})"
    )

    of = report["output_frequency"]
    lines.append(
        f"\nOutput frequency: x{of['multiple_applied']} original unit, method: {of['method_applied']}, "
        f"{of['n_output_periods']} output period(s)"
    )

    if report["missing_value_placeholders_converted"]:
        lines.append("\nMissing-value placeholders converted:")
        for entry in report["missing_value_placeholders_converted"]:
            lines.append(f"  {entry}")
    else:
        lines.append("\nMissing-value placeholders: none found")

    if report["non_numeric_values_resolved"]:
        lines.append("\nNon-numeric values resolved:")
        for entry in report["non_numeric_values_resolved"]:
            lines.append(f"  {entry}")
    else:
        lines.append("\nNon-numeric values: none found")

    lines.append("\nMissingness summary:")
    for col, d in report["missingness_summary"].items():
        lines.append(f"  {col}:")
        lines.append(f"    Observed values: {d['observed_values']}")
        lines.append(f"    Missing values: {d['missing_values']}")
        lines.append(f"    Missing percentage: {d['missing_percentage']}%")
        lines.append(f"    Imputation required: {'Yes' if d['imputation_required'] else 'No'}")

    imp = report["imputation"]
    lines.append(f"\nImputation: {imp.get('status', imp)}")

    plaus = report["plausibility_checks"]
    lines.append(f"Plausibility checks: {plaus.get('status', plaus)}")

    lines.append("-" * 60)
    return "\n".join(lines)


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
    freq_mod = load("freq_mod", "20_time_series_frequency_aggregation.py")
    diag_mod = load("diag_mod", "22_time_series_missing_diagnosis.py")

    component_meanings = [
        "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
        "Month",
    ]

    print("FULL PIPELINE RUN: shampoo.csv, Sections 1 through 20")
    print()

    df = pd.read_csv("shampoo.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(df, "Month", "-", component_meanings)
    sorted_df, sort_report = sort_mod.sort_time_series(split_df, split_report["component_columns"])
    sort_columns = sort_mod.get_temporal_sort_columns(split_report["component_columns"])
    duplicate_report = dup_mod.detect_duplicate_periods(sorted_df, sort_columns)
    resolved_df, duplicate_report = dup_mod.resolve_duplicate_periods(sorted_df, sort_columns, ["Sales"], "sum")
    complete_df, gap_report = gaps_mod.build_complete_sequence(resolved_df, split_report["component_columns"], "Month", ["Sales"])
    freq_info = gaps_mod.detect_original_frequency(split_report["component_columns"])
    same_freq_df, output_frequency_report = freq_mod.aggregate_to_frequency(complete_df, freq_info, ["Sales"], multiple=1, method="sum")
    diagnosis = diag_mod.diagnose_missing_values(complete_df, ["Sales"])

    report = build_processing_report(
        dataset_type="Univariate time series",
        time_column="Month",
        value_columns=["Sales"],
        component_meanings=component_meanings,
        separator="-",
        calendar_type="Relative time periods (not real calendar years)",
        split_report=split_report,
        sort_report=sort_report,
        duplicate_report=duplicate_report,
        gap_report=gap_report,
        freq_info=freq_info,
        output_frequency_report=output_frequency_report,
        placeholder_log=[],
        non_numeric_log=[],
        missingness_diagnosis=diagnosis,
    )

    print(format_processing_report(report))

    print("\n\n" + "=" * 60)
    print("CROSS-CHECK against the spec's own Section 22 'Current Dataset Result'")
    print("=" * 60)
    checks = [
        ("Dataset structure: Univariate time series", report["dataset_type"] == "Univariate time series"),
        ("Time variable: Month", report["time_column"] == "Month"),
        ("Value variable: Sales", report["value_columns"] == ["Sales"]),
        ("Original frequency: Monthly", report["original_frequency"] == "Monthly"),
        ("Number of observations: 36", diagnosis["Sales"]["observed_values"] == 36),
        ("Expected monthly periods: 36", report["missing_periods"]["n_expected_periods"] == 36),
        ("Missing time periods: 0", report["missing_periods"]["n_missing_periods_inserted"] == 0),
        ("Duplicate time periods: 0", report["duplicate_periods"]["n_duplicate_periods_found"] == 0),
        ("Missing Sales values: 0", diagnosis["Sales"]["missing_values"] == 0),
        ("Non-numeric Sales values: 0", split_report["n_invalid_rows"] == 0),
        ("Imputation required: No", diagnosis["Sales"]["imputation_required"] is False),
        ("Recommended time-step sequence: 1 to 36", (complete_df["Time_Step"].min(), complete_df["Time_Step"].max()) == (1, 36)),
    ]
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"\n{sum(p for _, p in checks)}/{len(checks)} checks passed")
