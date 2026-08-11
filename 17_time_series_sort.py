"""
Module Q: Time Series Sorting
--------------------------------
Section 4 of the spec. Takes Module P's output (validated component
columns + a validity flag per row) and sorts into true temporal order.

TWO RULES THE SPEC IS EXPLICIT ABOUT:
1. Sort by the VALIDATED NUMERIC components, never the raw text --
   lexicographic sort on strings like '1-1', '1-10', '1-11', '1-2'
   would put '1-10' right after '1-1' (wrong), instead of after '1-9'
   (correct). Tested below with exactly that trap.
2. "Identifier" and "Custom" components are explicitly NOT part of the
   time ordering (Module O's question text says so directly) -- only
   genuine temporal meanings (Year/Quarter/Month/Week/Day/Hour/
   Minute/Second/Relative period index) participate in the sort.

WHAT HAPPENS TO INVALID ROWS: a row whose time components failed
Module P's validation has no reliable position -- guessing where it
belongs would be exactly the "silent correction" the spec repeatedly
prohibits. So invalid rows are excluded from the sorted temporal
sequence entirely and reported separately, rather than interleaved
based on incomplete information. They stay in the returned dataframe
(nothing is deleted) and remain visible for the client to resolve.
"""

import pandas as pd

# Meanings that genuinely define time ORDER. "Identifier" and
# "Custom / other" are deliberately excluded -- Module O's own
# question text says identifiers aren't part of the ordering, and
# "custom" has no defined temporal meaning to sort by.
ORDERING_MEANINGS = {
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
}


def get_temporal_sort_columns(component_columns: list) -> list:
    """
    component_columns: the list of (meaning, output_name) pairs from
    Module P's report. Returns just the output-column names that
    genuinely define time order, IN THE SAME POSITIONAL ORDER they
    were given -- e.g. Relative_Year before Month_Number, since that's
    the natural coarse-to-fine sort hierarchy (sort by year first,
    then by month within each year).
    """
    return [name for meaning, name in component_columns if meaning in ORDERING_MEANINGS]


def sort_time_series(df: pd.DataFrame, component_columns: list) -> tuple:
    """
    Returns (sorted_df, report).

    sorted_df: valid rows sorted by the temporal columns (ascending,
    stable -- true duplicate periods keep their original relative
    order, since deciding what to DO about duplicates is Section 5's
    job, not this one's), followed by any invalid rows appended at
    the end, still flagged, never dropped.
    """
    sort_columns = get_temporal_sort_columns(component_columns)
    if not sort_columns:
        raise ValueError(
            "No component was marked as a genuine time-ordering meaning (Year/Month/Week/etc.) -- "
            "nothing to sort by. Check the answers from Module O's questions."
        )

    valid_df = df[df["_time_component_valid"]].copy()
    invalid_df = df[~df["_time_component_valid"]].copy()

    original_index_order = valid_df.index.tolist()
    sorted_valid_df = valid_df.sort_values(by=sort_columns, kind="mergesort")
    new_index_order = sorted_valid_df.index.tolist()

    was_already_sorted = original_index_order == new_index_order
    n_rows_reordered = sum(1 for a, b in zip(original_index_order, new_index_order) if a != b)

    sorted_valid_df = sorted_valid_df.reset_index(drop=True)
    invalid_df = invalid_df.reset_index(drop=True)

    result_df = pd.concat([sorted_valid_df, invalid_df], ignore_index=True)

    report = {
        "sort_columns_used": sort_columns,
        "was_already_sorted": was_already_sorted,
        "n_rows_reordered": n_rows_reordered,
        "n_valid_rows_sorted": len(sorted_valid_df),
        "n_invalid_rows_excluded_from_sort": len(invalid_df),
    }

    return result_df, report


if __name__ == "__main__":
    import importlib.util
    import sys

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    split_mod = load("split_mod", "16_time_series_split_validate.py")

    print("=" * 60)
    print("TEST 1: real shampoo.csv -- already sorted, should report 0 changes")
    print("=" * 60)
    df = pd.read_csv("shampoo.csv")
    split_df, split_report = split_mod.split_and_validate_time_column(
        df,
        time_column="Month",
        separator="-",
        component_meanings=[
            "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
            "Month",
        ],
    )
    sorted_df, sort_report = sort_time_series(split_df, split_report["component_columns"])
    print("Sort columns used:", sort_report["sort_columns_used"])
    print("Already sorted:", sort_report["was_already_sorted"])
    print("Rows reordered:", sort_report["n_rows_reordered"])
    print("Valid rows sorted:", sort_report["n_valid_rows_sorted"], " Invalid excluded:", sort_report["n_invalid_rows_excluded_from_sort"])

    print("\n" + "=" * 60)
    print("TEST 2: shuffled shampoo.csv -- must detect reordering correctly")
    print("=" * 60)
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    split_shuf, split_shuf_report = split_mod.split_and_validate_time_column(
        shuffled,
        time_column="Month",
        separator="-",
        component_meanings=[
            "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
            "Month",
        ],
    )
    sorted_shuf, sort_shuf_report = sort_time_series(split_shuf, split_shuf_report["component_columns"])
    print("Already sorted:", sort_shuf_report["was_already_sorted"], "(should be False)")
    print("Rows reordered:", sort_shuf_report["n_rows_reordered"], "/", len(df))
    print("First 5 rows after sorting (should be 1-01..1-05 again):")
    print(sorted_shuf[["Month", "Relative_Year", "Month_Number"]].head(5))

    print("\n" + "=" * 60)
    print("TEST 3: the exact lexicographic-sort trap from the spec")
    print("   (text order would wrongly put '1-10','1-11' right after '1-1')")
    print("=" * 60)
    trap_df = pd.DataFrame({"Month": ["1-2", "1-10", "1-1", "1-11"], "Sales": [1, 2, 3, 4]})
    trap_split, trap_report = split_mod.split_and_validate_time_column(
        trap_df,
        time_column="Month",
        separator="-",
        component_meanings=[
            "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
            "Month",
        ],
    )
    trap_sorted, _ = sort_time_series(trap_split, trap_report["component_columns"])
    print("Sorted order:", trap_sorted["Month"].tolist(), "  (correct numeric order: ['1-1', '1-2', '1-10', '1-11'])")

    print("\n" + "=" * 60)
    print("TEST 4: mix of valid + invalid rows -- invalid rows must not vanish or get guessed into place")
    print("=" * 60)
    mixed_df = pd.DataFrame({"Month": ["1-05", "A-03", "1-01", "1-13", "1-03"], "Sales": [1, 2, 3, 4, 5]})
    mixed_split, mixed_report = split_mod.split_and_validate_time_column(
        mixed_df,
        time_column="Month",
        separator="-",
        component_meanings=[
            "Year (relative period — e.g. 'Year 1' of the series, not a real calendar year)",
            "Month",
        ],
    )
    mixed_sorted, mixed_sort_report = sort_time_series(mixed_split, mixed_report["component_columns"])
    print("Valid rows sorted:", mixed_sort_report["n_valid_rows_sorted"], " Invalid excluded from sort:", mixed_sort_report["n_invalid_rows_excluded_from_sort"])
    print(mixed_sorted[["Month", "Relative_Year", "Month_Number", "_time_component_valid"]].to_string(index=False))
