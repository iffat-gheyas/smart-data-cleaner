"""
Module: Dataset Merge -- Row-wise Append & Column-wise Join
--------------------------------------------------------------------
Optional, standalone module -- runs only when the client explicitly
chooses to merge datasets, and does NOT modify anything about the
existing single-file cleaning pipeline (Modules 01-55). This is a
genuinely different kind of feature from everything else built so
far: it operates on TWO (or more) dataframes at once, not one.

Same "never silently guess" principle as the rest of this project,
applied to two distinct merge shapes:

  ROW-WISE APPEND (compare_columns_for_append / append_datasets):
  combining datasets that hold the SAME KIND of record (e.g. two
  months of the same sales export) by stacking rows underneath each
  other. Columns are matched by NAME, never by position -- a dataset
  with columns in a different order still aligns correctly. Any
  column present in one dataset but not another is surfaced to the
  client rather than silently dropped or silently filled.

  COLUMN-WISE JOIN (validate_merge_keys / find_overlapping_columns /
  join_datasets): combining DIFFERENT information about the SAME
  entities (e.g. customer details in one file, order totals in
  another) by matching rows on one or more key columns, exactly like
  a database join. The client always chooses the key column(s) and
  join type explicitly -- this module never guesses either.

Reuses existing modules rather than reimplementing them: duplicate
detection after an append is expected to run through Module 03
(03_duplicate_removal.py), not a new copy of that logic here.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Row-wise append (Merge Mode 1)
# ---------------------------------------------------------------------------

def compare_columns_for_append(datasets: list) -> dict:
    """
    Compares column sets across 2+ datasets (a list of DataFrames),
    entirely by NAME -- column ORDER is irrelevant here, matching the
    spec's explicit requirement ("match columns by name rather than
    column position").

    Returns:
      common_columns       columns present in EVERY dataset
      column_presence      {column_name: [list of dataset indices
                            (0-based) that HAVE this column]} -- lets
                            the caller show exactly which file(s) are
                            missing which column(s), not just a single
                            pairwise diff
      all_columns          the full union, in first-seen order across
                            all datasets (used as the final column
                            order for the appended result)
      has_differences       True if any column isn't present in every
                            dataset -- the trigger for asking the
                            client whether to continue
    """
    column_presence = {}
    all_columns = []

    for i, df in enumerate(datasets):
        for col in df.columns:
            if col not in column_presence:
                column_presence[col] = []
                all_columns.append(col)
            column_presence[col].append(i)

    n_datasets = len(datasets)
    common_columns = sorted(col for col, present_in in column_presence.items() if len(present_in) == n_datasets)
    has_differences = any(len(present_in) < n_datasets for present_in in column_presence.values())

    return {
        "common_columns": common_columns,
        "column_presence": column_presence,
        "all_columns": all_columns,
        "has_differences": has_differences,
    }


def append_datasets(datasets: list) -> tuple:
    """
    Row-wise append (Merge Mode 1) -- stacks 2+ datasets underneath
    each other, aligning columns by NAME. A column missing from one
    dataset is created for it and filled with missing values (NaN)
    for every row from that dataset -- never invented, never dropped
    silently.

    Returns (merged_df, actions_log, comparison) -- comparison is
    compare_columns_for_append()'s own output, useful for the client-
    facing "here's what differed" summary even after the merge runs.
    """
    comparison = compare_columns_for_append(datasets)
    all_columns = comparison["all_columns"]

    aligned = [df.reindex(columns=all_columns) for df in datasets]
    merged = pd.concat(aligned, ignore_index=True)

    actions_log = [
        f"Appended {len(datasets)} dataset(s) row-wise: {sum(len(d) for d in datasets)} total rows, "
        f"{len(all_columns)} column(s) after alignment"
    ]
    for col, present_in in comparison["column_presence"].items():
        if len(present_in) < len(datasets):
            missing_from = [i for i in range(len(datasets)) if i not in present_in]
            actions_log.append(
                f"Column '{col}' was missing from dataset(s) {missing_from} -- filled with missing "
                f"values for those rows"
            )

    return merged, actions_log, comparison


# ---------------------------------------------------------------------------
# 2. Column-wise join (Merge Mode 2)
# ---------------------------------------------------------------------------

JOIN_TYPE_DISPLAY = {
    "inner": "Inner Join -- keep only rows where the key matches in BOTH datasets",
    "left": "Left Join -- keep all rows from Dataset A, adding matched info from B where available",
    "right": "Right Join -- keep all rows from Dataset B, adding matched info from A where available",
    "outer": "Full Outer Join -- keep all rows from BOTH datasets, matched where possible",
}

_PANDAS_HOW = {"inner": "inner", "left": "left", "right": "right", "outer": "outer"}


def validate_merge_keys(df_a: pd.DataFrame, df_b: pd.DataFrame, key_columns: list) -> dict:
    """
    Pre-merge validation (spec Section "Validation") -- run BEFORE
    join_datasets(), so the client sees the consequences of their key
    choice before committing to it, exactly as the spec asks:
    "verify the selected key columns exist... report duplicate keys...
    report missing keys... report unmatched records."

    Returns:
      missing_key_columns_a / missing_key_columns_b   key columns that
                              don't exist in that dataset at all -- if
                              either is non-empty, the merge CANNOT
                              proceed with this key choice
      null_key_count_a / null_key_count_b   rows where the key itself
                              is missing (NaN in a key column) -- these
                              rows can never match anything and are
                              flagged rather than silently excluded
      duplicate_key_count_a / duplicate_key_count_b   rows sharing the
                              same key value(s) within ONE dataset --
                              worth knowing before a join, since it
                              multiplies matched rows unexpectedly
                              (a classic real-world join surprise)
      matching_key_count      number of DISTINCT key values present in
                              BOTH datasets
      only_in_a_count / only_in_b_count   distinct key values present
                              in only one dataset
    """
    missing_a = [k for k in key_columns if k not in df_a.columns]
    missing_b = [k for k in key_columns if k not in df_b.columns]

    result = {
        "missing_key_columns_a": missing_a,
        "missing_key_columns_b": missing_b,
    }
    if missing_a or missing_b:
        return result  # can't validate further -- the keys themselves don't exist

    null_key_count_a = df_a[key_columns].isna().any(axis=1).sum()
    null_key_count_b = df_b[key_columns].isna().any(axis=1).sum()

    duplicate_key_count_a = df_a.dropna(subset=key_columns).duplicated(subset=key_columns).sum()
    duplicate_key_count_b = df_b.dropna(subset=key_columns).duplicated(subset=key_columns).sum()

    keys_a = set(map(tuple, df_a.dropna(subset=key_columns)[key_columns].values.tolist()))
    keys_b = set(map(tuple, df_b.dropna(subset=key_columns)[key_columns].values.tolist()))

    result.update({
        "null_key_count_a": int(null_key_count_a),
        "null_key_count_b": int(null_key_count_b),
        "duplicate_key_count_a": int(duplicate_key_count_a),
        "duplicate_key_count_b": int(duplicate_key_count_b),
        "matching_key_count": len(keys_a & keys_b),
        "only_in_a_count": len(keys_a - keys_b),
        "only_in_b_count": len(keys_b - keys_a),
    })
    return result


def find_overlapping_columns(df_a: pd.DataFrame, df_b: pd.DataFrame, key_columns: list) -> list:
    """
    Non-key columns that exist, by NAME, in BOTH datasets -- these are
    exactly the ones that need a conflict-resolution decision (spec's
    "Duplicate Column Handling"), e.g. both files having their own
    "Country" column with possibly different values for the same
    customer.
    """
    return sorted(
        c for c in df_a.columns
        if c in df_b.columns and c not in key_columns
    )


def join_datasets(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_columns: list,
    how: str,
    column_conflict_choices: dict = None,
) -> tuple:
    """
    Column-wise join (Merge Mode 2). `how` is one of "inner"/"left"/
    "right"/"outer" (see JOIN_TYPE_DISPLAY for the client-facing
    explanation of each).

    `column_conflict_choices`: {column_name: "keep_a" | "keep_b" |
    "keep_both" | "prefer_non_missing"} for any column overlapping by
    NAME in both datasets (see find_overlapping_columns) --
      "keep_a" / "keep_b"     use that dataset's value, drop the other
      "keep_both"             keep BOTH as separate columns (suffixed
                               "_A" / "_B") -- the default when no
                               choice is given for a column, since
                               silently picking one side's value would
                               be exactly the kind of guess this
                               project avoids
      "prefer_non_missing"    use Dataset A's value unless it's
                               missing, in which case fall back to
                               Dataset B's -- useful when the two
                               datasets are meant to agree and one is
                               just more complete than the other

    Returns (merged_df, actions_log, match_summary). match_summary
    mirrors the spec's own report fields exactly (matching records,
    rows only in A, rows only in B).
    """
    column_conflict_choices = column_conflict_choices or {}
    overlapping = find_overlapping_columns(df_a, df_b, key_columns)

    merged = df_a.merge(
        df_b, on=key_columns, how=_PANDAS_HOW[how],
        suffixes=("_A", "_B"), indicator=True,
    )

    match_counts = merged["_merge"].value_counts()
    match_summary = {
        "matching_records": int(match_counts.get("both", 0)),
        "only_in_a": int(match_counts.get("left_only", 0)),
        "only_in_b": int(match_counts.get("right_only", 0)),
    }
    merged = merged.drop(columns=["_merge"])

    actions_log = [
        f"Joined two datasets on key(s) {key_columns} using a {how.upper()} join: "
        f"{match_summary['matching_records']} matching record(s), "
        f"{match_summary['only_in_a']} row(s) only in Dataset A, "
        f"{match_summary['only_in_b']} row(s) only in Dataset B"
    ]

    columns_added = []
    for col in overlapping:
        choice = column_conflict_choices.get(col, "keep_both")
        col_a, col_b = f"{col}_A", f"{col}_B"
        if col_a not in merged.columns or col_b not in merged.columns:
            continue  # shouldn't happen, but fail safe rather than KeyError

        if choice == "keep_a":
            merged[col] = merged[col_a]
            merged = merged.drop(columns=[col_a, col_b])
            actions_log.append(f"Column '{col}': kept Dataset A's value (client's choice), Dataset B's discarded")
        elif choice == "keep_b":
            merged[col] = merged[col_b]
            merged = merged.drop(columns=[col_a, col_b])
            actions_log.append(f"Column '{col}': kept Dataset B's value (client's choice), Dataset A's discarded")
        elif choice == "prefer_non_missing":
            merged[col] = merged[col_a].where(merged[col_a].notna(), merged[col_b])
            merged = merged.drop(columns=[col_a, col_b])
            actions_log.append(f"Column '{col}': used Dataset A's value, falling back to Dataset B's only where A was missing")
        else:  # "keep_both" -- the safe default, no silent pick
            actions_log.append(f"Column '{col}': kept BOTH datasets' values as '{col_a}' and '{col_b}' (no conflict resolution chosen)")

    # Any column that ONLY came from Dataset B (not overlapping, not a
    # key) is genuinely new information added by the join -- worth
    # naming explicitly in the report, matching the spec's "Columns
    # Added" field.
    columns_added = [c for c in df_b.columns if c not in df_a.columns and c not in key_columns]
    if columns_added:
        actions_log.append(f"Column(s) added from Dataset B: {columns_added}")

    return merged, actions_log, match_summary


# ---------------------------------------------------------------------------
# 3. Question generation
# ---------------------------------------------------------------------------
# Deliberately kept SEPARATE from Module 11's generate_questions() --
# that function's whole signature (profile, df) assumes ONE dataframe
# being cleaned; merge questions are about the RELATIONSHIP between
# TWO dataframes and belong to a genuinely different workflow (see
# this module's own header docstring). app.py calls these directly
# from a dedicated merge-mode section, not through the standard
# single-file question pipeline.

def generate_append_confirmation_question(comparison: dict):
    """
    Row-wise append's ONLY question -- per the spec: "If one dataset
    contains extra or missing columns: show the differences, ask the
    user whether to continue." Returns None if the datasets already
    have identical columns (nothing to confirm).
    """
    if not comparison["has_differences"]:
        return None

    n_datasets = max(len(v) for v in comparison["column_presence"].values())
    diff_lines = []
    for col, present_in in comparison["column_presence"].items():
        if len(present_in) < n_datasets:
            missing_from = [i for i in range(n_datasets) if i not in present_in]
            diff_lines.append(f"'{col}' (missing from dataset {missing_from})")
    diff_str = "; ".join(diff_lines)

    return {
        "id": "confirm_append_with_differences",
        "type": "yes_no",
        "question_text": (
            f"The datasets don't have identical columns: {diff_str}. Any missing column will be "
            f"created and filled with missing values for the rows that don't have it. Continue?"
        ),
        "options": [
            {"label": "Yes, continue with the append", "value": True},
            {"label": "No, stop -- let me fix the files first", "value": False},
        ],
        "default": True,
    }


def generate_key_column_question(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """
    "Which column(s) should be used to match rows?" -- per the spec,
    NEVER guessed, always an explicit client choice. Options are
    restricted to columns present, by name, in BOTH datasets -- a
    column only one side has can't possibly serve as a join key.
    Multi-select, supporting the spec's explicit composite-key
    requirement ("Customer_ID + Order_Date").
    """
    common_cols = sorted(set(df_a.columns) & set(df_b.columns))
    return {
        "id": "merge_key_columns",
        "type": "multi_select",
        "question_text": (
            "Which column(s) should be used to match rows between the two datasets? "
            "Select more than one for a composite key (e.g. Customer_ID + Order_Date)."
        ),
        "options": [{"label": c, "value": c} for c in common_cols],
        "default": [],
    }


def generate_join_type_question() -> dict:
    """The four join types, each explained in plain language, per the spec."""
    return {
        "id": "merge_join_type",
        "type": "single_select",
        "question_text": "How should rows be matched between the two datasets?",
        "options": [
            {"label": JOIN_TYPE_DISPLAY[k], "value": k} for k in ["inner", "left", "right", "outer"]
        ],
        "default": "left",
    }


def build_column_conflict_table(df_a: pd.DataFrame, df_b: pd.DataFrame, key_columns: list) -> list:
    """
    Builds the batch-review-style table data for overlapping non-key
    columns (spec's "Duplicate Column Handling") -- one row per
    overlapping COLUMN NAME, not per data value, mirroring the
    phone/email batch review table's shape but at the column level
    instead of the cell level. Each entry shows a sample value from
    each dataset side-by-side, so the client can see at a glance
    whether the two sides actually agree or genuinely conflict (e.g.
    "UK" vs "United Kingdom").

    Returns a list of dicts: column, sample_value_a, sample_value_b --
    consumed by app.py to render the editable decision table.
    """
    overlapping = find_overlapping_columns(df_a, df_b, key_columns)
    rows = []
    for col in overlapping:
        sample_a = df_a[col].dropna().iloc[0] if df_a[col].notna().any() else None
        sample_b = df_b[col].dropna().iloc[0] if df_b[col].notna().any() else None
        rows.append({
            "column": col,
            "sample_value_a": sample_a,
            "sample_value_b": sample_b,
        })
    return rows


if __name__ == "__main__":
    print("=== Row-wise append ===")
    file_a = pd.DataFrame({"Customer": ["John", "Jane"], "Product": ["Widget", "Gadget"], "Price": [10, 20]})
    file_b = pd.DataFrame({"Customer": ["Bob"], "Price": [15], "Discount": [0.1]})  # different order, missing Product, extra Discount

    print("File A:\n", file_a)
    print("File B:\n", file_b)

    comparison = compare_columns_for_append([file_a, file_b])
    print("\nComparison:", comparison)

    merged, log, _ = append_datasets([file_a, file_b])
    print("\nMerged:\n", merged)
    print("\nLog:")
    for line in log:
        print(" -", line)

    print("\n\n=== Column-wise join ===")
    customers = pd.DataFrame({
        "Customer_ID": [1, 2, 3, 3],  # 3 is a duplicate key on purpose
        "Name": ["Alice", "Bob", "Carol", "Carol"],
        "Country": ["UK", "USA", "UK", "UK"],
    })
    orders = pd.DataFrame({
        "Customer_ID": [1, 2, 4],
        "Total_Spend": [100, 250, 50],
        "Country": ["United Kingdom", "United States", "France"],
    })

    print("Customers:\n", customers)
    print("Orders:\n", orders)

    validation = validate_merge_keys(customers, orders, ["Customer_ID"])
    print("\nValidation:", validation)

    overlapping = find_overlapping_columns(customers, orders, ["Customer_ID"])
    print("Overlapping columns:", overlapping)

    merged, log, summary = join_datasets(
        customers, orders, ["Customer_ID"], how="left",
        column_conflict_choices={"Country": "prefer_non_missing"},
    )
    print("\nMerged (left join, Country: prefer_non_missing):\n", merged)
    print("\nMatch summary:", summary)
    print("\nLog:")
    for line in log:
        print(" -", line)

    print("\n\n=== Question generation ===")
    append_q = generate_append_confirmation_question(compare_columns_for_append([file_a, file_b]))
    print("Append confirmation question:", append_q)

    key_q = generate_key_column_question(customers, orders)
    print("\nKey column question:", key_q)

    join_q = generate_join_type_question()
    print("\nJoin type question:", join_q)

    conflict_table = build_column_conflict_table(customers, orders, ["Customer_ID"])
    print("\nConflict table:", conflict_table)
