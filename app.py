"""
app.py — Phase 2 Interactive Data Cleaning App (Streamlit)
=============================================================
The UI layer described in Phase2_InteractiveApp_Spec.md. This file
contains NO cleaning logic of its own -- every actual transformation
happens inside the Sprint 1 modules (01-09) and the Phase 2 logic
modules (10_column_profiling, 11_question_generation,
12_apply_answers_and_run). This file's only job is:

  upload -> preview + auto-detected profile -> dynamic clarifying
  questions -> advanced settings (collapsed) -> "Clean my data" button
  -> results + download

PRIVACY: the uploaded file is read directly from Streamlit's in-memory
upload buffer (never written to disk), and the cleaned output Excel
file is built in an in-memory BytesIO buffer for the download button
-- matching the project's "processed in-memory, not stored" design
principle from Sprint 1's spec.
"""

import io
import re
from importlib import import_module

import pandas as pd
import streamlit as st

ingestion = import_module("01_ingestion")
profiling = import_module("10_column_profiling")
qgen = import_module("11_question_generation")
runner = import_module("12_apply_answers_and_run")
report = import_module("09_cleaning_report")
ml_ready = import_module("14_ml_ready_encoding")
phone_conv = import_module("54_phone_conversion")
email_conv = import_module("55_email_conversion")
merge_mod = import_module("56_dataset_merge")
text_num = import_module("49_text_number_parsing")
measurement_conv = import_module("46_unit_standardisation_conversion")
currency_conv = import_module("48_currency_conversion")
duration_conv = import_module("51_duration_conversion")

ts_struct = import_module("15_time_series_structure_questions")
ts_split = import_module("16_time_series_split_validate")
ts_sort = import_module("17_time_series_sort")
ts_dup = import_module("18_time_series_duplicates")
ts_gaps = import_module("19_time_series_gaps_and_timestep")
ts_freq = import_module("20_time_series_frequency_aggregation")
ts_val = import_module("21_time_series_value_validation")
ts_diag = import_module("22_time_series_missing_diagnosis")
ts_out = import_module("23_time_series_ml_ready_outputs")
ts_report = import_module("24_time_series_processing_report")
ts_impq = import_module("25_time_series_imputation_questions")
ts_single_imp = import_module("26_time_series_single_imputation")
ts_backtest = import_module("27_time_series_backtesting")
ts_apply_imp = import_module("28_time_series_apply_imputation")
ts_multi_imp = import_module("29_time_series_multiple_imputation")
ts_multicol = import_module("31_time_series_multi_column_combination")
ts_cross_series = import_module("33_time_series_cross_series_imputation")
ts_contextual_features = import_module("34_time_series_contextual_features")
ts_fold_transforms = import_module("35_time_series_fold_transforms")
ts_contextual_models = import_module("36_time_series_contextual_models")
ts_neural_eligibility = import_module("37_time_series_neural_eligibility")
ts_neural = import_module("38_time_series_neural_imputation")
ts_mi_eval = import_module("39_time_series_mi_evaluation_metrics")
ts_gen_mi = import_module("40_time_series_generalized_mi_draws")
ts_mi_backtest = import_module("41_time_series_mi_backtest")
ts_bounds = import_module("42_time_series_plausibility_bounds")
ts_wrapper = import_module("43_time_series_standardized_wrapper")
ts_unified = import_module("44_time_series_unified_ranking")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded file directly from Streamlit's in-memory buffer.

    This mirrors Module A's load_data() logic (branch on extension,
    same supported types) but works on a file-like object instead of
    a filesystem path -- Module A itself is left unchanged, since it's
    also used standalone/in Colab where a path makes more sense.

    Deliberately does NOT catch/wrap read errors here -- extension
    validity and "can pandas actually parse this" are two genuinely
    different failure modes that deserve two different messages to the
    user, so the caller (below) checks the extension FIRST and handles
    read failures separately. Confirmed by testing this matters:
    pandas' ParserError is actually a subclass of ValueError, so a
    single shared except-ValueError would have silently swallowed a
    real parsing failure under the "wrong file type" message instead
    of the more useful "here's likely why parsing failed" guidance.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}. Expected .csv, .xls, or .xlsx")


def reset_derived_session_caches():
    """
    Clears every cache derived from a PREVIOUS dataset -- called
    whenever the app is about to start working on a genuinely
    different dataset, whether that's a new file upload OR switching
    from Dataset Merge into "clean this merged result".

    THE BUG THIS FIXES (confirmed by testing): the Dataset Merge
    "Use this merged dataset for cleaning" button used to clear only
    "widget_"-prefixed keys, written before several of the caches
    below existed at all. A cache built for one dataset (e.g.
    profile_table_base, keyed to a PREVIOUS file's column names) would
    silently carry over into the merged dataset's cleaning session --
    the moment the client edited the variable-type table, the code
    looked up a column against that stale, mismatched cache and threw
    an exception, which is what showed up as "everything got deleted,
    back to the upload screen."

    Every new per-dataset cache added to this app from now on should
    be cleared HERE, in this one function, rather than adding another
    scattered st.session_state.pop() call at each place a new dataset
    can enter the app -- that scattered pattern is exactly how this
    bug happened in the first place: it's easy to update the "new file
    upload" path when adding a cache and forget the merge path (or any
    future second entry point) needs the identical reset.
    """
    for key in list(st.session_state.keys()):
        if key.startswith("widget_"):
            del st.session_state[key]
    st.session_state.pop("profile_table_base", None)
    st.session_state.pop("phone_email_review_base", None)
    st.session_state.pop("phone_email_review_signature", None)
    st.session_state.pop("_correction_widget_registry", None)
    st.session_state.pop("value_corrections", None)


def build_profile_summary_table(profile: dict) -> pd.DataFrame:
    """Turn the profile dict into a flat table for display in the UI."""
    rows = []
    for col, info in profile.items():
        notes = []
        if info.get("disguised_missing_markers"):
            markers = ", ".join(info["disguised_missing_markers"].keys())
            notes.append(f"possible placeholder values: {markers}")
        if info.get("date_ambiguity") and info["date_ambiguity"]["needs_dayfirst_question"]:
            notes.append("contains ambiguous dates")
        rows.append({
            "Column": col,
            "Detected type": info["dtype"],
            "Likely role": info["semantic_role"],
            "Confidence": info["role_confidence"],
            # Kept SEPARATE from "Likely role" above, deliberately --
            # role answers "what does this column represent" (email,
            # country); this answers "how should its VALUES be treated
            # statistically" (a real measurement vs. a 2-category flag
            # vs. an unordered/ordered set of categories). A numeric
            # column can have role "numeric_measure" and STILL be a
            # binary flag (e.g. Sex stored as 1/2) -- this column is
            # what actually surfaces that, low-confidence suggestions
            # and all, so it can be reviewed rather than silently
            # trusted for something this consequential.
            "Suggested variable type": info.get("variable_type", ""),
            "Variable type confidence": info.get("variable_type_confidence", ""),
            "Missing %": info["missing_percent"],
            "Notes": "; ".join(notes) if notes else "",
        })
    return pd.DataFrame(rows)


STRUCTURAL_MISSINGNESS_KEYWORDS = {
    "bonus", "commission", "discount", "tip", "tips", "gratuity", "surcharge",
    "fee", "fees", "allowance", "overtime", "quantity", "qty", "refund",
    "deduction", "penalty", "coupon", "rebate",
}


def _looks_like_structural_missingness_column(column_name: str) -> bool:
    """
    Name-based heuristic for "this column's missing values might mean
    something specific (e.g. '$0'), not 'unknown'" -- reuses the exact
    examples the finalised spec names explicitly (Bonus, Commission,
    Discount, Quantity). Used to decide which columns get a prominent,
    individually-shown override control vs. which get quietly tucked
    into a secondary "show all" list.

    Word-boundary tokenisation, not naive substring search -- same
    discipline as Module I's _name_hint_score (confirmed earlier in
    this project as a real bug source, e.g. "id" matching inside
    "provide"): a column like "Quantity Description" or "Feedback"
    shouldn't false-positive just because a keyword is a substring.
    """
    normalised = column_name.lower().replace("_", " ").replace("-", " ")
    tokens = set(re.findall(r"[a-z0-9]+", normalised))
    return bool(tokens & STRUCTURAL_MISSINGNESS_KEYWORDS)


def _country_columns_with_values(profile: dict, df: pd.DataFrame, value_set: set) -> list:
    """
    Returns EVERY column Module I detected as role "country" that
    actually contains at least one value from the given set (e.g. UK
    constituent countries, or US state names) -- not just whether ANY
    column does.

    This used to be a single yes/no check (_country_column_has_values),
    which only ever produced ONE global checkbox covering every
    "country"-role column in the file. That's wrong whenever a file has
    more than one such column with genuinely different intents -- e.g.
    a real "What country do you work in?" column (where collapsing a
    stray US state into "United States" is a sensible data-entry-error
    fix) alongside a "What US state do you work in?" column (where the
    exact same collapse destroys the actual answer the question was
    asking for). Returning the full list lets the caller render one
    checkbox PER column instead of one for the whole file.
    """
    matching = []
    for col, info in profile.items():
        if info["semantic_role"] != "country":
            continue
        normalized = df[col].dropna().astype(str).apply(profiling._normalize_country_text)
        if normalized.isin(value_set).any():
            matching.append(col)
    return matching


def render_question(q: dict):
    """
    Render one question dict as the appropriate Streamlit widget and
    return the answer. Kept generic (driven entirely by the question
    dict's "type"/"options"/"default" fields) so Module J can add or
    change questions without this function needing to know about any
    specific question by name.
    """
    widget_key = f"widget_{q['id']}"

    if q["type"] in ("single_select", "yes_no"):
        label_to_value = {opt["label"]: opt["value"] for opt in q["options"]}
        value_to_label = {opt["value"]: opt["label"] for opt in q["options"]}
        option_labels = [opt["label"] for opt in q["options"]]
        default_label = value_to_label.get(q["default"]) if q["default"] is not None else None
        default_index = option_labels.index(default_label) if default_label in option_labels else None
        # The small (i) info icon the client asked for -- Streamlit's
        # widgets already have a built-in "help" parameter that renders
        # exactly this (a hoverable tooltip icon next to the label), so
        # no custom icon/popover needs building. Any question can use
        # this by setting "info_text" -- currently only Module J's
        # text-number confirmation does, but it's fully generic.
        help_text = q.get("info_text")

        if q.get("render_as_dropdown"):
            # Currently used only by Module J's per-value currency
            # resolution questions (generate_currency_value_
            # resolution_questions) -- a compact st.selectbox instead
            # of the st.radio used everywhere else, since there can be
            # MANY of these (one per distinct unresolved value) and a
            # full radio list of ~26 currencies repeated for each
            # would make the page unusably long.
            if q.get("example_row"):
                # "the clue can be in the cell" -- show the full row
                # this value came from, since another column can hint
                # at the right answer (e.g. a Country column saying
                # "UK" strongly suggesting GBP).
                st.caption("Example row this value came from:")
                st.dataframe(pd.DataFrame([q["example_row"]]), hide_index=True)
            selected_label = st.selectbox(
                q["question_text"], option_labels, index=default_index, key=widget_key, help=help_text
            )
            return label_to_value.get(selected_label) if selected_label is not None else None

        selected_label = st.radio(
            q["question_text"], option_labels, index=default_index, key=widget_key, help=help_text
        )
        return label_to_value.get(selected_label) if selected_label is not None else None

    elif q["type"] == "multi_select":
        value_to_label = {opt["value"]: opt["label"] for opt in q["options"]}
        option_values = [opt["value"] for opt in q["options"]]
        return st.multiselect(
            q["question_text"],
            options=option_values,
            default=q["default"] or [],
            format_func=lambda v: value_to_label.get(v, v),
            key=widget_key,
        )

    elif q["type"] == "numeric_input":
        # Used so far only by the currency exchange-rate questions
        # (Module J's generate_currency_rate_questions) -- a free
        # numeric entry, unlike every other question type above, which
        # all pick from a closed set of options. Deliberately has NO
        # sensible default (an exchange rate can't be guessed safely),
        # so the widget starts at 0.0 -- itself not a valid rate -- and
        # that's used as the "not actually answered yet" signal,
        # returning None until the client enters something greater
        # than zero. This reuses the SAME "missing_required" check
        # already in place below (q["default"] is None and answer is
        # None) without that check needing to know anything
        # numeric-input-specific.
        raw_value = st.number_input(
            q["question_text"],
            min_value=q.get("min_value", 0.0),
            value=0.0,
            step=0.01,
            format="%.6f",
            key=widget_key,
        )
        return raw_value if raw_value > 0 else None

    else:
        st.warning(f"Unrecognised question type '{q['type']}' for question '{q['id']}' — skipped.")
        return None


def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Automated Data Cleaning", layout="wide")

# ---------------------------------------------------------------------------
# Demo/presentation styling -- PURELY COSMETIC (CSS only, no Python
# logic touched at all, so this cannot affect behaviour in any way).
# Bigger overall text for readability in a recording, and section
# headers (every st.header() call -- e.g. "3a. Are there hidden
# missing-value markers?", "4. How should missing values be filled
# in?") made larger, bold, and colour-accented, so each new topic the
# client needs to look at is visually unmistakable at a glance rather
# than blending into a wall of text. Deliberately targets only stable,
# generic HTML tags (h1/h2/h3/p/button) and Streamlit's well-
# established widget classes (.stButton, .stAlert, etc.), NOT
# Streamlit's internal auto-generated class names (the "st-emotion-
# cache-xxxxx" style hashes), which change between Streamlit versions
# and would make this fragile.
st.markdown(
    """
    <style>
    html, body { font-size: 18px; }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
    }

    /* st.header() -- marks each major step/topic transition. Made
       large, bold, and accent-coloured with an underline, so moving
       from one topic (e.g. duplicates) to the next (e.g. missing
       values) is immediately obvious on screen. */
    h2 {
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: #0d6efd !important;
        border-bottom: 3px solid rgba(13, 110, 253, 0.25);
        padding-bottom: 0.3rem;
        margin-top: 1.6rem !important;
    }

    /* st.subheader() -- one clear step down from h2, still well above body text. */
    h3 {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #343a40 !important;
    }

    p, li, label { font-size: 1.05rem !important; }

    .stButton button, .stDownloadButton button {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
    }

    /* st.info / st.warning / st.success / st.error callouts -- the
       "pay attention to this" boxes used throughout (e.g. the phone/
       email review table, merge validation summaries). */
    .stAlert { font-size: 1.05rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Automated Data Cleaning")
st.caption(
    "Upload your messy Excel or CSV file and receive a professionally cleaned, standardised, and "
    "validated dataset — guided by a few quick questions along the way."
)

# BUILD MARKER -- bump this string every time a new app.py is handed
# over, so a stale server process (old code still running despite a
# fresh file upload/restart) is immediately visible at a glance,
# instead of being mistaken for a real regression in the logic.
#
# Hidden from the client-facing view by default (SHOW_BUILD_TAG =
# False) -- a raw build string/changelog fragment doesn't mean
# anything to someone watching a portfolio demo. Flip this to True
# during development/debugging whenever the stale-server check above
# is actually needed again; the mechanism itself is unchanged, just no
# longer shown by default.
SHOW_BUILD_TAG = False
APP_BUILD_TAG = "build 2026-07-23-c — One unified method now selected for the whole dataset (per client request): every column standardized before ranking so no single column's scale dominates the comparison, ONE ranking table shown, ONE method applied to every column. Works for both Single-Value and Multiple Imputation"
if SHOW_BUILD_TAG:
    st.caption(f"🔖 {APP_BUILD_TAG}")

# ---------------------------------------------------------------------------
# Dataset Merge -- optional, standalone mode (Module 56)
# ---------------------------------------------------------------------------
# Per the spec: "Do not modify the existing cleaning modules. This
# module should be optional and run only when the user selects Merge
# Datasets." Everything below, when merge mode is chosen, runs in its
# OWN self-contained branch and calls st.stop() at the end -- the
# existing single-file flow (Step 1 onwards) is completely untouched
# and simply never executes on a merge-mode rerun.

# ---------------------------------------------------------------------------
# Explore unique values per column (data exploration aid, no cleaning
# effect of its own -- purely for the client to browse/spot-check
# values before deciding how to answer the questions below)
# ---------------------------------------------------------------------------

_MAX_DROPDOWN_VALUES = 1000  # cap purely for Streamlit's own rendering performance on a very high-cardinality column


def _sorted_unique_values(series: pd.Series) -> list:
    """
    Unique, non-missing values, sorted for easy scanning. Falls back
    to sorting by string representation if the column has a genuine
    mix of types (e.g. numbers and text in the same column) -- plain
    sorted() raises TypeError on a mixed-type list, and this is
    exactly the kind of column where seeing the mess is most useful,
    so it shouldn't be the one case that crashes this feature.
    """
    values = series.dropna().unique().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _build_unique_values_excel(series: pd.Series, column_name: str) -> bytes:
    """
    Builds a small standalone Excel file listing every distinct
    non-missing value in `series`, with how many times each one
    occurs -- the count is a genuinely useful extra for spotting
    typos (e.g. "Scotland" appearing 15 times next to "Scotlant"
    appearing twice makes the typo obvious at a glance), even though
    the client only asked for the values themselves.

    Deliberately NOT capped at _MAX_DROPDOWN_VALUES -- that cap exists
    purely for the dropdown's own on-screen rendering performance, not
    a real limit on the data, so a download should still contain every
    distinct value regardless of how many there are.
    """
    counts = series.dropna().value_counts()
    try:
        counts = counts.sort_index()
    except TypeError:
        counts = counts.reindex(sorted(counts.index, key=str))

    export_df = pd.DataFrame({column_name: counts.index, "Count": counts.values})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, sheet_name="Unique Values", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def render_explore_unique_values(df: pd.DataFrame):
    """
    Renders the "Explore unique values per column" section -- one
    dropdown + correction textbox + download button per column -- and
    updates st.session_state.value_corrections from whatever's
    currently typed. Shared by BOTH the Standard and Basic Package
    flows (identical behaviour either way -- this is a data-
    understanding aid for whoever's running the cleaning, not a
    Standard-only feature, per explicit client request), so there's
    exactly one implementation to keep correct rather than two copies
    that could quietly drift out of sync.
    """
    with st.expander("Explore unique values per column", expanded=False):
        st.caption(
            "One dropdown per column, listing every distinct value it contains -- handy for spotting "
            "typos or inconsistent categories (e.g. 'UK' vs 'United Kingdom') before deciding how to "
            "answer the questions below. Browsing here doesn't change anything by itself -- but if you "
            "spot a genuine typo (e.g. 'Scotlant'), you can type the correct value in the box underneath "
            "and every matching cell in that column will be replaced with it in the cleaned output."
        )
        # Maps each correction text box's widget key back to the EXACT
        # (column, original_value) it corrects -- kept as an explicit
        # registry rather than parsing the information back out of the
        # key string itself, since a column name or value could contain
        # "__" (the delimiter a key-parsing approach would need) and
        # silently produce a wrong or broken match.
        st.session_state.setdefault("_correction_widget_registry", {})

        for col in df.columns:
            unique_values = _sorted_unique_values(df[col])
            n_unique = len(unique_values)
            n_missing = int(df[col].isna().sum())

            if n_unique == 0:
                st.write(f"**{col}** — no non-missing values to show ({n_missing} missing)")
                continue

            label = f"{col} — {n_unique} unique value(s)"
            if n_missing:
                label += f", {n_missing} missing"

            display_values = unique_values
            if n_unique > _MAX_DROPDOWN_VALUES:
                display_values = unique_values[:_MAX_DROPDOWN_VALUES]
                label += f" (showing first {_MAX_DROPDOWN_VALUES})"

            dropdown_key = f"widget_explore_unique__{col}"
            selected_value = st.selectbox(label, options=display_values, key=dropdown_key)

            # Keyed by (column, the SPECIFIC value currently selected) --
            # switching the dropdown to a different value shows a DIFFERENT
            # text box, but each one's own previously-typed correction (if
            # any) is remembered independently across reruns, via Streamlit's
            # normal session_state persistence for widget keys.
            correction_key = f"widget_value_correction__{col}__{display_values.index(selected_value)}"
            st.session_state["_correction_widget_registry"][correction_key] = (col, selected_value)
            st.text_input(
                f"Replace {selected_value!r} with (leave blank to keep as-is):",
                key=correction_key,
            )

            # Own distinct key, never touched by the correction-widget
            # registry above -- a download button is a plain one-off
            # action, not something whose typed state needs to persist or
            # be collected into `value_corrections`, so it's kept
            # completely separate from that mechanism.
            st.download_button(
                f"Download all '{col}' unique values (.xlsx)",
                data=_build_unique_values_excel(df[col], col),
                file_name=f"{col}_unique_values.xlsx",
                key=f"widget_download_unique_values__{col}",
            )

        # Collected fresh on every rerun from whatever's CURRENTLY in
        # session_state for each registered correction box -- this is what
        # actually reaches the cleaning pipeline (merged into `answers`
        # once it exists), not just a UI convenience.
        _value_corrections = {}
        for _widget_key, (_col, _original_value) in st.session_state["_correction_widget_registry"].items():
            _typed = st.session_state.get(_widget_key, "").strip()
            if _typed and _typed != _original_value:
                _value_corrections.setdefault(_col, {})[_original_value] = _typed
        st.session_state.value_corrections = _value_corrections

st.divider()
top_level_mode = st.radio(
    "What would you like to do?",
    [
        "Clean a single file",
        "Merge two datasets",
    ],
    key="widget_top_level_mode",
)

if top_level_mode == "Merge two datasets":
    if "merge_result" not in st.session_state:
        st.session_state.merge_result = None  # (df, log, type, details) once a merge has run

    st.header("Merge Two Datasets")
    st.caption(
        "Combine two files either by stacking rows (row-wise append -- datasets holding the "
        "SAME kind of record, e.g. two months of the same export) or by matching columns "
        "(column-wise join -- datasets describing the SAME entities with different information, "
        "e.g. customer details in one file and order totals in another)."
    )

    upload_col_a, upload_col_b = st.columns(2)
    with upload_col_a:
        merge_file_a = st.file_uploader("Dataset A", type=["csv", "xls", "xlsx"], key="widget_merge_file_a")
    with upload_col_b:
        merge_file_b = st.file_uploader("Dataset B", type=["csv", "xls", "xlsx"], key="widget_merge_file_b")

    if merge_file_a is None or merge_file_b is None:
        st.info("Upload both datasets to continue.")
        st.stop()

    try:
        merge_df_a = load_uploaded_file(merge_file_a)
        merge_df_b = load_uploaded_file(merge_file_b)
    except Exception as e:
        st.error(f"Couldn't read one of the files: {e}")
        st.stop()

    st.write(f"**Dataset A** ({merge_file_a.name}): {len(merge_df_a)} rows, {len(merge_df_a.columns)} columns")
    st.dataframe(merge_df_a.head(5), width="stretch")
    st.write(f"**Dataset B** ({merge_file_b.name}): {len(merge_df_b)} rows, {len(merge_df_b.columns)} columns")
    st.dataframe(merge_df_b.head(5), width="stretch")

    st.subheader("Merge type")
    merge_mode_choice = st.radio(
        "How should these datasets be merged?",
        [
            "Row-wise append -- stack matching rows underneath each other",
            "Column-wise join -- match rows on a shared key and combine their columns",
        ],
        key="widget_merge_type_choice",
    )

    merge_details = {}

    # -----------------------------------------------------------------
    # Row-wise append
    # -----------------------------------------------------------------
    if merge_mode_choice.startswith("Row-wise"):
        comparison = merge_mod.compare_columns_for_append([merge_df_a, merge_df_b])
        append_question = merge_mod.generate_append_confirmation_question(comparison)

        proceed_with_append = True
        if append_question:
            st.warning(append_question["question_text"])
            append_choice_label = st.radio(
                "Continue?",
                [opt["label"] for opt in append_question["options"]],
                key="widget_merge_append_confirm",
            )
            proceed_with_append = (append_choice_label == "Yes, continue with the append")
        else:
            st.success("Both datasets have identical columns -- nothing to confirm.")

        if proceed_with_append and st.button("Run row-wise append", key="widget_run_append"):
            merged_result_df, merge_log, comparison = merge_mod.append_datasets([merge_df_a, merge_df_b])
            merge_details = {"Merge Type": "Row-wise (append)"}
            st.session_state.merge_result = (merged_result_df, merge_log, "row-wise", merge_details)

    # -----------------------------------------------------------------
    # Column-wise join
    # -----------------------------------------------------------------
    else:
        key_question = merge_mod.generate_key_column_question(merge_df_a, merge_df_b)
        if not key_question["options"]:
            st.error(
                "These two datasets have no column names in common at all -- a column-wise join "
                "needs at least one shared column to match rows on."
            )
            st.stop()

        selected_keys = st.multiselect(
            key_question["question_text"],
            options=[opt["value"] for opt in key_question["options"]],
            key="widget_merge_keys",
        )

        if not selected_keys:
            st.info("Select at least one key column to continue.")
            st.stop()

        join_question = merge_mod.generate_join_type_question()
        join_labels = [opt["label"] for opt in join_question["options"]]
        join_values = [opt["value"] for opt in join_question["options"]]
        join_choice_label = st.radio(
            join_question["question_text"], join_labels,
            index=join_values.index(join_question["default"]),
            key="widget_merge_join_type",
        )
        selected_join = join_values[join_labels.index(join_choice_label)]

        validation = merge_mod.validate_merge_keys(merge_df_a, merge_df_b, selected_keys)
        if validation["missing_key_columns_a"] or validation["missing_key_columns_b"]:
            st.error(
                f"Key column(s) not found -- missing from Dataset A: "
                f"{validation['missing_key_columns_a'] or 'none'}; missing from Dataset B: "
                f"{validation['missing_key_columns_b'] or 'none'}."
            )
            st.stop()

        st.write("**Pre-merge validation:**")
        st.write(f"- Matching key value(s): {validation['matching_key_count']}")
        st.write(f"- Only in Dataset A: {validation['only_in_a_count']}")
        st.write(f"- Only in Dataset B: {validation['only_in_b_count']}")
        st.write(
            f"- Duplicate keys -- Dataset A: {validation['duplicate_key_count_a']}, "
            f"Dataset B: {validation['duplicate_key_count_b']}"
        )
        st.write(
            f"- Rows with a missing key value -- Dataset A: {validation['null_key_count_a']}, "
            f"Dataset B: {validation['null_key_count_b']}"
        )

        conflict_rows = merge_mod.build_column_conflict_table(merge_df_a, merge_df_b, selected_keys)
        conflict_choices = {}

        if conflict_rows:
            st.write(
                "**Both datasets have these columns in common (besides the key) -- how should "
                "conflicts be resolved?**"
            )
            # Same "don't rebuild the source on every rerun" fix already
            # applied to the profile-type editor and the phone/email
            # review table earlier in this file -- cached once per
            # distinct key-column choice, not rebuilt from
            # conflict_rows every rerun, or st.data_editor's edit-
            # tracking gets confused by a changing source under the
            # same widget key.
            conflict_signature = tuple(selected_keys)
            if st.session_state.get("merge_conflict_signature") != conflict_signature:
                conflict_table_df = pd.DataFrame([
                    {
                        "Column": r["column"],
                        "Sample from A": r["sample_value_a"],
                        "Sample from B": r["sample_value_b"],
                        "Decision": "Keep both columns",
                    }
                    for r in conflict_rows
                ])
                st.session_state.merge_conflict_base = conflict_table_df
                st.session_state.merge_conflict_signature = conflict_signature

            edited_conflicts = st.data_editor(
                st.session_state.merge_conflict_base,
                key="widget_merge_conflict_editor",
                hide_index=True,
                width="stretch",
                disabled=["Column", "Sample from A", "Sample from B"],
                column_config={
                    "Decision": st.column_config.SelectboxColumn(
                        options=[
                            "Keep both columns", "Keep Dataset A's value",
                            "Keep Dataset B's value", "Prefer non-missing value",
                        ],
                        required=True,
                    ),
                },
            )
            _CONFLICT_LABEL_TO_CHOICE = {
                "Keep both columns": "keep_both",
                "Keep Dataset A's value": "keep_a",
                "Keep Dataset B's value": "keep_b",
                "Prefer non-missing value": "prefer_non_missing",
            }
            for _, row in edited_conflicts.iterrows():
                conflict_choices[row["Column"]] = _CONFLICT_LABEL_TO_CHOICE[row["Decision"]]

        if st.button("Run column-wise join", key="widget_run_join"):
            merged_result_df, merge_log, match_summary = merge_mod.join_datasets(
                merge_df_a, merge_df_b, selected_keys, selected_join, conflict_choices
            )
            merge_details = {
                "Merge Type": "Column-wise (join)",
                "Merge Keys": ", ".join(selected_keys),
                "Join Type": merge_mod.JOIN_TYPE_DISPLAY[selected_join].split(" -- ")[0],
                "Matching Records": match_summary["matching_records"],
                "Rows only in Dataset A": match_summary["only_in_a"],
                "Rows only in Dataset B": match_summary["only_in_b"],
                "Duplicate Keys (A)": int(validation["duplicate_key_count_a"]),
                "Duplicate Keys (B)": int(validation["duplicate_key_count_b"]),
            }
            st.session_state.merge_result = (merged_result_df, merge_log, "column-wise", merge_details)

    # -----------------------------------------------------------------
    # Results, common to both modes
    # -----------------------------------------------------------------
    if st.session_state.merge_result:
        result_df, result_log, result_type, result_details = st.session_state.merge_result

        st.subheader("Merge Summary")
        st.success(f"Merge complete ({result_type}): {len(result_df)} rows, {len(result_df.columns)} columns")
        if result_details:
            st.table(pd.DataFrame(list(result_details.items()), columns=["Detail", "Value"]))

        st.write("**Audit trail (also included in the downloaded report):**")
        for line in result_log:
            st.write(f"- {line}")

        st.dataframe(result_df.head(20), width="stretch")

        if result_type == "row-wise":
            run_dedup_after_append = st.checkbox(
                "Run duplicate detection on the merged result",
                value=False, key="widget_merge_run_dedup",
            )
            if run_dedup_after_append:
                dedup_mod = import_module("03_duplicate_removal")
                rows_before_dedup = len(result_df)
                result_df = dedup_mod.remove_exact_duplicates(result_df)
                removed_count = rows_before_dedup - len(result_df)
                st.write(f"Removed {removed_count} exact duplicate row(s) after appending.")

        merged_excel_buffer = io.BytesIO()
        with pd.ExcelWriter(merged_excel_buffer, engine="xlsxwriter") as writer:
            result_df.to_excel(writer, sheet_name="Merged Data", index=False)
            summary_sheet_rows = list(result_details.items()) + [("", "")] + [("Actions", line) for line in result_log]
            pd.DataFrame(summary_sheet_rows, columns=["Detail", "Value"]).to_excel(
                writer, sheet_name="Merge Summary", index=False
            )
        merged_excel_buffer.seek(0)

        st.download_button(
            "Download merged file",
            data=merged_excel_buffer,
            file_name="merged_dataset.xlsx",
            key="widget_download_merged",
        )
        st.caption(
            "To clean this merged file, go back to \"Clean a single file\" above and upload the "
            "downloaded file -- merging and cleaning are separate steps."
        )

    st.stop()

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.profile = None
    st.session_state.questions = None
    st.session_state.cleaned_result = None  # (cleaned_df, excel_bytes) once pipeline has run
    st.session_state.mi_datasets_workbook = None  # (workbook_bytes, count, first_dataset_preview_df) once M-datasets pipeline has run
    st.session_state.ts_result = None  # (excel_bytes, processing_report_text) once Time Series pipeline has run
    st.session_state.ts_mi_datasets_workbook = None  # (workbook_bytes, count) once TS Multiple Imputation M-datasets mode has run


# ---------------------------------------------------------------------------
# Step 1: Upload
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload your file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None and (
    st.session_state.df is None or st.session_state.get("uploaded_filename") != uploaded_file.name
):
    file_extension_ok = uploaded_file.name.lower().endswith((".csv", ".xls", ".xlsx"))
    if not file_extension_ok:
        st.error(f"Unsupported file type: {uploaded_file.name}. Please upload a .csv, .xls, or .xlsx file.")
        st.stop()

    try:
        df = load_uploaded_file(uploaded_file)
    except Exception as e:
        # The extension is already confirmed valid at this point, so
        # anything raised here is a genuine PARSING problem, not a
        # wrong-file-type problem -- e.g. a CSV with descriptive/
        # license text before the real header row (confirmed as a
        # real case by testing against a public dataset), inconsistent
        # column counts, or a bad encoding. Without this catch, the
        # user would see a raw Python traceback instead of a clear,
        # actionable message.
        st.error(
            f"Couldn't read this file: {e}\n\n"
            f"This usually means the file has extra text (like a title, "
            f"description, or license notice) before the actual data "
            f"table starts. Try removing any lines above the real column "
            f"headers and re-uploading."
        )
        st.stop()

    # Clear stale question-widget state (and every other per-dataset
    # cache) left over from a PREVIOUSLY uploaded file in this same
    # session. Streamlit remembers a widget's value by its `key`
    # across reruns and silently IGNORES the `default=` argument once
    # that key has been used before -- without this, a new file whose
    # questions happen to reuse an id from an earlier file (e.g. two
    # different files both having a column called "Item") would
    # inherit the OLD file's answer instead of the new file's actual
    # default. Confirmed as a real bug by testing multiple files in
    # one session. See reset_derived_session_caches() for the full
    # list of what this covers and why it's centralised in one place.
    reset_derived_session_caches()

    with st.spinner("Analysing your file..."):
        profile = profiling.profile_dataframe(df)
        questions = qgen.generate_questions(profile, df)

    st.session_state.df = df
    st.session_state.profile = profile
    st.session_state.questions = questions
    st.session_state.uploaded_filename = uploaded_file.name
    st.session_state.cleaned_result = None  # clear any previous run's results
    st.session_state.mi_datasets_workbook = None
    st.session_state.ts_result = None
    st.session_state.ts_mi_datasets_workbook = None

if st.session_state.df is None:
    st.info("Upload a .csv, .xls, or .xlsx file to get started.")
    st.stop()

df = st.session_state.df
profile = st.session_state.profile
questions = st.session_state.questions

st.button("Start over with a different file", on_click=reset_session)


# ---------------------------------------------------------------------------
# Step 2: Preview + auto-detected profile
# ---------------------------------------------------------------------------

st.header("1. Preview & detected columns")
st.write(f"**{len(df)} rows, {len(df.columns)} columns**")
st.dataframe(df.head(10), width="stretch")

VARIABLE_TYPE_OPTIONS = ["continuous", "binary", "categorical", "ordinal", "integer"]

with st.expander("Show detected column profile", expanded=False):
    # Built ONCE per file (stored in session state), not rebuilt from
    # `profile` on every rerun -- confirmed as the real cause of a
    # genuine "first edit reverts, second sticks" bug: `profile` gets
    # mutated (below) the moment an edit is detected, so rebuilding
    # the editor's SOURCE data from it every rerun fed a changing
    # target back into st.data_editor under the same key, which
    # confuses its internal edit-tracking. Keeping the source frozen
    # while the editor's own widget state accumulates edits naturally
    # is the standard fix for this class of Streamlit bug.
    if "profile_table_base" not in st.session_state:
        st.session_state.profile_table_base = build_profile_summary_table(profile)
    profile_table = st.session_state.profile_table_base

    edited_profile_table = st.data_editor(
        profile_table,
        width="stretch",
        key="widget_variable_type_editor",  # "widget_" prefix -> auto-cleared on new file upload, same as every other widget here
        disabled=[c for c in profile_table.columns if c != "Suggested variable type"],
        column_config={
            "Suggested variable type": st.column_config.SelectboxColumn(
                options=VARIABLE_TYPE_OPTIONS,
                required=True,
                help=(
                    "Auto-detected, but not always right for a numeric-looking column that's "
                    "actually a code (e.g. Sex stored as 1/2) -- change it here if needed. "
                    "continuous = a real measurement; binary = exactly 2 categories; "
                    "categorical = 3+ unordered categories; ordinal = 3+ categories with a "
                    "genuine order (e.g. a severity scale); integer = a count that can never be "
                    "fractional (e.g. number of passengers) -- forces whole-number rounding after "
                    "imputation even if a few raw values had stray decimals."
                ),
            ),
        },
    )
    # Only columns where the user actually changed the suggestion --
    # kept separate from the auto-detected profile so downstream steps
    # (the opt-in Machine Learning Ready section) can tell "the user
    # reviewed and corrected this" apart from "never reviewed".
    # Defensive lookup (dict, not .loc[]) -- if profile_table_base
    # somehow doesn't contain a given column at all (a stale cache
    # from a previous dataset slipping through despite
    # reset_derived_session_caches()), this skips that row silently
    # rather than raising a KeyError that would take down the whole
    # page. Belt-and-braces alongside the real fix above.
    _base_types_by_column = dict(zip(profile_table["Column"], profile_table["Suggested variable type"]))
    variable_type_overrides = {
        row["Column"]: row["Suggested variable type"]
        for _, row in edited_profile_table.iterrows()
        if row["Column"] in _base_types_by_column
        and row["Suggested variable type"] != _base_types_by_column[row["Column"]]
    }
    # Applied directly onto `profile` (in place -- profile is the same
    # dict object living in st.session_state, so this persists across
    # reruns) so EVERY downstream step -- the imputation-method
    # dropdowns below, question generation, and the actual cleaning
    # pipeline -- sees the user's correction rather than the raw
    # auto-detected guess. Without this, a column the user corrected
    # from "continuous" to "binary" here would still silently be
    # treated as numeric everywhere else in the app.
    for col, new_type in variable_type_overrides.items():
        if col in profile:
            profile[col]["variable_type"] = new_type



render_explore_unique_values(df)


st.header("2. What do you want to do with this data?")
output_mode_label = st.radio(
    "Output goal",
    [
        "General Cleaning — Basic Package (essentials only)",
        "General Cleaning — Standard Package (recommended default)",
        "Machine Learning Ready — Cross-Sectional Data",
        "Machine Learning Ready — Time Series",
    ],
    index=1,
    help=(
        "General Cleaning — Basic Package: the essentials only -- basic text clean-up, duplicate "
        "detection and removal, missing value treatment, and outlier detection (flagging only, "
        "values are never modified). "
        "General Cleaning — Standard Package: everything Basic does, PLUS unit/currency/duration "
        "standardisation, phone/email validation, column splitting (name/address/date), and dataset "
        "merging -- a normal, human-readable Excel file either way. "
        "Machine Learning Ready — Cross-Sectional Data: does everything Standard Package does, AND "
        "additionally recodes binary columns to 0/1, one-hot encodes multi-category columns into new 0/1 "
        "columns per category, and rounds ordinal columns to a valid observed level after "
        "imputation — ready to feed directly into a model. Review the 'Suggested variable type' "
        "column above first, since this behaviour is driven entirely by it. Built for cross-sectional "
        "data (one independent row per subject). "
        "Machine Learning Ready — Time Series: for time-stamped, sequential data (one series over "
        "time) instead of independent rows -- its own dedicated pipeline further down."
    ),
)
basic_package_mode = output_mode_label.startswith("General Cleaning — Basic")
prepare_for_ml = output_mode_label.startswith("Machine Learning Ready — Cross")
time_series_mode = output_mode_label.startswith("Machine Learning Ready — Time")

if basic_package_mode:
    st.caption(
        "Basic Package covers the essentials: basic text clean-up (casing and whitespace), "
        "duplicate detection and removal, missing value treatment, outlier detection (flagging "
        "only, values are never modified), and a cleaning report. For unit/currency/duration "
        "standardisation, phone/email validation, and column splitting, choose General Cleaning — "
        "Standard Package above instead."
    )

    if "basic_result" not in st.session_state:
        st.session_state.basic_result = None  # (cleaned_df, excel_bytes) once the Basic pipeline has run

    st.subheader("A few quick questions")
    basic_questions = qgen.generate_basic_mode_questions(profile, df)
    basic_answers = {"value_corrections": st.session_state.get("value_corrections", {})}
    if not basic_questions:
        st.write("Nothing ambiguous detected in this file — no questions needed.")
    for q in basic_questions:
        answer = render_question(q)
        basic_answers[q["id"]] = answer

    if st.button("Run Basic Cleaning", key="widget_run_basic_cleaning"):
        basic_output_buffer = io.BytesIO()
        with st.spinner("Cleaning..."):
            basic_cleaned_df = runner.run_basic_cleaning_pipeline(
                df.copy(), profile, basic_answers, basic_output_buffer
            )
        basic_output_buffer.seek(0)
        st.session_state.basic_result = (basic_cleaned_df, basic_output_buffer.getvalue())

    if st.session_state.basic_result:
        basic_result_df, basic_excel_bytes = st.session_state.basic_result
        st.subheader("Result")
        st.success(f"Cleaning complete: {len(basic_result_df)} rows, {len(basic_result_df.columns)} columns")
        st.dataframe(basic_result_df.head(20), width="stretch")
        st.download_button(
            "Download cleaned file",
            data=basic_excel_bytes,
            file_name="basic_cleaned_data.xlsx",
            key="widget_download_basic_cleaned",
        )

    st.stop()

if time_series_mode:
    st.header("2a. Time Series Structure")
    ts_placeholder = "— Select an answer —"

    # --- Stage 1: time column(s) -- supports selecting several jointly-
    # forming-the-timestamp columns (e.g. separate Date + Time), not
    # just one. Defaults to just the single best-scoring candidate,
    # since most files still have exactly one time column.
    ts_candidates = ts_struct.detect_time_column_candidates(df)
    ts_q_time = ts_struct.generate_time_columns_question(df, ts_candidates)
    ts_time_columns = st.multiselect(
        ts_q_time["text"], ts_q_time["options"], default=ts_q_time["default"], key="ts_time_columns"
    )
    if not ts_time_columns:
        st.info("Select at least one time column above to continue.")
        st.stop()

    # --- Stage 2: value column(s) -- excludes EVERY selected time column ---
    ts_q_value = ts_struct.generate_value_columns_question(df, ts_time_columns)
    ts_value_columns = st.multiselect(
        ts_q_value["text"], ts_q_value["options"], default=ts_q_value["default"], key="ts_value_columns"
    )
    if not ts_value_columns:
        st.info("Select at least one value column above to continue.")
        st.stop()

    # --- Stages 3+4: separator + component meaning(s), looped once PER
    # selected time column -- each column gets its own independent
    # question set (a Date column's separator has nothing to do with a
    # Time column's), collected into one config list Module AE/31 then
    # merges. Widget keys are disambiguated by column name so multiple
    # columns' questions never collide.
    ts_time_column_configs = []
    ts_missing_any_answer = False
    for ts_col in ts_time_columns:
        if len(ts_time_columns) > 1:
            st.markdown(f"**Time column: '{ts_col}'**")

        ts_col_sample_values = ts_struct.preview_time_values(df[ts_col])
        ts_col_detected_sep = ts_struct.detect_component_separator(df[ts_col])
        ts_q_sep = ts_struct.generate_separator_question(ts_col, ts_col_sample_values, ts_col_detected_sep)
        ts_sep_label = st.selectbox(
            ts_q_sep["text"], ts_q_sep["options"], index=ts_q_sep["options"].index(ts_q_sep["default"]),
            key=f"ts_separator__{ts_col}",
        )
        ts_col_separator = None if ts_sep_label == "Other / no separator (single value)" else ts_sep_label

        if ts_col_separator:
            ts_n_components = max(len(v.split(ts_col_separator)) for v in ts_col_sample_values)
            ts_component_questions = ts_struct.generate_component_meaning_questions(
                ts_col, ts_col_sample_values, ts_col_separator, ts_n_components
            )
            ts_col_meanings = []
            for cq in ts_component_questions:
                choice = st.selectbox(
                    cq["text"], [ts_placeholder] + cq["options"], index=0, key=f"{cq['id']}__{ts_col}"
                )
                if choice == ts_placeholder:
                    ts_missing_any_answer = True
                else:
                    ts_col_meanings.append(choice)
        else:
            ts_single_q = ts_struct.generate_single_element_meaning_question(ts_col, ts_col_sample_values)
            ts_single_choice = st.selectbox(
                ts_single_q["text"], [ts_placeholder] + ts_single_q["options"], index=0,
                key=f"{ts_single_q['id']}__{ts_col}",
            )
            if ts_single_choice == ts_placeholder:
                ts_missing_any_answer = True
                ts_col_meanings = []
            else:
                ts_col_meanings = [ts_single_choice]

        ts_time_column_configs.append(
            {"column": ts_col, "separator": ts_col_separator, "component_meanings": ts_col_meanings}
        )

    if ts_missing_any_answer:
        st.warning("Please answer every question above before continuing.")
        st.stop()

    # Derived, not asked separately: the component-meaning question(s)
    # above already distinguish 'Year (real calendar year)' from 'Year
    # (relative period...)' explicitly -- across ALL configured columns.
    ts_all_meanings = [m for config in ts_time_column_configs for m in config["component_meanings"]]
    ts_calendar_type = (
        "Real calendar dates"
        if "Year (real calendar year)" in ts_all_meanings
        else "Relative time periods (not real calendar years)"
    )
    # Kept for the processing report's display text -- a simple joined
    # summary across however many columns/separators are configured,
    # rather than a single separator that no longer applies once
    # there's more than one time column.
    ts_separator_summary = "; ".join(
        f"'{c['column']}': {c['separator'] if c['separator'] else 'single value'}" for c in ts_time_column_configs
    )

    # --- Combine & validate across all configured time columns (Module AE / 31) ---
    try:
        ts_split_df, ts_split_report = ts_multicol.combine_multiple_time_columns(df, ts_time_column_configs)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if ts_split_report["n_invalid_rows"]:
        with st.expander(
            f"⚠️ {ts_split_report['n_invalid_rows']} row(s) have invalid time values — click to review",
            expanded=True,
        ):
            st.dataframe(ts_split_report["invalid_rows"], width="stretch")
            st.caption(
                "These rows stay in the file but are excluded from the time-based steps below. "
                "Nothing was silently corrected or dropped -- fix them at the source and re-upload, "
                "or continue and address them afterward."
            )

    # --- Sort (Module Q / 17) ---
    try:
        ts_sort_columns = ts_sort.get_temporal_sort_columns(ts_split_report["component_columns"])
    except ValueError as e:
        st.error(str(e))
        st.stop()

    ts_sorted_df, ts_sort_report = ts_sort.sort_time_series(ts_split_df, ts_split_report["component_columns"])
    if not ts_sort_report["was_already_sorted"]:
        st.info(f"ℹ️ {ts_sort_report['n_rows_reordered']} row(s) weren't in chronological order and have been sorted.")

    # --- Duplicate periods (Module R / 18) -- only asked if any exist ---
    ts_dup_detection = ts_dup.detect_duplicate_periods(ts_sorted_df, ts_sort_columns)
    if ts_dup_detection["n_duplicate_periods"] > 0:
        st.warning(
            f"⚠️ Found {ts_dup_detection['n_duplicate_periods']} time period(s) with more than one "
            f"observation ({ts_dup_detection['n_rows_involved']} row(s) total)."
        )
        with st.expander("Show duplicate periods", expanded=False):
            st.json(ts_dup_detection["duplicate_details"])
        ts_dup_method_map = {
            "Keep first observation": "keep_first",
            "Keep last observation": "keep_last",
            "Sum": "sum",
            "Mean": "mean",
            "Median": "median",
            "Minimum": "min",
            "Maximum": "max",
            "Review manually (keep all, flagged)": "manual",
            "Stop processing": "stop",
        }
        ts_dup_method_label = st.selectbox(
            "How should duplicate periods be resolved? (sum vs. mean vs. keeping one depends on what "
            "the duplicate rows actually represent -- see the details above)",
            [ts_placeholder] + list(ts_dup_method_map.keys()),
            index=0,
            key="ts_duplicate_method",
        )
        if ts_dup_method_label == ts_placeholder:
            st.warning("Please choose how to resolve the duplicate periods above before continuing.")
            st.stop()
        try:
            ts_resolved_df, ts_dup_report = ts_dup.resolve_duplicate_periods(
                ts_sorted_df, ts_sort_columns, ts_value_columns, ts_dup_method_map[ts_dup_method_label]
            )
        except ts_dup.StopProcessingRequested as e:
            st.error(str(e))
            st.stop()
    else:
        ts_resolved_df = ts_sorted_df.copy()
        ts_resolved_df["_period_duplicate_count"] = 1
        ts_dup_report = {**ts_dup_detection, "method_applied": "n/a — none found", "n_groups_resolved": 0}

    # --- Missing time steps + canonical Time_Step (Module S / 19) ---
    try:
        ts_complete_df, ts_gap_report = ts_gaps.build_complete_sequence(
            ts_resolved_df, ts_split_report["component_columns"], ts_time_columns, ts_value_columns
        )
    except ts_gaps.UnsupportedTemporalStructure as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        # Covers the "every row failed time-component validation"
        # case -- e.g. a component meaning that doesn't actually match
        # the data (month names instead of numbers, since month-name
        # mapping isn't built yet). A real, honest dead end rather
        # than something to guess past, so surfaced clearly here
        # instead of crashing the whole app.
        st.error(str(e))
        st.stop()

    ts_freq_info = ts_gaps.detect_original_frequency(ts_split_report["component_columns"], ts_split_df)
    if ts_gap_report["n_missing_periods_inserted"] > 0:
        st.info(
            f"ℹ️ {ts_gap_report['n_missing_periods_inserted']} time period(s) were entirely absent "
            f"from the file and have been inserted (values left blank, never estimated at this stage)."
        )

    # --- Output frequency (Module T / 20) ---
    st.header("2b. Output Frequency")
    ts_q_freq = ts_freq.generate_output_frequency_question(ts_freq_info)
    ts_multiple = int(
        st.number_input(
            ts_q_freq["text"],
            min_value=ts_q_freq["min_value"],
            step=1,
            value=ts_q_freq["default"],
            key="ts_output_k",
        )
    )
    if ts_multiple > 1:
        st.caption(f"That's approximately {ts_freq.describe_output_step_length(ts_freq_info, ts_multiple)} per output time step.")

    # One shared aggregation method for now (this dataset is univariate) --
    # per-value-column methods become relevant once Multivariate mode exists.
    ts_agg_method = "sum"
    if ts_multiple > 1:
        ts_agg_label = st.selectbox(
            f"Aggregating {', '.join(ts_value_columns)} to a coarser frequency -- how should values "
            f"within each new period be combined? (sum vs. mean give genuinely different numbers, so "
            f"pick deliberately.)",
            [ts_placeholder] + ts_freq.AGGREGATION_METHODS,
            index=0,
            key="ts_aggregation_method",
        )
        if ts_agg_label == ts_placeholder:
            st.warning("Please choose an aggregation method above before continuing.")
            st.stop()
        ts_agg_method = ts_agg_label

    # --- Missing-value placeholders per value column (Module U / 21) ---
    st.header("2c. Value Column Validation")
    ts_confirmed_placeholders = {}
    for vc in ts_value_columns:
        ts_q_ph = ts_val.generate_missing_placeholder_question(ts_complete_df[vc], vc)
        if ts_q_ph is None:
            continue
        ts_selected_labels = st.multiselect(
            ts_q_ph["text"], ts_q_ph["options"], default=ts_q_ph["options"], key=ts_q_ph["id"]
        )
        selected_positions = [ts_q_ph["options"].index(s) for s in ts_selected_labels]
        ts_confirmed_placeholders[vc] = [ts_q_ph["option_values"][i] for i in selected_positions]

    ts_after_placeholders_df, ts_placeholder_log = ts_val.apply_confirmed_placeholders(
        ts_complete_df, ts_confirmed_placeholders
    )

    # --- Remaining non-numeric values, one question per unique value (Module U / 21) ---
    ts_remaining_non_numeric = ts_val.detect_remaining_non_numeric_values(ts_after_placeholders_df, ts_value_columns)
    ts_action_map = {
        "Convert to a specific number": "convert",
        "Classify as missing": "missing",
        "Remove the affected row(s)": "remove_row",
        "Leave column non-numeric and exclude it from cleaning": "exclude_column",
        "I'll correct the source data manually (leave as-is for now)": "manual",
    }
    ts_resolutions = {}
    ts_missing_resolution = False
    for col, entries in ts_remaining_non_numeric.items():
        for e in entries:
            rq = ts_val.generate_non_numeric_resolution_question(col, e["value"], e["count"], e["affected_time_steps"])
            action_label = st.selectbox(rq["text"], [ts_placeholder] + rq["options"], index=0, key=rq["id"])
            if action_label == ts_placeholder:
                ts_missing_resolution = True
                continue
            action = ts_action_map[action_label]
            entry = {"action": action}
            if action == "convert":
                entry["replacement_number"] = st.number_input(
                    f"Replace '{e['value']}' with:", key=f"{rq['id']}__number", value=0.0
                )
            ts_resolutions[(col, e["value"])] = entry

    if ts_missing_resolution:
        st.warning("Please resolve every non-numeric value flagged above before continuing.")
        st.stop()

    ts_resolved_values_df, ts_excluded_cols, ts_manual_cols, ts_non_numeric_log = ts_val.apply_non_numeric_resolutions(
        ts_after_placeholders_df, ts_resolutions
    )
    ts_final_values_df, ts_finalize_report = ts_val.finalize_numeric_conversion(
        ts_resolved_values_df, ts_value_columns, ts_excluded_cols, ts_manual_cols
    )
    if ts_excluded_cols:
        st.info(f"Excluded from cleaning entirely, per your choice above: {', '.join(ts_excluded_cols)}")
    if ts_manual_cols:
        st.info(f"Left for manual correction, per your choice above: {', '.join(ts_manual_cols)}")

    # --- Missing-value diagnosis -- THE gate (Module V / 22) ---
    st.header("2d. Missing-Value Diagnosis")
    ts_active_value_columns = [c for c in ts_value_columns if c not in ts_excluded_cols and c not in ts_manual_cols]
    ts_diagnosis = ts_diag.diagnose_missing_values(ts_final_values_df, ts_active_value_columns)
    ts_needs_imputation = ts_diag.columns_needing_imputation(ts_diagnosis)

    st.text(ts_diag.format_diagnosis_report(ts_diagnosis))

    ts_imputed_df = ts_final_values_df.copy()
    ts_mi_datasets_this_run = None  # populated only if a column uses Multiple Imputation's M-datasets/Both mode
    # BUG FIXED after client report: build_processing_report() was
    # being called with NEITHER imputation_summary= NOR
    # plausibility_summary= at all -- both are optional parameters
    # that silently default to None, and the report unconditionally
    # falls back to "Not applicable" whenever they're None, REGARDLESS
    # of whether imputation actually ran. The per-column loop below
    # already computes real results (ts_imp_report / ts_mi_result) and
    # shows them as one-off st.success() messages, but never collected
    # them anywhere the final report could use -- confirmed by testing:
    # a column correctly diagnosed as needing imputation, with Multiple
    # Imputation correctly selected and correctly run, still produced
    # a report claiming "Not applicable — no missing values required
    # imputation", flatly contradicting the per-column diagnosis just
    # above it in the same report. These two lists are appended to
    # inside the loop and turned into the actual summaries passed to
    # build_processing_report() further down.
    ts_imputation_summary_lines = []
    ts_bounds_summary_lines = []

    ts_mi_output_mode_map = {
        "One consolidated dataset (value + uncertainty columns) — recommended for most uses": "consolidated",
        "M complete datasets (no averaging — for formal statistical pooling)": "multiple_datasets",
        "Both": "both",
    }

    if not ts_needs_imputation:
        st.success("✅ No missing values detected in any value column -- imputation is correctly skipped.")
    else:
        st.info(f"Missing values detected in: {', '.join(ts_needs_imputation)}. Configure how to fill each below.")

        # Asked ONCE for the whole dataset -- single-value vs multiple
        # imputation is a decision about the analysis workflow, not
        # something that genuinely varies column by column. Previously
        # asked separately for every column, which was repetitive.
        ts_global_approach_q = ts_impq.generate_global_imputation_approach_question(ts_needs_imputation)
        ts_global_approach_label = st.radio(
            ts_global_approach_q["text"],
            ts_global_approach_q["options"],
            index=ts_global_approach_q["options"].index(ts_global_approach_q["default"]),
            key=ts_global_approach_q["id"],
        )

        # These were STILL being asked separately for every column even
        # after the approach question above was made global -- exactly
        # what was reported as still not fixed. Fixed the same way:
        # asked once here, reused for every column that ends up using
        # Multiple Imputation, rather than repeated per column.
        # m (draw count) and output mode remain sensible GLOBAL
        # workflow preferences -- but WHICH METHOD is actually best is
        # now a per-column decision (matching Single-Value imputation's
        # own UX exactly), since different columns can genuinely have
        # very different best-performing methods, evidenced directly
        # in earlier testing. The old fixed single-option "which MI
        # method" selectbox is retired now that a full ranked
        # comparison across every method exists.
        ts_global_mi_m = None
        ts_global_mi_mode = None
        if ts_global_approach_label.startswith("Multiple imputation"):
            ts_global_mi_m = st.slider(
                "Number of plausible draws (m)", min_value=5, max_value=30, value=10, key="ts_global_mi_m",
            )
            ts_global_mi_output_label = st.radio(
                "Multiple Imputation output",
                list(ts_mi_output_mode_map.keys()),
                index=0,
                key="ts_global_mi_output_mode",
            )
            ts_global_mi_mode = ts_mi_output_mode_map[ts_global_mi_output_label]

        # Plausibility bounds stay PER COLUMN (genuinely column-specific
        # characteristics -- e.g. Precipitation can't go negative, but
        # Temperature can), collected here before the unified ranking.
        ts_column_bounds_dict = {}
        for ts_impute_col in ts_needs_imputation:
            ts_observed_values = ts_imputed_df.loc[ts_imputed_df[ts_impute_col].notna(), ts_impute_col]
            ts_suggested_lower = ts_bounds.detect_suggested_lower_bound(ts_observed_values)
            ts_bounds_dict = {}
            if ts_suggested_lower is not None:
                ts_use_lower_bound = st.checkbox(
                    f"Prevent negative imputed values in '{ts_impute_col}' (every observed value is >= 0)",
                    value=True, key=f"ts_lower_bound_enabled__{ts_impute_col}",
                )
                if ts_use_lower_bound:
                    ts_bounds_dict["lower"] = ts_suggested_lower
            with st.expander(f"Custom plausibility bounds for '{ts_impute_col}' (optional)", expanded=False):
                ts_custom_lower_enabled = st.checkbox("Set a custom lower bound", value=False, key=f"ts_custom_lower_enabled__{ts_impute_col}")
                if ts_custom_lower_enabled:
                    ts_bounds_dict["lower"] = st.number_input(
                        "Lower bound", value=float(ts_suggested_lower or 0.0), key=f"ts_custom_lower__{ts_impute_col}"
                    )
                ts_custom_upper_enabled = st.checkbox("Set a custom upper bound", value=False, key=f"ts_custom_upper_enabled__{ts_impute_col}")
                if ts_custom_upper_enabled:
                    ts_bounds_dict["upper"] = st.number_input(
                        "Upper bound", value=float(ts_observed_values.max()) if len(ts_observed_values) else 100.0,
                        key=f"ts_custom_upper__{ts_impute_col}",
                    )
            ts_column_bounds_dict[ts_impute_col] = ts_bounds_dict if ts_bounds_dict else None

        # --- ONE unified ranking across ALL columns, per the client's
        # explicit request: comparing raw errors from columns on very
        # different scales would unfairly let whichever has the
        # largest numbers dominate the aggregate decision, so every
        # column is standardized first (Module AI/35) and every
        # method's error is computed in that common, comparable unit
        # (Module AR/44). Each column counts equally toward the
        # overall verdict, regardless of how many test points it
        # happened to contribute.
        ts_columns_data = {col: ts_imputed_df.set_index("Time_Step")[col] for col in ts_needs_imputation}
        ts_cycle_lengths_by_col = {col: ts_freq_info["cycle_length"] for col in ts_needs_imputation}
        ts_other_series_dicts = {
            col: (
                {c: ts_imputed_df.set_index("Time_Step")[c] for c in ts_active_value_columns if c != col}
                if len([c for c in ts_active_value_columns if c != col]) > 0 else None
            )
            for col in ts_needs_imputation
        }

        with st.spinner(f"Comparing every method across all {len(ts_needs_imputation)} column(s)..."):
            if ts_global_approach_label.startswith("Single-value"):
                ts_unified_results = ts_unified.run_unified_single_value_backtest(
                    ts_columns_data, ts_cycle_lengths_by_col, ts_other_series_dicts
                )
            else:
                ts_unified_results = ts_unified.run_unified_mi_backtest(
                    ts_columns_data, ts_cycle_lengths_by_col, ts_other_series_dicts, m=ts_global_mi_m
                )

        if ts_unified_results.empty or ts_unified_results.iloc[0]["n_successful"] == 0:
            st.warning("Not enough currently-observed data to run a reliable comparison -- defaulting to linear interpolation for every column.")
            ts_unified_chosen_method = "linear"
        else:
            if ts_global_approach_label.startswith("Single-value"):
                st.caption(
                    "Tested every method across all columns (values standardized first so no single column's scale "
                    "dominates the comparison) -- ranked best to worst by average standardized MAE:"
                )
                st.dataframe(
                    ts_unified_results[["rank", "method", "n_columns_tested", "columns_tested", "n_successful", "standardized_mae"]],
                    width="stretch",
                )
            else:
                st.caption(
                    "Tested every method's uncertainty quality across all columns (values standardized first) -- ranked "
                    "by combined score (interval accuracy weighted 2x, MAAPE-based accuracy weighted 1x):"
                )
                st.dataframe(
                    ts_unified_results[["rank", "method", "n_columns_tested", "columns_tested", "n_successful", "combined_score"]],
                    width="stretch",
                )

            ts_unified_options = ts_unified_results["method"].tolist()
            ts_unified_default = ts_unified_results.iloc[0]["method"]
            ts_unified_chosen_method = st.selectbox(
                f"Recommended method for ALL columns: '{ts_unified_default}'. Use this, or choose a different tested method:",
                ts_unified_options, index=ts_unified_options.index(ts_unified_default), key="ts_unified_override",
            )
            if ts_unified_chosen_method != ts_unified_default:
                st.caption(f"Using '{ts_unified_chosen_method}' — overriding the recommended '{ts_unified_default}'.")

        # --- Apply the ONE chosen method to every column ---
        for ts_impute_col in ts_needs_imputation:
            ts_column_bounds = ts_column_bounds_dict[ts_impute_col]
            if ts_column_bounds:
                ts_bounds_summary_lines.append(
                    f"{ts_impute_col}: lower={ts_column_bounds.get('lower')}, upper={ts_column_bounds.get('upper')} "
                    f"(applied only to imputed values, never to real observations)"
                )

            if ts_global_approach_label.startswith("Single-value"):
                ts_result_df, ts_imp_report = ts_unified.apply_unified_method_to_column(
                    ts_imputed_df, ts_impute_col, ts_unified_chosen_method,
                    cycle_length=ts_freq_info["cycle_length"], other_series_dict=ts_other_series_dicts[ts_impute_col],
                    bounds=ts_column_bounds,
                )
                ts_imputed_df = ts_result_df
                st.success(f"Filled {ts_imp_report['n_imputed']} missing value(s) in '{ts_impute_col}' using '{ts_imp_report['method']}'.")
                ts_imputation_summary_lines.append(
                    f"{ts_impute_col}: {ts_imp_report['n_imputed']} value(s) filled using '{ts_imp_report['method']}' "
                    f"(single-value imputation)"
                )
                if ts_imp_report["n_could_not_fill"]:
                    st.warning(
                        f"{ts_imp_report['n_could_not_fill']} value(s) in '{ts_impute_col}' could not be filled "
                        f"(Time_Step {ts_imp_report['time_steps_could_not_fill']}) and remain genuinely missing -- "
                        f"never estimated by guessing."
                    )
                    ts_imputation_summary_lines.append(
                        f"  -- {ts_imp_report['n_could_not_fill']} value(s) could not be filled and remain missing "
                        f"(Time_Step {ts_imp_report['time_steps_could_not_fill']})"
                    )
            else:
                ts_mi_result = ts_unified.apply_unified_mi_to_column(
                    ts_imputed_df, ts_impute_col, ts_unified_chosen_method,
                    cycle_length=ts_freq_info["cycle_length"], other_series_dict=ts_other_series_dicts[ts_impute_col],
                    m=ts_global_mi_m, output_mode="both", bounds=ts_column_bounds,
                )
                ts_imputed_df = ts_mi_result["consolidated_df"]
                st.success(
                    f"Filled {ts_mi_result['n_imputed']} missing value(s) in '{ts_impute_col}' using "
                    f"'{ts_mi_result['method_used']}' (m={ts_global_mi_m})."
                )
                ts_imputation_summary_lines.append(
                    f"{ts_impute_col}: {ts_mi_result['n_imputed']} value(s) filled using "
                    f"'{ts_mi_result['method_used']}' (Multiple Imputation, m={ts_global_mi_m}, "
                    f"mode={ts_global_mi_mode})"
                )
                if ts_global_mi_mode in ("consolidated", "both"):
                    st.caption(
                        f"Uncertainty columns added for '{ts_impute_col}': '_imputation_sd', "
                        f"'_imputation_lower_95', '_imputation_upper_95'."
                    )
                if ts_global_mi_mode in ("multiple_datasets", "both"):
                    ts_mi_datasets_this_run = ts_mi_result["m_datasets"]
                    st.info(
                        f"{ts_global_mi_m} complete datasets generated for '{ts_impute_col}' -- available as a separate "
                        f"download once you click 'Generate Time Series cleaned files' below."
                    )

    # --- Generate & download outputs (Modules W/X, 23-24) ---
    st.header("2e. Generate Cleaned Files")
    if st.button("Generate Time Series cleaned files", type="primary", key="ts_generate_btn"):
        ts_minimal_output = ts_out.build_minimal_ml_ready_output(ts_imputed_df, ts_active_value_columns)
        ts_original_output = ts_out.build_original_frequency_output(
            ts_imputed_df, ts_time_columns, ts_split_report["component_columns"], ts_active_value_columns
        )
        ts_agg_df, ts_agg_report = ts_freq.aggregate_to_frequency(
            ts_imputed_df, ts_freq_info, ts_active_value_columns, ts_multiple, ts_agg_method
        )
        ts_selected_freq_output = ts_out.build_user_selected_frequency_output(ts_agg_df, ts_agg_method)

        ts_imputation_summary = (
            {"status": "\n".join(ts_imputation_summary_lines)}
            if ts_imputation_summary_lines
            else {"status": "Not applicable — no missing values required imputation"}
        )
        ts_plausibility_summary = (
            {"status": "Custom plausibility bounds applied:\n" + "\n".join(ts_bounds_summary_lines)}
            if ts_bounds_summary_lines
            else {"status": "No custom plausibility bounds were set for any column"}
        )

        ts_processing_report = ts_report.build_processing_report(
            dataset_type="Univariate time series" if len(ts_active_value_columns) == 1 else "Multivariate time series",
            time_column=ts_time_columns,
            value_columns=ts_active_value_columns,
            component_meanings=ts_all_meanings,
            separator=ts_separator_summary,
            calendar_type=ts_calendar_type,
            split_report=ts_split_report,
            sort_report=ts_sort_report,
            duplicate_report=ts_dup_report,
            gap_report=ts_gap_report,
            freq_info=ts_freq_info,
            output_frequency_report={"multiple_applied": ts_multiple, "method_applied": ts_agg_method, "n_output_periods": len(ts_selected_freq_output)},
            placeholder_log=ts_placeholder_log,
            non_numeric_log=ts_non_numeric_log,
            missingness_diagnosis=ts_diagnosis,
            imputation_summary=ts_imputation_summary,
            plausibility_summary=ts_plausibility_summary,
        )

        ts_excel_buffer = io.BytesIO()
        # engine="xlsxwriter", NOT "openpyxl" -- confirmed via direct XML
        # inspection that pandas 3.0.2's openpyxl writer produces an
        # invalid empty <c t="inlineStr"></c> cell (no <is> child, which
        # the OOXML schema requires) for ANY missing value sitting in a
        # text/object column, triggering Excel's "problem with content"
        # repair dialog. This isn't fixable by choosing a different NA
        # sentinel (None/pd.NA/np.nan/"" all reproduce it identically) --
        # it's a bug in pandas' own openpyxl-writer layer for this
        # version. xlsxwriter doesn't have the same bug (verified).
        with pd.ExcelWriter(ts_excel_buffer, engine="xlsxwriter") as writer:
            # Minimal ML-ready sheet goes FIRST -- this is what a client
            # opening the file for modelling purposes actually wants to
            # see immediately, not buried behind the audit-trail sheet.
            ts_minimal_output.to_excel(writer, sheet_name="ML-Ready (Time_Step + Values)", index=False)
            ts_original_output.to_excel(writer, sheet_name="Original Frequency (Detailed)", index=False)
            ts_selected_freq_output.to_excel(writer, sheet_name="Selected Frequency", index=False)
            # Guard against Excel's own file parser treating a cell as an
            # attempted formula whenever its text starts with =, +, -, or
            # @ -- confirmed via a real Excel repair log ("Removed
            # Records: Formula") even though the cell is explicitly
            # string-typed. The banner lines that actually caused this
            # are fixed at the source in Module X, but this is a general
            # safety net for any future report line (a dynamic value, a
            # new field) that might coincidentally start with one of
            # these characters. A single leading space, not an
            # apostrophe -- verified directly that a leading apostrophe
            # gets written into the cell literally and stays visible
            # ('=text, not =text); Excel's "force text" apostrophe
            # convention is a live-typing UI behavior, not something
            # file-writing libraries honour. A leading space is
            # genuinely unobtrusive and still moves the actual first
            # character away from the ones Excel's parser checks for.
            ts_report_lines = ts_report.format_processing_report(ts_processing_report).split("\n")
            ts_report_lines = [
                f" {line}" if line[:1] in ("=", "+", "-", "@") else line for line in ts_report_lines
            ]
            pd.DataFrame({"Processing Report": ts_report_lines}).to_excel(
                writer, sheet_name="Processing Report", index=False
            )
        ts_excel_buffer.seek(0)

        # If Multiple Imputation's M-datasets/Both mode was used for any
        # column, also build a separate workbook: one sheet per draw,
        # each a full minimal ML-ready view (Time_Step + values) for
        # that specific completed dataset.
        ts_mi_workbook_bytes = None
        if ts_mi_datasets_this_run is not None:
            ts_mi_buffer = io.BytesIO()
            with pd.ExcelWriter(ts_mi_buffer, engine="xlsxwriter") as mi_writer:
                for i, variant_df in enumerate(ts_mi_datasets_this_run):
                    variant_minimal = ts_out.build_minimal_ml_ready_output(variant_df, ts_active_value_columns)
                    variant_minimal.to_excel(mi_writer, sheet_name=f"Dataset {i+1}", index=False)
            ts_mi_buffer.seek(0)
            ts_mi_workbook_bytes = ts_mi_buffer.getvalue()

        # Store in session_state rather than rendering the download
        # button here directly. st.download_button, like any widget,
        # triggers its OWN script rerun when clicked -- and on that
        # rerun, this "if st.button(Generate...)" condition resets to
        # False, so this entire block (download button included) would
        # simply stop rendering and the button would vanish mid-click.
        # Persisting to session_state and rendering the download button
        # in an unconditional section below (same pattern the existing
        # General/ML "Results" section already uses correctly) is what
        # keeps it visible and functional across that rerun.
        st.session_state.ts_result = (
            ts_excel_buffer.getvalue(),
            ts_report.format_processing_report(ts_processing_report),
        )
        if ts_mi_workbook_bytes is not None:
            st.session_state.ts_mi_datasets_workbook = (ts_mi_workbook_bytes, len(ts_mi_datasets_this_run))

    if st.session_state.ts_result is not None:
        ts_excel_bytes, ts_report_text = st.session_state.ts_result
        st.success("Done — your cleaned time series file is ready below.")
        st.download_button(
            "Download cleaned Time Series file (.xlsx)",
            data=ts_excel_bytes,
            file_name="cleaned_time_series.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ts_download_btn",
        )
        # Preview read back from the stored bytes (not a variable held
        # over from the button click) -- same reasoning as the download
        # button itself: anything only computed inside the "if
        # st.button(...)" block disappears the moment a DIFFERENT
        # widget (like this download button) triggers its own rerun,
        # so reading it back from what's already persisted in
        # session_state is what keeps it visible afterward.
        ts_preview_df = pd.read_excel(io.BytesIO(ts_excel_bytes), sheet_name="ML-Ready (Time_Step + Values)")
        st.dataframe(ts_preview_df.head(20), width="stretch")
        with st.expander("Processing report", expanded=False):
            st.text(ts_report_text)

    if st.session_state.ts_mi_datasets_workbook is not None:
        ts_mi_bytes, ts_mi_count = st.session_state.ts_mi_datasets_workbook
        st.download_button(
            f"Download {ts_mi_count} Multiple Imputation datasets (.xlsx)",
            data=ts_mi_bytes,
            file_name="time_series_mi_datasets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ts_mi_download_btn",
        )

    st.stop()


# ---------------------------------------------------------------------------
# Step 3: Dynamic clarifying questions
# ---------------------------------------------------------------------------

st.header("3. A few quick questions")

answers = {}
missing_required = False

# Collected earlier, in the "Explore unique values per column" section
# -- kept as a plain reserved key in `answers` (not a question ID)
# since it doesn't come from a generated question at all, just direct
# client-typed corrections.
answers["value_corrections"] = st.session_state.get("value_corrections", {})

if not questions:
    st.write("Nothing ambiguous detected in this file — no questions needed.")
else:
    for q in questions:
        # Generic "hide this question if some OTHER question's CURRENT
        # answer equals a given value" mechanism -- not specific to any
        # one feature (currently used by the currency default-currency
        # gate, so a client who says "yes, there's one default" never
        # has to click through the per-value dropdowns that answer
        # would make redundant). Works because Streamlit reruns this
        # entire script on every widget change, so by the time this
        # loop reaches a DEPENDENT question, the widget it depends on
        # (rendered earlier in the same pass, since generation order
        # already puts the gate before what it hides) has already
        # produced its current value in `answers`.
        dep = q.get("depends_on")
        if dep and answers.get(dep["question_id"]) == dep["hide_if_value"]:
            continue  # hidden this run -- no widget rendered, no answer recorded

        # BUG FIXED after testing: a rate question now exists for
        # EVERY detected currency (see Module 11's own fix note), so
        # if the client's LIVE target happens to be this question's
        # own source currency, it would otherwise render a nonsensical
        # "convert 1 GBP into GBP" -- skipped entirely here instead,
        # the same way a depends_on-hidden question is skipped above,
        # since no rate is needed to convert a currency into itself.
        if q["id"].startswith("exchange_rate__"):
            rate_col = q["affected_columns"][0]
            live_target = answers.get(f"target_currency__{rate_col}", q["assumed_target_currency"])
            if q["source_currency"] == live_target:
                continue

        # BUG FIXED after client feedback: an exchange-rate question
        # ("What is the exchange rate to convert 1 EUR into USD?") was
        # generated ONCE, at question-generation time, phrased against
        # the column's DEFAULT (most common) currency -- if the client
        # then picked a DIFFERENT target currency (e.g. GBP instead of
        # the default USD), the rate question kept asking about USD
        # regardless, since the static question list is never rebuilt.
        # Fixed here rather than by rebuilding the whole question list:
        # target_currency__<col> always renders BEFORE its column's
        # rate questions (Module 11 generates them in that order), so
        # by this point in the SAME rerun, `answers` already holds
        # whatever the client just picked -- this rewrites the rate
        # question's own text and target-currency code to match it
        # live, every rerun, so the question on screen always asks
        # about the currency actually selected, not a frozen default.
        if q["id"].startswith("exchange_rate__"):
            rate_col = q["affected_columns"][0]
            live_target = answers.get(f"target_currency__{rate_col}", q["assumed_target_currency"])
            live_target_label = qgen.CURRENCY_DISPLAY_LABELS.get(live_target, live_target)
            source_currency = q["source_currency"]
            source_label = qgen.CURRENCY_DISPLAY_LABELS.get(source_currency, source_currency)
            q = dict(q)  # don't mutate the cached question list itself
            q["live_target_currency"] = live_target
            q["question_text"] = (
                f"What is the exchange rate to convert 1 {source_currency} into "
                f"{live_target}? (i.e. 1 {source_currency} = ? {live_target}) -- used to "
                f"convert '{rate_col}' into your chosen target currency, {live_target_label}."
            )

        answer = render_question(q)
        answers[q["id"]] = answer
        if q["default"] is None and answer is None:
            missing_required = True

# ---------------------------------------------------------------------------
# Step 3-review: Batch review for ambiguous phone/email values
# ---------------------------------------------------------------------------
# Per the client's own spec (Section 5): "do not interrupt the user
# with one question per bad value" -- unresolved phone/email values are
# collected here into ONE editable table instead of the per-value
# dropdown pattern used for currency/duration. Computed AFTER the main
# question loop above (needs the client's confirm/standardize/format/
# region/casing answers to know what "unresolved" even means for each
# column), but BEFORE the pipeline actually runs (Module 12's apply_
# phone_standardization / apply_email_standardization read the
# decisions collected here back out of `answers`).
#
# Deliberately given NO numbered heading (unlike "3", "3a" above) to
# avoid colliding with this file's pre-existing "3a" label used for
# the missing-value-markers section right below -- this section sits
# between the two.

phone_review_items = phone_conv.find_unresolved_phone_values(df, profile, answers)
email_review_items = email_conv.find_unresolved_email_values(df, profile, answers)

if phone_review_items or email_review_items:
    st.header("Review: Ambiguous Phone & Email Values")
    st.caption(
        "These values couldn't be safely standardised automatically. Pick a decision for each one "
        "below -- 'Don't know' is treated as explicit permission to remove the value and replace it "
        "with a missing value (your chosen missing-data policy then applies to it normally)."
    )

    # Built only when the relevant upstream answers actually change (a
    # signature of every standardize/format/region/casing answer that
    # could affect which values are unresolved), NOT rebuilt from
    # phone_review_items/email_review_items on every rerun -- same bug
    # already documented and fixed above for the profile-type editor:
    # this table's OWN edits get written back into `answers` further
    # down, which would otherwise feed a changing source back into
    # st.data_editor under the same key on every rerun, confusing its
    # edit-tracking (rows could even disappear entirely once decided,
    # since find_unresolved_* skips anything already answered -- an
    # even more disruptive version of the same bug).
    review_signature = tuple(sorted(
        (k, v) for k, v in answers.items()
        if k.startswith((
            "standardize_phone__", "target_phone_format__", "default_phone_region__",
            "standardize_email__", "email_casing__",
        ))
    ))
    if st.session_state.get("phone_email_review_signature") != review_signature:
        review_rows = []
        for item in phone_review_items:
            review_rows.append({
                "Type": "Phone",
                "Column": item["column"],
                "Row": item["row_index"],
                "Original value": str(item["original_value"]),
                "Problem": item["problem"],
                "Expected format": item["expected_format"],
                "Decision": "Keep original value",
                "Corrected value": "",
            })
        for item in email_review_items:
            review_rows.append({
                "Type": "Email",
                "Column": item["column"],
                "Row": item["row_index"],
                "Original value": str(item["original_value"]),
                "Problem": item["problem"],
                "Expected format": "",
                "Decision": "Keep original value",
                "Corrected value": "",
            })
        st.session_state.phone_email_review_base = pd.DataFrame(review_rows)
        st.session_state.phone_email_review_signature = review_signature

    review_df = st.session_state.phone_email_review_base
    edited_review_df = st.data_editor(
        review_df,
        key="widget_phone_email_review_editor",  # "widget_" prefix -> auto-cleared on new file upload
        hide_index=True,
        width="stretch",
        disabled=["Type", "Column", "Row", "Original value", "Problem", "Expected format"],
        column_config={
            "Decision": st.column_config.SelectboxColumn(
                options=["Keep original value", "Enter corrected value", "Mark as missing", "Don't know"],
                required=True,
            ),
        },
    )

    _DECISION_LABEL_TO_ACTION = {
        "Keep original value": "keep",
        "Enter corrected value": "correct",
        "Mark as missing": "missing",
        "Don't know": "dont_know",
    }
    for _, row in edited_review_df.iterrows():
        action = _DECISION_LABEL_TO_ACTION[row["Decision"]]
        prefix = "phone_review" if row["Type"] == "Phone" else "email_review"
        decision = {"action": action}
        if action == "correct":
            decision["corrected_value"] = row["Corrected value"]
        answers[f"{prefix}__{row['Column']}__{row['Row']}"] = decision

# ---------------------------------------------------------------------------
# Step 3b: Missing-value handling (per-column method choice)
# ---------------------------------------------------------------------------

NUMERIC_IMPUTATION_METHODS = {
    "KNN (smart estimate based on similar rows) — default": "knn",
    "Multiple Imputation (MICE — adds value + sd + 95% interval columns)": "multiple",
    "Mean (column average)": "mean",
    "Median (column midpoint)": "median",
    "Zero": "zero",
    "Leave missing (do not fill in)": "leave",
}
CATEGORICAL_IMPUTATION_METHODS = {
    "Leave missing (do not fill in) — default": "leave",
    "KNN (smart estimate based on similar rows)": "knn",
    "Multiple Imputation (adds value + probability columns per category)": "multiple",
    "Mode (most common value) — default when preparing for ML": "mode",
}

def _strip_optional_quotes(token: str) -> str:
    """
    Quotes around a value are entirely optional -- Hello, "Hello",
    and 'Hello' all resolve to the exact same value. Added after
    real-user testing found the opposite: without this, a
    quoted entry like "Hello" was being matched LITERALLY
    (including the quote characters), so it silently matched
    NOTHING in real data and failed with no error message at all --
    exactly the kind of silent failure that's worse than a visible
    one, since it looked like it worked.
    """
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("\"", "'"):
        return token[1:-1].strip()
    return token


st.header("3a. Are there hidden missing-value markers?")
st.caption(
    "Answered here (not in Advanced Settings) because it affects which columns show up as "
    "needing an imputation method below."
)
extra_missing_text = st.text_input(
    "Additional values to treat as missing, across every column (comma-separated)",
    value="",
    help="Covers placeholder conventions we might not catch automatically — e.g. if your file uses '99' as a placeholder number, type 99 here and it'll be treated as missing wherever it appears, even in columns that didn't trigger a question above. Quotes are optional — 99, \"99\", and '99' all work the same way.",
)
extra_missing_values = [_strip_optional_quotes(v) for v in extra_missing_text.split(",") if v.strip()]

with st.expander("Column-specific missing-value markers (optional)", expanded=False):
    st.caption(
        "The field above applies everywhere. Use this when a value only means 'missing' in "
        "ONE specific column — e.g. 'low' might be a real category in a 'Priority' column but "
        "genuinely mean missing in a 'Rating' column. Pick from the column's own values below, "
        "or type a value that isn't in the list. Leave blank for columns where this doesn't apply."
    )
    column_specific_missing_values = {}
    for col in df.columns:
        column_own_values = sorted(df[col].dropna().unique().tolist(), key=str)
        selected = st.multiselect(
            f"'{col}': values to treat as missing",
            options=column_own_values,
            default=[],
            accept_new_options=True,  # lets the client type a value not already present in the data
            key=f"widget_colmissing__{col}",
        )
        markers = [_strip_optional_quotes(str(v)) for v in selected if str(v).strip()]
        if markers:
            column_specific_missing_values[col] = markers


def _count_marker_matches(series: pd.Series, markers: list) -> int:
    """
    Counts how many cells in `series` WOULD become missing given
    `markers`, without actually modifying anything -- used purely to
    decide whether a column needs an imputation-method dropdown below,
    mirroring the real matching logic in Module K's
    apply_user_specified_missing_values / apply_column_specific_missing_values
    (numeric columns matched numerically, text matched case-insensitively).
    """
    if not markers:
        return 0
    if pd.api.types.is_numeric_dtype(series):
        numeric_markers = set()
        for v in markers:
            try:
                numeric_markers.add(float(str(v).strip()))
            except ValueError:
                continue
        if not numeric_markers:
            return 0
        return int(series.isin(numeric_markers).sum())
    text_markers = {str(v).strip().lower() for v in markers}
    normalised = series.astype(str).str.strip().str.lower()
    return int((normalised.isin(text_markers) & series.notna()).sum())


binary_positive_value_overrides = {}
if prepare_for_ml:
    binary_cols = [col for col, info in profile.items() if info.get("variable_type") == "binary" and col in df.columns]
    if binary_cols:
        with st.expander("Machine Learning Ready: binary column encoding (optional)", expanded=False):
            st.caption(
                "Each binary column gets recoded to 0/1. Choose which raw value becomes 1 below, "
                "or leave on Auto for a consistent default rule (disclosed in the Actions Log either way)."
            )
            for col in binary_cols:
                distinct_vals = sorted(df[col].dropna().unique().tolist(), key=str)
                if len(distinct_vals) != 2:
                    continue
                choice = st.selectbox(
                    f"'{col}': which value should be 1?",
                    ["Auto (default rule)"] + [str(v) for v in distinct_vals],
                    index=0,
                    key=f"widget_binpos__{col}",
                )
                if choice != "Auto (default rule)":
                    for v in distinct_vals:
                        if str(v) == choice:
                            binary_positive_value_overrides[col] = v
                            break

ordinal_category_order_overrides = {}
if prepare_for_ml:
    # Only TEXT-labelled ordinal columns need ranking -- a column
    # that's already numeric (e.g. a 1-5 severity score) already HAS
    # its order built in and needs no extra input.
    #
    # Deliberately checks the RAW dataframe's actual dtype here, NOT
    # info["dtype"] -- confirmed as a real bug by testing: profile is
    # the SAME object across Streamlit reruns, and the loop below used
    # to mutate info["dtype"] to "numeric" the instant a column was
    # fully ranked. Since this filter also checked info["dtype"], a
    # column would silently vanish from the list (and lose its ranking
    # from ordinal_category_order_overrides) the moment ANY other
    # widget triggered the next rerun -- exactly what happens when
    # ranking a second column, since every rerun rebuilds this dict
    # from scratch by re-iterating this now-shorter list. df[col]'s
    # actual dtype never gets mutated by this code, so checking it
    # directly keeps the list -- and therefore every already-completed
    # ranking -- stable across reruns.
    text_ordinal_cols = [
        col for col, info in profile.items()
        if info.get("variable_type") == "ordinal" and col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if text_ordinal_cols:
        with st.expander("Machine Learning Ready: rank ordinal categories (required)", expanded=True):
            st.caption(
                "There's no way to know 'high' outranks 'moderate' just from the text — rank each "
                "column's categories from LOWEST to HIGHEST by clicking them in order below. "
                "They'll be encoded as 0, 1, 2, ... in the order you pick."
            )
            for col in text_ordinal_cols:
                # Values the client already said mean "missing" for
                # this column (via the global field or this column's
                # own marker list, see section 2a above) are excluded
                # from what needs ranking -- confirmed as a real bug
                # by testing: requiring 'vhigh' to be ranked even
                # after the client explicitly marked it as a missing
                # placeholder meant the ranking could never be
                # "complete", so the column silently stayed
                # categorical/text forever, no matter how thoroughly
                # the client ranked the REAL categories.
                missing_markers_for_col = {str(v).strip().lower() for v in extra_missing_values}
                missing_markers_for_col |= {str(v).strip().lower() for v in column_specific_missing_values.get(col, [])}
                raw_unique = df[col].dropna().unique().tolist()
                unique_cats = sorted(
                    [v for v in raw_unique if str(v).strip().lower() not in missing_markers_for_col],
                    key=str,
                )
                ordered = st.multiselect(
                    f"'{col}': click categories in order, lowest to highest",
                    options=unique_cats,
                    default=[],
                    key=f"widget_ordinal_rank__{col}",
                )
                if len(ordered) == len(unique_cats) and len(ordered) > 0:
                    ordinal_category_order_overrides[col] = ordered
                    # Still patched for downstream display purposes
                    # (so the Missing-Value Handling section below
                    # shows this column under "Numeric columns") --
                    # but this mutation is now harmless, since
                    # text_ordinal_cols above no longer depends on it.
                    profile[col]["dtype"] = "numeric"
                    profile[col]["semantic_role"] = "numeric_measure"
                elif ordered:
                    st.warning(f"'{col}': rank all {len(unique_cats)} categories (currently {len(ordered)}) or this column won't be encoded as ordinal.")

numeric_text_value_overrides = {}
if prepare_for_ml:
    # Columns the client marked "integer" or "continuous" that still
    # mix in some non-numeric text (e.g. a "doors" column with
    # 2/3/4 alongside "5more") -- these need a real NUMBER for each
    # stray text value, not just a rank position (unlike ordinal),
    # since the column's actual magnitude matters for a genuine
    # count or measurement.
    cols_needing_number_resolution = {}
    for col, info in profile.items():
        if info.get("variable_type") not in ("integer", "continuous") or col not in df.columns:
            continue
        unresolved = ml_ready.find_unresolved_text_values(df[col])
        if unresolved:
            cols_needing_number_resolution[col] = unresolved

    if cols_needing_number_resolution:
        with st.expander("Machine Learning Ready: resolve non-numeric text values (required)", expanded=True):
            st.caption(
                "These columns are marked Integer/Continuous but contain some text values that "
                "aren't plain numbers (e.g. '5more'). Enter the number each one should become — "
                "leave blank to treat it as missing instead."
            )
            for col, unresolved in cols_needing_number_resolution.items():
                st.markdown(f"**'{col}'**")
                value_map = {}
                for val in unresolved:
                    entry = st.text_input(
                        f"'{val}' should become:",
                        value="",
                        key=f"widget_numtext__{col}__{val}",
                    )
                    if entry.strip():
                        try:
                            value_map[val] = float(entry.strip())
                        except ValueError:
                            st.warning(f"'{entry}' isn't a number — '{val}' will be left as missing.")
                if value_map:
                    numeric_text_value_overrides[col] = value_map

NON_ANALYTICAL_ROLES = {
    "id_key": "identifier",
    "full_name": "personal name",
    "address": "address",
    "email": "email",
    "phone": "phone number",
    "date_or_datetime": "timestamp",
}
drop_non_analytical_columns = []
if prepare_for_ml:
    non_analytical_cols = [
        (col, NON_ANALYTICAL_ROLES[info["semantic_role"]])
        for col, info in profile.items()
        if info["semantic_role"] in NON_ANALYTICAL_ROLES and col in df.columns
    ]
    if non_analytical_cols:
        with st.expander("Machine Learning Ready: identifiers, personal info & timestamps", expanded=False):
            st.caption(
                "These columns are never encoded for ML — they're kept exactly as-is by default, "
                "since they're identifiers, personal info, or timestamps rather than analytical "
                "variables. Check any you'd like REMOVED from the ML-ready output entirely instead "
                "(e.g. genuine PII, or an ID column risking data leakage into a model)."
            )
            for col, kind in non_analytical_cols:
                remove = st.checkbox(
                    f"Remove '{col}' ({kind}) from ML-ready output",
                    value=False,
                    key=f"widget_dropcol__{col}",
                )
                if remove:
                    drop_non_analytical_columns.append(col)

st.header("4. How should missing values be filled in? (optional)")
st.caption(
    "By default, missing values are estimated using KNN (a 'similar rows' guess). "
    "That's not always right — e.g. a blank bonus column more likely means '$0 bonus' "
    "than 'unknown bonus'. Change the default below, or override individual columns."
)

def _is_reclassified_as_categorical(info: dict) -> bool:
    """
    Mirrors Module K's build_pipeline_config exactly: a numeric-CODED
    binary/categorical column (e.g. "Sex" stored as 1/2) is routed to
    CATEGORICAL imputation regardless of its semantic role. This has
    to be kept in sync with that routing logic -- otherwise this UI
    would offer a numeric-method dropdown for a column whose method
    choice the pipeline would silently ignore, since it's actually
    processed as categorical underneath.
    """
    return (
        info.get("variable_type") in ("binary", "categorical")
        and info["dtype"] == "numeric"
        and info["semantic_role"] in ("numeric_measure", "numeric_text")
    )


def _effective_missing_count(col: str, info: dict) -> int:
    """
    The auto-detected missing_count PLUS whatever the client's own
    missing-value markers would ADD on top, from THREE separate
    sources: the global "Additional values..." field, this column's
    entry in "Column-specific missing-value markers" (2a), AND the
    per-column disguised-missing markers confirmed back in section 2's
    dynamic questions (e.g. confirming 'na' means missing in a
    specific column) -- so a column that had ZERO auto-detected
    missing values but genuinely will have some once these markers
    take effect still gets an imputation-method dropdown below,
    instead of silently staying invisible because the ORIGINAL
    profile (computed before the client answered anything) didn't know
    about them yet.

    That third source (disguised-missing answers) was previously
    MISSING from this calculation entirely -- confirmed by testing:
    a file whose only missing values were disguised markers like 'na'
    (all correctly detected and pre-confirmed in section 2's
    questions) showed "No numeric or categorical columns have missing
    values" here in section 3, hiding the KNN/Mean/Median/Zero method
    choice even though Module 12's apply_disguised_missing_answers()
    already applies those exact same confirmed markers when the
    pipeline actually runs. So the real cleaning worked, but the
    client had no visibility into or control over WHICH method was
    used, since the column never appeared as needing one.
    """
    if col not in df.columns:
        return info["missing_count"]
    extra = _count_marker_matches(df[col], extra_missing_values)
    extra += _count_marker_matches(df[col], column_specific_missing_values.get(col, []))
    extra += _count_marker_matches(df[col], answers.get(f"disguised_missing__{col}", []) or [])
    return info["missing_count"] + extra


# Preview which columns WILL become numeric (or categorical) once
# Modules 49/45/47/50 (word-number, weight/length, currency, duration
# standardisation) actually run -- NOT necessarily the role Module 10's
# ORIGINAL profiling assigned, since a column like "Age" (raw text
# mixing "25 years", "nineteen years", plain numbers, some genuinely
# blank) only becomes "numeric_measure" AFTER those conversions run.
# Checking the original profile alone would mean this section never
# offers a missing-value method choice for such a column at all --
# confirmed by testing: an Age column with 2 genuinely missing values
# showed NO imputation option whatsoever here, even though Module 12's
# real pipeline correctly imputed it anyway (silently defaulting to
# KNN, with no visibility or control for the client) -- the exact same
# category of bug already documented and fixed above for disguised-
# missing markers (_effective_missing_count's own docstring), just
# with standardisation instead of missing-marker detection as the root
# cause. The missing-value COUNT itself doesn't need recomputing here
# -- Module 10 already counts raw NaN regardless of semantic role, so
# profile[col]["missing_count"] was already correct all along; the
# column just never reached the eligibility list below.
#
# Wrapped defensively: this preview exists purely to improve THIS UI
# section's column list, not to affect the real pipeline at all, so a
# failure here should never break the rest of the page -- it falls
# back to the original (pre-conversion) profile roles if anything goes
# wrong.
try:
    _preview_df = df.copy()
    _preview_profile = {col: dict(info) for col, info in profile.items()}
    _preview_df, _ = text_num.apply_text_number_conversion(_preview_df, _preview_profile, answers)
    _preview_df, _, _ = measurement_conv.apply_measurement_unit_conversion(_preview_df, _preview_profile, answers)
    _preview_df, _, _ = currency_conv.apply_currency_conversion(_preview_df, _preview_profile, answers)
    _preview_df, _, _ = duration_conv.apply_duration_conversion(_preview_df, _preview_profile, answers)
except Exception:
    _preview_profile = {}


def _effective_semantic_role(col: str) -> str:
    return _preview_profile.get(col, {}).get("semantic_role", profile[col]["semantic_role"])


numeric_missing_cols = [
    col for col, info in profile.items()
    if _effective_semantic_role(col) in ("numeric_measure", "numeric_text")
    and not _is_reclassified_as_categorical(info)
    and _effective_missing_count(col, info) > 0
]
categorical_missing_cols = [
    col for col, info in profile.items()
    if (_effective_semantic_role(col) in ("categorical", "country", "state") or _is_reclassified_as_categorical(info))
    and _effective_missing_count(col, info) > 0
]

numeric_flagged = [c for c in numeric_missing_cols if _looks_like_structural_missingness_column(c)]
numeric_other = [c for c in numeric_missing_cols if c not in numeric_flagged]
categorical_flagged = [c for c in categorical_missing_cols if _looks_like_structural_missingness_column(c)]
categorical_other = [c for c in categorical_missing_cols if c not in categorical_flagged]

numeric_imputation_overrides = {}
categorical_imputation_overrides = {}


def _render_column_override_row(col: str, methods: dict, key_prefix: str, warn: bool = False) -> str:
    """
    Renders one column's override selectbox, returns the chosen label.
    `warn` shows a short business-meaning prompt above the dropdown --
    used for columns whose NAME suggests missingness might mean
    something specific (e.g. Bonus, Commission, Discount, Quantity),
    per the spec's "short warning or business-meaning prompt" guidance
    -- rather than treating every column identically.
    """
    if warn:
        st.caption(f"⚠️ '{col}' — a blank here might genuinely mean '0' or 'none', not 'unknown'. Worth a specific choice rather than the shared default.")
    if profile[col].get("variable_type") == "ordinal" and profile[col].get("dtype") != "numeric":
        st.caption(
            f"ℹ️ '{col}' is marked **ordinal** but hasn't been ranked yet — it's still being treated "
            f"as a plain categorical column here. Go to 'Machine Learning Ready: rank ordinal "
            f"categories' above and click through all of its categories, lowest to highest, to "
            f"convert it to numbers."
        )
    return st.selectbox(
        f"'{col}' ({_effective_missing_count(col, profile[col])} missing)",
        ["Use default"] + list(methods.keys()),
        index=0,
        key=f"{key_prefix}__{col}",
    )


# Multiple Imputation is General-mode-EXCLUDED / ML-mode-only, and
# Leave-missing is the mirror image: ML-mode-EXCLUDED / General-mode-
# only. Leaving a column unfilled is fine for a plain "clean my data"
# delivery, but breaks ML-ready encoding downstream -- a binary/
# one-hot/ordinal column with missing values left in it can't be
# encoded to 0/1 (see the categorical-default help text above), so
# offering "Leave missing" as a real choice in ML mode would let a
# client pick a combination that silently fails later. Filtered here
# (rather than at the two dicts' original definitions) so both modes
# keep drawing from one shared source of truth for the method codes.
numeric_methods_available = (
    {label: code for label, code in NUMERIC_IMPUTATION_METHODS.items() if code != "leave"} if prepare_for_ml
    else {label: code for label, code in NUMERIC_IMPUTATION_METHODS.items() if code != "multiple"}
)
categorical_methods_available = (
    {label: code for label, code in CATEGORICAL_IMPUTATION_METHODS.items() if code != "leave"} if prepare_for_ml
    else {label: code for label, code in CATEGORICAL_IMPUTATION_METHODS.items() if code != "multiple"}
)
categorical_method_labels_available = list(categorical_methods_available.keys())

with st.expander("Missing-value handling", expanded=False):
    if numeric_missing_cols:
        st.markdown("**Numeric columns**")
        numeric_default_label = st.selectbox(
            "Default method for numeric columns",
            list(numeric_methods_available.keys()),
            index=0,
            help=(
                "Applies to every numeric column below unless overridden individually. "
                "KNN is the smartest estimate but slower on very large files — Mean/Median/Zero "
                "are much faster and often good enough for big, mostly-numeric datasets."
            ),
        )
        numeric_imputation_default = numeric_methods_available[numeric_default_label]

        if numeric_flagged:
            st.caption("These columns are worth a second look — missing here may not mean 'unknown':")
            for col in numeric_flagged:
                label = _render_column_override_row(col, numeric_methods_available, "numimp", warn=True)
                if label != "Use default":
                    numeric_imputation_overrides[col] = numeric_methods_available[label]

        if numeric_other:
            with st.expander(f"Show all {len(numeric_other)} other numeric column(s) with missing values", expanded=False):
                for col in numeric_other:
                    label = _render_column_override_row(col, numeric_methods_available, "numimp")
                    if label != "Use default":
                        numeric_imputation_overrides[col] = numeric_methods_available[label]
    else:
        numeric_imputation_default = "knn"

    if categorical_missing_cols:
        st.markdown("**Categorical columns**")
        categorical_default_index = categorical_method_labels_available.index(
            "Mode (most common value) — default when preparing for ML" if prepare_for_ml
            else "Leave missing (do not fill in) — default"
        )
        categorical_default_label = st.selectbox(
            "Default method for categorical columns",
            categorical_method_labels_available,
            index=categorical_default_index,
            help=(
                "Applies to every categorical column below unless overridden individually. "
                "Normally defaults to 'leave missing' rather than auto-filling — this kind of "
                "survey data skews toward demographic/identity columns (gender, race, education), "
                "where auto-guessing a category is a bigger judgment call than estimating a number "
                "like age. Switches to 'Mode' by default when 'Machine Learning Ready — Cross-Sectional "
                "Data' is "
                "selected above, since a column left with missing values can't be encoded to 0/1 "
                "or one-hot columns — encoding requires complete data first."
            ),
        )
        categorical_imputation_default = categorical_methods_available[categorical_default_label]

        if categorical_flagged:
            st.caption("These columns are worth a second look — missing here may not mean 'unknown':")
            for col in categorical_flagged:
                label = _render_column_override_row(col, categorical_methods_available, "catimp", warn=True)
                if label != "Use default":
                    categorical_imputation_overrides[col] = categorical_methods_available[label]

        if categorical_other:
            with st.expander(f"Show all {len(categorical_other)} other categorical column(s) with missing values", expanded=False):
                for col in categorical_other:
                    label = _render_column_override_row(col, categorical_methods_available, "catimp")
                    if label != "Use default":
                        categorical_imputation_overrides[col] = categorical_methods_available[label]
    else:
        categorical_imputation_default = "leave"

    any_multiple_selected = (
        numeric_imputation_default == "multiple" or categorical_imputation_default == "multiple"
        or "multiple" in numeric_imputation_overrides.values()
        or "multiple" in categorical_imputation_overrides.values()
    )
    if any_multiple_selected:
        multiple_imputation_m = st.slider(
            "Multiple Imputation: number of plausible draws (m)",
            min_value=5, max_value=30, value=10,
            help=(
                "How many independent estimates to generate per missing value. Higher gives a "
                "more stable estimate but takes proportionally longer to run."
            ),
        )
        mi_output_mode_label = st.radio(
            "Multiple Imputation output",
            [
                "One consolidated dataset (value + uncertainty columns) — recommended for most uses",
                "M complete datasets (no averaging — for formal statistical pooling)",
                "Both",
            ],
            index=0,
            help=(
                "Consolidated: one clean file with the filled-in value plus columns disclosing how "
                "uncertain each estimate is (e.g. for client delivery or a dashboard). "
                "M complete datasets: a separate, fully complete file per draw — no averaging, no "
                "uncertainty columns — for feeding into Rubin's-rules-style pooled analysis or "
                "training one model per dataset. Both produces every file."
            ),
        )
        mi_output_mode = {
            "One consolidated dataset (value + uncertainty columns) — recommended for most uses": "consolidated",
            "M complete datasets (no averaging — for formal statistical pooling)": "multiple_datasets",
            "Both": "both",
        }[mi_output_mode_label]
    else:
        multiple_imputation_m = 10
        mi_output_mode = "consolidated"

    if not numeric_missing_cols and not categorical_missing_cols:
        st.write("No numeric or categorical columns have missing values in this file.")


# ---------------------------------------------------------------------------
# Step 4: Advanced settings (collapsed, sensible defaults)
# ---------------------------------------------------------------------------

with st.expander("Advanced settings", expanded=False):
    knn_relevant = (
        bool(numeric_missing_cols or categorical_missing_cols)  # nothing to impute at all -> irrelevant regardless of method
        and (
            numeric_imputation_default == "knn"
            or categorical_imputation_default in ("knn", "multiple")  # categorical Multiple Imputation also uses k internally
            or "knn" in numeric_imputation_overrides.values()
            or any(m in ("knn", "multiple") for m in categorical_imputation_overrides.values())
        )
    )
    if knn_relevant:
        knn_k = st.slider(
            "Neighbours (k) used for filling in missing values",
            min_value=2, max_value=15, value=5,
            help="How many similar rows to look at when filling in a missing number or category. Higher = smoother, more averaged results; lower = more sensitive to individual similar rows.",
        )
    else:
        knn_k = 5  # irrelevant while nothing uses KNN/neighbour-based Multiple Imputation, but still needs a value to pass through
    drop_threshold = st.slider(
        "Drop a column if at least this % of it is missing",
        min_value=50, max_value=100, value=90,
        help="Columns that are almost entirely empty (e.g. an unused field) get removed rather than kept and filled in.",
    )
    cap_outliers = st.checkbox(
        "Cap extreme outlier values in numeric columns",
        value=False,
        help=(
            "Off by default. When ON, values far outside the normal range for a numeric "
            "column (e.g. an age of 250) get pulled in to the nearest 'reasonable' bound "
            "instead of being left as-is. This is genuinely useful for columns where an "
            "extreme value is almost certainly a data-entry mistake — but for a naturally "
            "wide-ranging column like salary, a real high earner can be flagged the same way "
            "a typo would be, and capping would incorrectly flatten a real, valid value. "
            "Outliers are always detected and reported either way — this only controls "
            "whether their VALUES get changed."
        ),
    )
    if cap_outliers:
        outlier_multiplier = st.slider(
            "Outlier sensitivity (higher = only cap more extreme values)",
            min_value=1.0, max_value=4.0, value=1.5, step=0.5,
            help="Standard statistical convention is 1.5. Raise this if genuinely valid values in a wide-ranging column (like salary) are being caught too aggressively.",
        )
    else:
        outlier_multiplier = 1.5  # irrelevant while capping is off, but detection still uses it to decide what counts as an outlier in the report
    uk_cols = _country_columns_with_values(profile, df, profiling.UK_CONSTITUENT_COUNTRIES)
    us_cols = _country_columns_with_values(profile, df, set(profiling.US_STATES_TO_COUNTRY.keys()))

    # PER-COLUMN checkboxes, not one global checkbox per rule. A file
    # can genuinely have more than one "country"-role column (e.g. a
    # real country column AND a "what US state do you work in?"
    # column) that need OPPOSITE answers to the same question -- see
    # _country_columns_with_values' docstring for why a single global
    # toggle was actively wrong, not just imprecise.
    merge_uk_constituents_by_column = {}
    if uk_cols:
        st.markdown("**England / Scotland / Wales / Northern Ireland → \"United Kingdom\"**")
        for col in uk_cols:
            merge_uk_constituents_by_column[col] = st.checkbox(
                f"Apply to '{col}'",
                value=False,
                key=f"merge_uk__{col}",
                help=(
                    "These are the UK's real constituent countries, not incorrect data — "
                    "leave unchecked to keep them separate for this column if it needs that "
                    "regional detail."
                ),
            )

    merge_us_states_by_column = {}
    if us_cols:
        st.markdown("**US states (e.g. \"New York\", \"California\") → \"United States\"**")
        for col in us_cols:
            merge_us_states_by_column[col] = st.checkbox(
                f"Apply to '{col}'",
                value=True,
                key=f"merge_us__{col}",
                help=(
                    "Unlike UK nations, a US state in a genuine country column is almost "
                    "always a data-entry mistake — on by default. Uncheck this for a column "
                    "that's actually ASKING for a US state (e.g. 'What state do you work "
                    "in?'), where collapsing every answer to 'United States' would destroy "
                    "the detail the question was meant to capture."
                ),
            )

advanced_settings = {
    "knn_k": knn_k,
    "drop_column_missing_threshold": drop_threshold,
    "cap_outliers": cap_outliers,
    "outlier_iqr_multiplier": outlier_multiplier,
    "numeric_imputation_default": numeric_imputation_default,
    "numeric_imputation_overrides": numeric_imputation_overrides,
    "categorical_imputation_default": categorical_imputation_default,
    "categorical_imputation_overrides": categorical_imputation_overrides,
    "multiple_imputation_m": multiple_imputation_m,
    "prepare_for_ml": prepare_for_ml,
    "binary_positive_value_overrides": binary_positive_value_overrides,
    "ordinal_category_order_overrides": ordinal_category_order_overrides,
    "numeric_text_value_overrides": numeric_text_value_overrides,
    "drop_non_analytical_columns": drop_non_analytical_columns,
    "merge_uk_constituents_by_column": merge_uk_constituents_by_column,
    "merge_us_states_by_column": merge_us_states_by_column,
    "extra_missing_values": extra_missing_values,
    "column_specific_missing_values": column_specific_missing_values,
}


# ---------------------------------------------------------------------------
# Step 5: Run
# ---------------------------------------------------------------------------

st.header("5. Clean your data")

if missing_required:
    st.warning("Please answer the question(s) above before cleaning — these can't be safely guessed.")

_MI_OUTPUT_MODE_CONFIRMATION = {
    "consolidated": "Will produce: **one consolidated dataset** (with uncertainty columns).",
    "multiple_datasets": f"Will produce: **{multiple_imputation_m} complete datasets** (one workbook, no uncertainty columns).",
    "both": f"Will produce: **one consolidated dataset AND {multiple_imputation_m} complete datasets**.",
}
if any_multiple_selected:
    st.caption(_MI_OUTPUT_MODE_CONFIRMATION[mi_output_mode])

def _build_mi_datasets_workbook(results: list) -> bytes:
    """
    Packages the list of dicts from run_interactive_pipeline_multiple_datasets()
    into ONE downloadable Excel workbook, one sheet per dataset
    (Dataset_01, Dataset_02, ...) -- simpler to open and browse than a
    zip of separate files for a moderate M, per real-user feedback.

    Rather than repeating a full Summary + Actions Log per dataset (M
    near-identical copies would just be noise), this adds ONE shared
    "Summary" sheet and ONE shared "Actions Log" sheet up front --
    every dataset goes through the exact same pipeline steps and ends
    up the exact same shape; the only thing that ever differs between
    them is WHICH Multiple Imputation draw got used, which the Actions
    Log sheet's note makes explicit. Each dataset's own outlier-flag
    columns are still split into a per-dataset "..._Outliers" sheet,
    matching the single-dataset report's own separation of concerns.
    """
    excel_buffer = io.BytesIO()
    first = results[0]
    comparison = report.build_comparison(first["before_summary"], first["after_summary"])

    summary_df = pd.DataFrame([
        {"Metric": "Rows (before)", "Value": comparison["rows_before"]},
        {"Metric": "Rows (after)", "Value": comparison["rows_after"]},
        {"Metric": "Rows removed", "Value": comparison["rows_removed"]},
        {"Metric": "Columns (before)", "Value": comparison["columns_before"]},
        {"Metric": "Columns (after)", "Value": comparison["columns_after"]},
        {"Metric": "Duplicate rows (before)", "Value": comparison["duplicate_rows_before"]},
        {"Metric": "Duplicate rows (after)", "Value": comparison["duplicate_rows_after"]},
        {"Metric": "Number of datasets (M)", "Value": len(results)},
    ])
    actions_df = pd.DataFrame({"Action performed": first["actions_log"]})

    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        actions_df.to_excel(writer, sheet_name="Actions Log", index=False)
        for result in results:
            main_df, outlier_flags_df = report._split_off_outlier_flag_columns(result["df"])
            sheet_name = f"Dataset_{result['index']:02d}"
            main_df.to_excel(writer, sheet_name=sheet_name, index=False)
            if outlier_flags_df is not None:
                flagged = report._build_outlier_flags_sheet(outlier_flags_df)
                if not flagged.empty:
                    flagged.to_excel(writer, sheet_name=f"{sheet_name}_Outliers"[:31], index=False)

    return excel_buffer.getvalue()


if st.button(
    "Clean my data",
    disabled=missing_required or time_series_mode,
    type="primary",
    help=(
        "Time Series mode isn't built yet — see the warning above." if time_series_mode else None
    ),
):
    with st.spinner("Cleaning your data..."):
        st.session_state.cleaned_result = None
        st.session_state.mi_datasets_workbook = None

        if mi_output_mode in ("consolidated", "both"):
            excel_buffer = io.BytesIO()
            cleaned_df = runner.run_interactive_pipeline(
                df.copy(), profile, answers, excel_buffer, advanced_settings=advanced_settings
            )
            st.session_state.cleaned_result = (cleaned_df, excel_buffer.getvalue())

        if mi_output_mode in ("multiple_datasets", "both"):
            results = runner.run_interactive_pipeline_multiple_datasets(
                df.copy(), profile, answers, advanced_settings=advanced_settings
            )
            # Stores Dataset_01's dataframe alongside the workbook
            # purely for a representative preview -- there's no single
            # "the" cleaned dataset in this mode (that's the whole
            # point), but showing nothing at all before download isn't
            # very reassuring either, so the first of the M datasets
            # stands in as a sample of what the workbook contains.
            st.session_state.mi_datasets_workbook = (
                _build_mi_datasets_workbook(results), len(results), results[0]["df"]
            )



# ---------------------------------------------------------------------------
# Step 6: Results
# ---------------------------------------------------------------------------

if st.session_state.cleaned_result is not None or st.session_state.mi_datasets_workbook is not None:
    st.header("6. Results")

    if st.session_state.cleaned_result is not None:
        cleaned_df, excel_bytes = st.session_state.cleaned_result

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df)} → {len(cleaned_df)}")
        col2.metric("Columns", f"{len(df.columns)} → {len(cleaned_df.columns)}")
        col3.metric("Missing values (raw)", f"{int(df.isna().sum().sum())} → {int(cleaned_df.isna().sum().sum())}")

        st.subheader("Cleaned data preview")
        st.dataframe(cleaned_df.head(20), width="stretch")

        st.download_button(
            "Download cleaned Excel file",
            data=excel_bytes,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

    if st.session_state.mi_datasets_workbook is not None:
        workbook_bytes, dataset_count, first_dataset_preview = st.session_state.mi_datasets_workbook
        st.subheader("M complete datasets")
        st.write(
            f"{dataset_count} fully complete datasets generated, each with one plausible draw per "
            f"Multiple-Imputation column (no averaging, no uncertainty columns) — ready for pooled "
            f"statistical analysis or training one model per dataset. Each is its own sheet "
            f"(Dataset_01, Dataset_02, ...) in the workbook below, with a shared Summary and Actions Log."
        )
        st.caption("Preview of Dataset_01 (one representative sample — the other datasets differ only in their Multiple-Imputation draws):")
        st.dataframe(first_dataset_preview.head(20), width="stretch")
        st.download_button(
            "Download all datasets (one workbook)",
            data=workbook_bytes,
            file_name="cleaned_imputation_datasets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
