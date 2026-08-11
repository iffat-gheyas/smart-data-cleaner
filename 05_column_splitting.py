"""
Module E: Column Splitting
----------------------------
Splits three kinds of "combined" columns into separate, analysis-ready
fields:
  1. full_name  -> first_name, last_name
  2. address    -> street, city, postcode
  3. a standardised datetime column -> year, month, day, hour, minute, second

This module assumes date/time columns have ALREADY been through
Module D (date/time standardisation) -- splitting a properly parsed,
consistent datetime is straightforward; splitting a column that still
has five different formats in it is not.
"""

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Full name splitting
# ---------------------------------------------------------------------------

# Common titles that sometimes appear in front of a name. Checked in
# lowercase with any trailing period stripped, so "Dr", "dr.", "DR" all
# match the same way regardless of whether Module B's casing
# standardisation has already run.
TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "professor"}


def split_full_name(df: pd.DataFrame, column: str = "full_name") -> pd.DataFrame:
    """
    Split a full-name column into first_name and last_name.

    Design choice: the FIRST remaining word (after removing a title,
    if present) becomes first_name. Everything after it becomes
    last_name -- this correctly keeps hyphenated surnames like
    "Harper-Taylor" together as one piece (a hyphen isn't a space, so
    it's never split), and also means a middle name would end up
    folded into last_name rather than lost. This is a reasonable
    default, not a perfect solution for every possible name structure.

    OUTPUT COLUMN NAMES, per explicit client request: always
    "{original column name}: First Name" / "{original column name}:
    Last Name" -- NOT a fixed "first_name"/"last_name" regardless of
    the source column. This matters specifically when a file has more
    than one name column (e.g. "Customer Name" and "Contact Person"),
    since a fixed name would make the two splits indistinguishable or
    silently collide.

    COLUMN POSITION, per explicit client request: the two new columns
    are inserted exactly where the original column used to be, not
    appended at the end of the dataframe -- df[new_col] = ... always
    appends, so simply assigning and then dropping the original (the
    previous approach) left every split column trailing after every
    OTHER column in the file, however unrelated. Using df.insert() at
    the original column's own position keeps the file's overall shape
    recognisable to whoever's reading it.
    """
    df = df.copy()
    first_names = []
    last_names = []

    for value in df[column]:
        if pd.isna(value):
            first_names.append(np.nan)
            last_names.append(np.nan)
            continue

        tokens = value.strip().split()

        # Drop a leading title if present, e.g. "Dr Karl Kerr" -> "Karl Kerr"
        if tokens and tokens[0].strip(".").lower() in TITLES:
            tokens = tokens[1:]

        if not tokens:
            first_names.append(np.nan)
            last_names.append(np.nan)
            continue

        first_names.append(tokens[0])
        last_names.append(" ".join(tokens[1:]) if len(tokens) > 1 else np.nan)

    original_position = df.columns.get_loc(column)
    df = df.drop(columns=[column])
    df.insert(original_position, f"{column}: First Name", first_names)
    df.insert(original_position + 1, f"{column}: Last Name", last_names)
    return df


# ---------------------------------------------------------------------------
# 2. Address splitting
# ---------------------------------------------------------------------------

# The exact set of separators our synthetic generator uses to join
# street / city / postcode. Checked in this order deliberately: the
# first three are distinctive enough that they won't appear by
# accident inside a street name, so we check for them first. "\n" is
# checked last because a street address can ALSO contain its own
# internal newline (e.g. a "Flat 53" line) even when it wasn't the
# separator chosen for this particular row. Plain "," (no required
# space) is checked LAST of all -- it's the most likely to appear
# accidentally, so it only gets used once none of the more distinctive
# separators are found.
ADDRESS_SEPARATOR_CANDIDATES = [" | ", " - ", ", ", "\n", ","]

# Same postcode shape Module A's profiling uses to DETECT an address
# column in the first place -- reused here as the actual split anchor,
# so detection and splitting stay consistent with each other. Captures
# the postcode itself (group 1) plus any leading whitespace/comma
# before it, so the remainder can be cleanly separated from it.
_UK_POSTCODE_AT_END_RE = re.compile(r"[,\s]*([A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2})\s*$")

# Common British street-type suffix words -- used ONLY as a last-resort
# fallback (see split_address_column) when a value has NO separator at
# all between street and city, e.g. "89 King Street Manchester M2 4WU".
# The LAST occurrence of one of these words marks where the street
# portion ends and the city name begins -- correctly handles even a
# multi-word city name ("12 North Road Newcastle upon Tyne NE1 4LP" ->
# street "12 North Road", city "Newcastle upon Tyne"), which a simple
# "last word is the city" guess would get wrong.
_STREET_SUFFIX_WORDS = {
    "street", "road", "lane", "avenue", "boulevard", "square", "close",
    "drive", "way", "gardens", "place", "terrace", "crescent", "court",
    "row", "hill", "park", "grove", "walk", "mews",
}


def _detect_address_separator(text: str):
    for sep in ADDRESS_SEPARATOR_CANDIDATES:
        if sep in text:
            return sep
    return None


def _split_street_and_city_by_suffix(remainder: str):
    """
    Fallback used ONLY when there's no separator at all between street
    and city (see split_address_column) -- finds the LAST word that
    looks like a street-type suffix (Road/Street/Avenue/...) and treats
    everything up to and including it as the street, everything after
    it as the city. Returns (street, city) -- city is None if no
    suffix word was found at all (street kept as the whole remainder
    rather than guessing where it splits).
    """
    words = remainder.split()
    suffix_index = None
    for i, word in enumerate(words):
        if word.strip(".,").lower() in _STREET_SUFFIX_WORDS:
            suffix_index = i  # keep looking -- we want the LAST match
    if suffix_index is None:
        return remainder, None
    street = " ".join(words[: suffix_index + 1])
    city = " ".join(words[suffix_index + 1 :])
    return street, (city if city else None)


def split_address_column(df: pd.DataFrame, column: str = "address") -> pd.DataFrame:
    """
    Split a combined address column into street, city, and postcode.

    REWRITTEN after real-file testing found the previous separator-
    counting approach (requiring the SAME separator to appear at least
    twice) failing on most genuinely messy addresses -- of 19 real
    values from a client file, only 6 had two or more separators; the
    rest had exactly one, or none at all. Now anchors on the POSTCODE
    instead, which was present and recognisable in every one of those
    19 values regardless of comma style:
      1. Peel the postcode off the END first, using the same shape
         Module A's detection already relies on.
      2. Split whatever's left (street + city) on the LAST separator
         found in it, if any.
      3. If there's no separator at all, fall back to finding the last
         street-type suffix word (Road/Street/Avenue/...) as the
         street/city boundary -- see _split_street_and_city_by_suffix.
      4. If NEITHER a postcode nor a separator can be found at all,
         the value is kept whole as street, with city/postcode left
         missing -- never guessed at.

    OUTPUT COLUMN NAMES, per explicit client request: always
    "{original column name}: Street" / "{original column name}: City" /
    "{original column name}: Postcode" -- NOT a fixed "street"/"city"/
    "postcode" regardless of the source column. This matters
    specifically when a file has more than one address column (e.g.
    "Customer Address" and "Company Address" in the same row), since a
    fixed name would make the two splits indistinguishable or silently
    overwrite each other.

    HONEST LIMITATION: the postcode anchor is UK-shaped specifically
    (matching this project's existing UK-postcode-validation
    convention elsewhere) -- an address using a different country's
    postal code format falls back to the old separator-based approach
    (steps 2-4 above still apply, just without a postcode anchor to
    start from).

    COLUMN POSITION, per explicit client request: the three new
    columns are inserted exactly where the original column used to be
    (see split_full_name's docstring for the same fix and reasoning),
    not appended at the end of the dataframe.
    """
    df = df.copy()
    streets, cities, postcodes = [], [], []

    for value in df[column]:
        if pd.isna(value):
            streets.append(np.nan)
            cities.append(np.nan)
            postcodes.append(np.nan)
            continue

        text = str(value).strip()
        postcode_match = _UK_POSTCODE_AT_END_RE.search(text)

        if postcode_match:
            postcode = postcode_match.group(1).strip()
            remainder = text[: postcode_match.start()].strip()
            remainder = remainder.rstrip(",").strip()

            sep = _detect_address_separator(remainder)
            if sep is not None:
                parts = remainder.rsplit(sep, 1)
                street = parts[0].strip()
                city = parts[1].strip() if len(parts) == 2 else None
                # Validate the split: if the "city" side STILL contains
                # a street-suffix word, the separator wasn't actually
                # the street/city boundary -- e.g. "Flat 2A, 17 Mill
                # Road Cambridge" has its only comma after the FLAT
                # number, not between street and city, so naively
                # splitting on it wrongly puts "17 Mill Road Cambridge"
                # entirely into "city". Caught during testing; falls
                # back to the suffix-word method on the WHOLE remainder
                # in this case, which correctly finds "Road" as the
                # true boundary regardless of where the comma sits.
                if city and any(
                    w.strip(".,").lower() in _STREET_SUFFIX_WORDS for w in city.split()
                ):
                    suffix_street, suffix_city = _split_street_and_city_by_suffix(remainder)
                    if suffix_city:
                        street, city = suffix_street, suffix_city
            else:
                street, city = _split_street_and_city_by_suffix(remainder)

            streets.append(street)
            cities.append(city if city else np.nan)
            postcodes.append(postcode)
            continue

        # No recognisable postcode at the end -- fall back to the
        # original separator-counting approach entirely (see the
        # module docstring's honest limitation on non-UK postcodes).
        sep = _detect_address_separator(text)
        if sep is None:
            streets.append(text)
            cities.append(np.nan)
            postcodes.append(np.nan)
            continue

        parts = text.rsplit(sep, 2)
        if len(parts) == 3:
            streets.append(parts[0].strip())
            cities.append(parts[1].strip())
            postcodes.append(parts[2].strip())
        else:
            streets.append(text)
            cities.append(np.nan)
            postcodes.append(np.nan)

    original_position = df.columns.get_loc(column)
    df = df.drop(columns=[column])
    df.insert(original_position, f"{column}: Street", streets)
    df.insert(original_position + 1, f"{column}: City", cities)
    df.insert(original_position + 2, f"{column}: Postcode", postcodes)
    return df


# ---------------------------------------------------------------------------
# 3. Datetime splitting
# ---------------------------------------------------------------------------


def split_datetime_column(
    df: pd.DataFrame,
    column: str,
    include_time: bool = True,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Split an already-standardised date or datetime column into
    separate year/month/day (and optionally hour/minute/second)
    columns.

    Set include_time=False for pure date columns (e.g. signup_date)
    where there's no meaningful time component -- no point creating
    hour/minute/second columns that would just be zero everywhere.

    OUTPUT COLUMN NAMES, per explicit client request: always
    "{original column name}: Year", "{original column name}: Month",
    etc. -- NOT the old "{column}_year" (lowercase, underscore) style.
    e.g. splitting "Delivery DateTime" produces "Delivery DateTime:
    Year", "Delivery DateTime: Month", "Delivery DateTime: Day", and
    (if include_time) "Delivery DateTime: Hour", "Delivery DateTime:
    Minute", "Delivery DateTime: Second". This also matters when a
    file has more than one date/time column, since each one's split
    fields stay clearly attributed to their source column.

    COLUMN POSITION, per explicit client request: the new columns are
    inserted starting exactly where the original column used to be
    (see split_full_name's docstring for the same fix and reasoning),
    not appended at the end of the dataframe. If drop_original=False,
    the original column is kept in place and the new columns are
    inserted immediately AFTER it, rather than at the very end.
    """
    df = df.copy()

    # By this point the column should already be in one consistent
    # format thanks to Module D, so this parse step is safe and won't
    # hit the dayfirst ambiguity problem we dealt with there.
    parsed = pd.to_datetime(df[column], format="mixed", errors="coerce")

    new_columns = {f"{column}: Year": parsed.dt.year, f"{column}: Month": parsed.dt.month, f"{column}: Day": parsed.dt.day}
    if include_time:
        new_columns[f"{column}: Hour"] = parsed.dt.hour
        new_columns[f"{column}: Minute"] = parsed.dt.minute
        new_columns[f"{column}: Second"] = parsed.dt.second

    if drop_original:
        insert_at = df.columns.get_loc(column)
        df = df.drop(columns=[column])
    else:
        insert_at = df.columns.get_loc(column) + 1

    for offset, (col_name, values) in enumerate(new_columns.items()):
        df.insert(insert_at + offset, col_name, values)

    return df


if __name__ == "__main__":
    from importlib import import_module

    ingestion = import_module("01_ingestion")
    text_std = import_module("02_text_standardisation")
    datetime_std = import_module("04_datetime_standardisation")

    # Colab-friendly hardcoded path, same as previous modules.
    filepath = "dirty_customer_data.xlsx"

    df = ingestion.load_data(filepath)

    # Realistic pipeline order: text standardisation, then date
    # standardisation, THEN splitting.
    df = text_std.trim_whitespace(df)
    df = text_std.standardise_casing(df, {"full_name": "title", "company": "title"})
    df, _ = datetime_std.standardise_date_column(df, "signup_date", dayfirst=True)
    df, _ = datetime_std.standardise_date_column(df, "last_order_date", dayfirst=True)
    df, _ = datetime_std.standardise_date_column(
        df, "transaction_timestamp", dayfirst=True, output_format="%Y-%m-%d %H:%M:%S"
    )

    print("--- Full name splitting ---")
    print(df[["full_name"]].head(8))
    df = split_full_name(df)
    print(df[["full_name: First Name", "full_name: Last Name"]].head(8))

    print("\n--- Address splitting ---")
    print(df[["address"]].head(8))
    df = split_address_column(df)
    print(df[["address: Street", "address: City", "address: Postcode"]].head(8))

    print("\n--- Datetime splitting ---")
    df = split_datetime_column(df, "signup_date", include_time=False)
    df = split_datetime_column(df, "last_order_date", include_time=False)
    df = split_datetime_column(df, "transaction_timestamp", include_time=True)
    print(
        df[
            [
                "transaction_timestamp: Year",
                "transaction_timestamp: Month",
                "transaction_timestamp: Day",
                "transaction_timestamp: Hour",
                "transaction_timestamp: Minute",
                "transaction_timestamp: Second",
            ]
        ].head(8)
    )
