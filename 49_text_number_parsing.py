"""
Module: Text-Number Parsing -- Sub-step A (Detection & Conversion)
---------------------------------------------------------------------
Generic-mode feature, and DELIBERATELY THE EARLIEST of the four unit-
related features in the pipeline (weight/length, currency, and this
one) -- Modules 45-48 all expect a DIGIT at the start of a value
("50 pounds" parses fine; "fifty pounds" does not, since their number-
extraction regexes look for \\d, not a word). This module's only job is
to turn "fifty pounds" into "50 pounds" -- it does NOT recognise or
convert any unit itself; that's left entirely to the modules that
already do it. Once this module runs, "fifty pounds" and "50 pounds"
are indistinguishable to everything downstream.

SCOPE, kept deliberately narrow:
  - Only converts a WORD-FORM NUMBER at the START of a cell ("fifty
    pounds" -> "50 pounds"). A currency/unit word BEFORE the number
    ("pounds fifty") is not handled -- flagged as a known limitation,
    not silently guessed at, since it's a much rarer phrasing.
  - Never touches a cell that's already digit-based ("50kg" passes
    through completely unchanged).
  - Never invents a unit or interprets one -- "fifty" alone becomes
    "50" alone; whatever text (if any) follows the number word is
    preserved EXACTLY as typed and just reattached after the digit.

A REAL BUG CAUGHT BEFORE BUILDING ON TOP OF IT: the word2number library
(used here for the actual word-to-value conversion, since re-
implementing English number-word arithmetic from scratch would be
reinventing a solved problem) will happily -- and WRONGLY -- parse
"nineteen eighty four" as 1524, because it never checks whether the
word sequence is actually a GRAMMATICALLY VALID number phrase, only
whether every individual word is A number word. "nineteen" and "eighty"
are both valid number words, but "nineteen eighty" is not a valid
number (no English speaker means one number when they say that) --
it's two different numbers with nothing connecting them. This module
therefore does its OWN lightweight grammar check (see
_is_valid_number_phrase) before ever trusting word2number's output,
rather than passing raw token runs to it unchecked.

A SECOND bug caught in the same testing pass: word2number silently
DROPS negative signs -- word_to_num("minus five") returns 5, not -5.
Handled here by stripping and reapplying the sign ourselves.
"""

import re

import pandas as pd
from word2number import w2n

# ---------------------------------------------------------------------------
# 1. Number-word vocabulary
# ---------------------------------------------------------------------------

UNITS_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
TEENS_WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALE_WORDS = {"hundred", "thousand", "million", "billion"}
CONNECTOR_WORDS = {"and", "a"}  # "a" as in "a hundred" = 100
NEGATIVE_WORDS = {"minus", "negative"}
DECIMAL_WORD = "point"

PLAIN_NUMBER_WORDS = set(UNITS_WORDS) | set(TEENS_WORDS) | set(TENS_WORDS)
ALL_NUMBER_VOCAB = (
    PLAIN_NUMBER_WORDS | SCALE_WORDS | CONNECTOR_WORDS | NEGATIVE_WORDS | {DECIMAL_WORD}
)


def _tokenize(text: str) -> list:
    """Splits on whitespace AND hyphens ("twenty-five" -> ["twenty", "five"])."""
    return [t for t in re.split(r"[\s\-]+", text.lower()) if t]


def _is_valid_number_phrase(tokens: list) -> bool:
    """
    The grammar check that catches "nineteen eighty four": rejects any
    sequence containing two CONSECUTIVE "plain" number words (units,
    teens, or tens) UNLESS the first is a TENS word and the second is a
    UNITS word (the one genuinely valid adjacency in English -- "twenty
    five", "sixty three", etc.).

    Two plain words in a row that AREN'T a tens+units pair means two
    unrelated numbers got jammed together with nothing (no "hundred",
    "thousand", no "and") connecting them -- not a single valid number.
    "nineteen" (teens) followed by "eighty" (tens) fails this check,
    correctly rejecting the whole phrase. "nineteen hundred" passes,
    since "hundred" is a SCALE word, not a plain one, so the adjacency
    check doesn't apply to that pair at all.

    Ignores CONNECTOR/SCALE/negative/decimal tokens for this check --
    they're allowed to sit between plain-number groups precisely
    because they're what makes multiple groups valid together (e.g.
    "two hundred and five": "two"+"hundred" and "and"+"five" are both
    fine pairs; "hundred"+"and" and "and"+"five" involve a connector).
    """
    plain_positions = [i for i, t in enumerate(tokens) if t in PLAIN_NUMBER_WORDS]
    for a, b in zip(plain_positions, plain_positions[1:]):
        if b != a + 1:
            continue  # not adjacent -- a scale/connector word sits between them, fine
        first, second = tokens[a], tokens[b]
        if first in TENS_WORDS and second in UNITS_WORDS:
            continue  # the one valid adjacency, e.g. "twenty five"
        return False  # any other adjacent plain-plain pair is invalid
    return True


def _find_leading_number_phrase(tokens: list):
    """
    Finds the LONGEST valid leading run of tokens (from the start of
    the list) that forms a legitimate number phrase, stopping at the
    first token that isn't part of the number vocabulary at all (that's
    where the unit text starts, e.g. "pounds" in "fifty pounds").

    Returns (phrase_tokens, remainder_tokens) -- phrase_tokens is empty
    if no valid number phrase was found at all (e.g. the cell doesn't
    start with a number word).

    IMPORTANT DISTINCTION, added after catching "nineteen eighty four"
    being partially accepted as "nineteen" (19) with "eighty four" left
    over as if it were a unit: stopping early because the NEXT token
    isn't a number word at all (e.g. "pounds") is a genuine, clean
    number/unit boundary. Stopping early because the NEXT token IS a
    number word, but adding it would break the grammar check (e.g.
    "eighty" after "nineteen"), is NOT a boundary -- it means the whole
    token run was an attempted number phrase that simply isn't valid,
    and the entire thing is rejected rather than salvaging a prefix.
    """
    best_end = 0
    for end in range(1, len(tokens) + 1):
        candidate = tokens[:end]
        if candidate[-1] not in ALL_NUMBER_VOCAB:
            break
        if not _is_valid_number_phrase(candidate):
            break
        best_end = end

    if best_end == 0:
        return [], tokens

    # Was the stopping point a genuine unit boundary (next token isn't
    # number vocabulary at all), or a grammar failure (next token IS
    # vocabulary, just not valid in sequence)? Only the former is a
    # real number+unit split -- the latter means reject everything.
    if best_end < len(tokens) and tokens[best_end] in ALL_NUMBER_VOCAB:
        return [], tokens

    return tokens[:best_end], tokens[best_end:]


def convert_leading_word_number(raw) -> tuple:
    """
    Converts a WORD-FORM number at the start of a cell to its digit
    form, keeping everything else exactly as typed.

      "fifty"                    -> (50.0,  "50",              True)
      "fifty pounds"              -> (50.0,  "50 pounds",       True)
      "twenty-five kg"            -> (25.0,  "25 kg",           True)
      "one hundred and fifty USD" -> (150.0, "150 USD",         True)
      "minus five degrees"        -> (-5.0,  "-5 degrees",      True)
      "50kg"                      -> (None,  "50kg",            False) -- already digit, untouched
      "nineteen eighty four"      -> (None,  "nineteen eighty four", False) -- REJECTED, not a valid number phrase
      "a dozen"                   -> (None,  "a dozen",         False) -- "dozen" isn't in our vocabulary
      "First"                     -> (None,  "First",           False) -- ordinal, not a cardinal number word

    Returns (numeric_value_or_None, converted_text, was_converted).
    converted_text is ALWAYS a string -- equal to the original text
    (str(raw)) when was_converted is False, so a caller can safely use
    converted_text unconditionally without checking was_converted first.
    """
    if pd.isna(raw):
        return (None, raw, False)

    text = str(raw).strip()
    if text == "":
        return (None, text, False)

    # Already digit-based -- nothing for this module to do; Modules
    # 45/47's own parsers already handle this correctly.
    if re.match(r"^-?\d", text):
        return (None, text, False)

    tokens = _tokenize(text)
    if not tokens:
        return (None, text, False)

    negative = False
    start = 0
    if tokens[0] in NEGATIVE_WORDS:
        negative = True
        start = 1

    phrase_tokens, remainder_tokens = _find_leading_number_phrase(tokens[start:])
    if not phrase_tokens:
        return (None, text, False)  # doesn't start with a recognisable number word at all

    phrase_str = " ".join(phrase_tokens)
    try:
        value = w2n.word_to_num(phrase_str)
    except ValueError:
        return (None, text, False)  # word2number itself rejected it -- leave untouched

    if negative:
        value = -value

    # Reattach whatever's left EXACTLY as it appeared in the original
    # text (preserving original casing/spacing for the unit part),
    # rather than reconstructing it from the lower-cased token list.
    # BUG FIXED after testing: splitting only on whitespace here (while
    # _tokenize() splits on whitespace AND hyphens) caused "twenty-five
    # kg" to lose "kg" entirely -- consumed_word_count=2 (for tokens
    # "twenty"+"five") was being matched against only 2 whitespace-
    # delimited words ("twenty-five" counted as ONE), so "kg" (meant to
    # be the 2nd word, i.e. the remainder) was wrongly swallowed as if
    # it were part of the number phrase. Splitting on hyphens here too
    # keeps the element count aligned with _tokenize()'s token count --
    # note ".strip()" alone does NOT identify a bare "-" delimiter
    # piece as empty (strip() only removes whitespace), so a proper
    # regex fullmatch is used instead to tell a delimiter piece from a
    # real word piece.
    consumed_word_count = start + len(phrase_tokens)
    original_words = re.split(r"([\s\-]+)", text)  # keep whitespace/hyphen runs as their own elements
    remainder_parts = []
    consumed = 0
    i = 0
    while i < len(original_words):
        piece = original_words[i]
        is_delimiter = bool(re.fullmatch(r"[\s\-]+", piece)) if piece else False
        if not is_delimiter:
            consumed += 1
            if consumed > consumed_word_count:
                remainder_parts.append(piece)
        elif consumed >= consumed_word_count:
            remainder_parts.append(piece)
        i += 1
    remainder_str = "".join(remainder_parts).strip().strip("-").strip()

    number_str = str(int(value)) if float(value).is_integer() else str(value)
    converted_text = f"{number_str} {remainder_str}".strip() if remainder_str else number_str

    return (float(value), converted_text, True)


# Matches a maximal run of Module 49's OWN number-word vocabulary
# (zero-nine, ten-nineteen, twenty-ninety, hundred/thousand/million/
# billion, and/point/minus/negative) ANYWHERE in a string -- built
# from ALL_NUMBER_VOCAB so it can never match a unit word from a
# completely different vocabulary (weight/currency/duration units
# aren't in this list at all), which is what keeps this safe to run
# on a whole compound value like "15 years nine months" without
# touching "years" or "months".
_NUMBER_WORD_RUN_RE = re.compile(
    r"\b(?:" + "|".join(sorted(ALL_NUMBER_VOCAB, key=len, reverse=True)) + r")\b"
    r"(?:[\s-]+\b(?:" + "|".join(sorted(ALL_NUMBER_VOCAB, key=len, reverse=True)) + r")\b)*",
    re.IGNORECASE,
)


def convert_all_word_numbers_in_text(text) -> str:
    """
    Converts EVERY word-form number found anywhere in the text, not
    just a leading one -- added after real-file testing found a real
    gap: convert_leading_word_number() only ever looks at the START of
    a value, so "15 years nine months" (digit-led, word-number
    embedded LATER) was left completely untouched -- "nine" never
    became "9" at all, silently losing part of a compound duration.

    Reuses convert_leading_word_number() itself for the actual
    conversion of each candidate run found -- the SAME grammar
    validation (rejecting "nineteen eighty four"-style invalid
    sequences) applies to every run, not just a leading one, so a
    genuinely nonsensical run is left untouched rather than guessed at.

      "15 years nine months"  -> "15 years 9 months"
      "20 yrs 7 months"       -> unchanged (already all-digit)
      "thirteen hrs fifteen mins" -> "13 hrs 15 mins"
      "nineteen eighty four"  -> unchanged (still correctly rejected --
                                  same grammar check, just applied to
                                  this run in isolation)

    Only ever touches tokens from Module 49's OWN number-word
    vocabulary -- a duration/weight/currency UNIT word right next to a
    number (e.g. "months", "kg", "GBP") is never part of the matched
    run at all, so it's never at risk of being mistaken for part of
    the number phrase.
    """
    if not isinstance(text, str):
        return text

    def _replace(match):
        run_text = match.group(0)
        _, converted, was_converted = convert_leading_word_number(run_text)
        return converted if was_converted else run_text

    return _NUMBER_WORD_RUN_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# 2. Column-level detection
# ---------------------------------------------------------------------------

_PLAIN_DIGIT_RE = re.compile(r"^-?\d")


def detect_text_number_columns(df: pd.DataFrame, sample_size: int = 200) -> dict:
    """
    Flags columns worth running convert_leading_word_number() over --
    same "only surface what's actually relevant" principle as Modules
    45/47's detection functions.

    A column qualifies if it shows a meaningful mix of WORD-form
    numbers (converted_word_number_rate) alongside content that's
    already numeric-looking (plain_digit_rate) -- deliberately does
    NOT require the column to already look like a weight/currency
    column (unlike Modules 45/47), since this also needs to handle a
    plain quantity column with no unit at all (e.g. "Age" with some
    rows as "fifty" instead of "50"). The combined coverage threshold
    (0.6) exists specifically to avoid flagging genuine free-text
    columns where a number word appears coincidentally (e.g. a book
    title containing "One Hundred") -- a real quantity/measurement
    column should have the VAST MAJORITY of its values look numeric
    one way or the other, which a book-title column won't.

    Returns {column_name: info_dict}:
      word_number_rate      fraction of sampled values that convert
                             successfully via convert_leading_word_number
      plain_digit_rate       fraction already starting with a digit
      example_conversions    a few (raw, converted) pairs, for the
                              confirmation question this feeds into
      n_sampled
    """
    results = {}

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        sample = sample.sample(n=min(sample_size, len(sample)), random_state=0) if len(sample) > sample_size else sample

        # Only ever consider genuinely TEXT-typed values -- a column
        # pandas already reads as numeric can't contain word-numbers
        # at all (they'd have prevented the numeric dtype in the first
        # place), so skip straight past those without even sampling.
        text_like = [v for v in sample if isinstance(v, str)]
        if not text_like:
            continue

        n = len(sample)
        plain_digit_count = sum(
            isinstance(v, (int, float)) or (isinstance(v, str) and bool(_PLAIN_DIGIT_RE.match(v.strip())))
            for v in sample
        )

        conversions = [(v, convert_leading_word_number(v)) for v in text_like]
        word_number_hits = [(raw, conv[1]) for raw, conv in conversions if conv[2]]

        word_number_rate = len(word_number_hits) / n
        plain_digit_rate = plain_digit_count / n

        if word_number_rate <= 0:
            continue  # nothing this module could do here at all
        if (word_number_rate + plain_digit_rate) < 0.6:
            continue  # doesn't look like a quantity/measurement column overall

        results[col] = {
            "word_number_rate": round(word_number_rate, 2),
            "plain_digit_rate": round(plain_digit_rate, 2),
            "example_conversions": word_number_hits[:5],
            "n_sampled": n,
        }

    return results


# ---------------------------------------------------------------------------
# 3. Apply conversion
# ---------------------------------------------------------------------------

def apply_text_number_conversion(df: pd.DataFrame, profile: dict, answers: dict) -> tuple:
    """
    For every column confirmed (convert_text_numbers__<col> = True, the
    default), rewrites word-form numbers to digits using
    convert_leading_word_number(), leaving any unit/currency text
    exactly as typed. Runs BEFORE Modules 46/48 in the pipeline -- see
    this module's own header docstring for why the ordering matters.

    "No" answers leave the column completely untouched -- same
    "respect an explicit opt-out" convention as every other
    standardize-style gate in this project.

    PROFILE PATCHING, done carefully: this step does NOT know whether
    the converted column will end up being a weight/currency column
    (unit text still attached, e.g. "50 kg") or a genuinely plain
    numeric column (no unit at all, e.g. "50") -- that's for Modules
    45/47 to determine, downstream. So profile is only patched to
    "numeric_measure" when EVERY non-null value in the column is, after
    conversion, a clean bare number with nothing else attached (the
    "Age" case) -- if any unit-like text remains (the "Weight" case),
    profile is left untouched, exactly as it was before this step, so
    Module 45/47's OWN detection and profile-patching runs normally
    when the pipeline reaches them next.

    Returns (df, actions_log).
    """
    df = df.copy()
    actions_log = []

    detected = detect_text_number_columns(df)

    for col, info in detected.items():
        if col not in df.columns:
            continue

        confirm_answer = answers.get(f"convert_text_numbers__{col}")
        if confirm_answer is False:
            actions_log.append(
                f"Left '{col}' with word-form numbers as originally written, per client's choice "
                f"not to convert them (e.g. {info['example_conversions'][:3]})"
            )
            continue

        converted_values = []
        n_converted = 0
        examples = []

        for raw in df[col]:
            if pd.isna(raw):
                converted_values.append(raw)
                continue
            # UPGRADED after real-file testing: previously used
            # convert_leading_word_number(), which only ever looks at
            # the START of a value -- "15 years nine months" (digit-
            # led, but with "nine" embedded LATER) passed straight
            # through untouched, silently losing that part of a
            # compound duration. convert_all_word_numbers_in_text()
            # catches every occurrence, not just a leading one, while
            # still using the exact same grammar-validated conversion
            # underneath (so "nineteen eighty four" is still correctly
            # left alone, for example).
            original_text = str(raw)
            converted_text = convert_all_word_numbers_in_text(original_text)
            was_converted = converted_text != original_text
            converted_values.append(converted_text)
            if was_converted:
                n_converted += 1
                if len(examples) < 5:
                    examples.append((raw, converted_text))

        df[col] = converted_values

        # Only claim "numeric_measure" if genuinely EVERY value is now
        # a bare number with nothing else attached -- otherwise leave
        # profile alone for Module 45/47 to sort out.
        non_null = df[col].dropna()
        all_bare_numeric = len(non_null) > 0 and all(
            bool(re.fullmatch(r"-?\d+(\.\d+)?", str(v).strip())) for v in non_null
        )
        if all_bare_numeric:
            # BUG FIXED after full-pipeline testing: this used to patch
            # profile[col] to "numeric" while leaving the actual pandas
            # dtype as object/string (converted_values are always
            # strings, e.g. "50", so a unit could be reattached
            # generically for the OTHER case). That mismatch -- profile
            # claiming numeric, dtype still text -- was caught by a
            # LATER validation step, which correctly (and safely)
            # skipped numeric processing for the column rather than
            # crashing, but that meant this promise was hollow. Now
            # actually casts the column to a real numeric dtype here,
            # so profile and reality agree.
            df[col] = pd.to_numeric(df[col])
            profile[col]["dtype"] = "numeric"
            profile[col]["semantic_role"] = "numeric_measure"
            profile[col]["role_confidence"] = 1.0
            profile[col]["variable_type"] = "continuous"
            profile[col]["variable_type_confidence"] = 0.9

        actions_log.append(
            f"Converted {n_converted} word-form number(s) to digits in '{col}' "
            f"(e.g. {examples}) -- any unit/currency text was kept exactly as written"
        )

    return df, actions_log



if __name__ == "__main__":
    tests = [
        "fifty",
        "fifty pounds",
        "twenty-five kg",
        "one hundred and fifty USD",
        "minus five degrees",
        "50kg",
        "50",
        "nineteen eighty four",
        "a dozen",
        "First",
        "Second",
        "two point five metres",
        "a hundred dollars",
        "seventy kg",
        "",
        None,
    ]
    print("=== convert_leading_word_number() spot checks ===")
    for t in tests:
        print(f"  {t!r:30} -> {convert_leading_word_number(t)}")

    print("\n=== detect_text_number_columns() ===")
    test_df = pd.DataFrame({
        "Weight": ["fifty kg", "seventy kg", "45kg", "sixty kg", "38kg", "fifty five kg", "62kg", "forty kg"],
        "Age": ["fifty", "32", "twenty five", "41", "nineteen", "60", "thirty three", "28"],
        "Book_Title": [
            "One Hundred Years of Solitude", "The Great Gatsby", "Nineteen Eighty-Four",
            "Fifty Shades of Grey", "War and Peace", "The Catcher in the Rye",
            "Twenty Thousand Leagues", "Pride and Prejudice",
        ],
        "Rank": ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth"],
        "Product_Code": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
    })

    print(test_df)
    print()

    detected = detect_text_number_columns(test_df)
    for col, info in detected.items():
        print(f"\nColumn: {col}")
        for k, v in info.items():
            print(f"  {k}: {v}")

    print("\n=== Columns NOT flagged (should be Book_Title, Rank, Product_Code) ===")
    not_flagged = [c for c in test_df.columns if c not in detected]
    print(not_flagged)
