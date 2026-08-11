"""
Module: Currency Standardisation -- Sub-step A (Detection & Parsing)
----------------------------------------------------------------------
Generic-mode feature, sibling to Modules 45/46 (weight & length). Same
overall shape -- detect which columns are money, parse each cell into
(amount, currency_code), later ask the client which currency to
standardise into -- but ONE deliberate difference drives most of the
design choices in this file:

CURRENCY CONVERSION RATES ARE NOT FIXED CONSTANTS. A kilogram is always
0.001 tonnes, forever -- that ratio can be hardcoded once. USD-to-GBP
is a different number every day. This module therefore only detects
and PARSES currency values; it never converts anything itself, and the
later conversion step (a follow-on module, not built yet) will need
the client to SUPPLY a rate rather than picking one from a fixed
table -- the currency-specific extra step the client asked for.

SECOND design difference from Module 45, worth being explicit about:
currency symbols conventionally come BEFORE the number ("$100"), while
weight/length units come AFTER ("100kg"). parse_currency_value() below
checks for a known PREFIX first (symbols like $/£/€, or compound ones
like "C$", "A$"), then falls back to the SUFFIX pattern (word-form
codes/names like "100 USD", "500 rupees") -- both paths feed into the
same (amount, currency_code, raw_token) result shape Module 46 already
established, so the eventual conversion module can reuse that pattern
directly.

SCOPE NOTE: covers roughly 25 major world currencies (see
CURRENCY_ALIASES) -- not the full ISO 4217 list. Easy to extend the
alias table if a client file needs one that's missing; flagged here
rather than silently guessed if it's ever hit (see
unrecognised_currency_samples in detect_currency_columns).
"""

import re
import difflib

import pandas as pd

from importlib import import_module

# Reuses the SAME "I don't know, treat as missing" sentinel Module 45
# defined for the weight/length unresolved-unit question, rather than
# each feature inventing its own -- one shared concept, one definition.
unit_std = import_module("45_unit_standardisation")
TREAT_AS_MISSING = unit_std.TREAT_AS_MISSING
LEAVE_AS_IS = unit_std.LEAVE_AS_IS


# ---------------------------------------------------------------------------
# 1. Currency alias tables
# ---------------------------------------------------------------------------

# PREFIX aliases -- symbols and compound-symbol tokens that appear
# BEFORE the number ("$100", "C$50", "Rs. 500"). Checked longest-first
# by the parser (see _PREFIX_TOKENS_BY_LENGTH below) so "c$" matches
# before the bare "$" inside it would otherwise short-circuit to.
CURRENCY_PREFIX_ALIASES = {
    "us$": "USD", "$": "USD",
    "c$": "CAD",
    "a$": "AUD",
    "nz$": "NZD",
    "hk$": "HKD",
    "s$": "SGD",
    "r$": "BRL",
    "£": "GBP",
    "€": "EUR",
    "¥": "JPY",
    "₹": "INR", "rs.": "INR", "rs": "INR",
    "₩": "KRW",
    "₽": "RUB",
    "₺": "TRY",
    "₦": "NGN",
    "₫": "VND",
}

# SUFFIX aliases -- 3-letter ISO codes and common word forms that
# appear AFTER the number ("100 USD", "500 rupees", "100kr" is
# deliberately NOT included, see ambiguity note below).
CURRENCY_SUFFIX_ALIASES = {
    "usd": "USD", "us dollars": "USD", "dollars": "USD", "dollar": "USD",
    "gbp": "GBP", "pounds sterling": "GBP", "pounds": "GBP", "pound": "GBP",
    "eur": "EUR", "euros": "EUR", "euro": "EUR",
    "jpy": "JPY", "yen": "JPY",
    "inr": "INR", "rupees": "INR", "rupee": "INR",
    "krw": "KRW", "won": "KRW",
    "rub": "RUB", "rubles": "RUB", "ruble": "RUB",
    "try": "TRY", "lira": "TRY",
    "cad": "CAD",
    "aud": "AUD",
    "nzd": "NZD",
    "hkd": "HKD",
    "sgd": "SGD",
    "chf": "CHF", "swiss francs": "CHF", "francs": "CHF",
    "cny": "CNY", "rmb": "CNY", "yuan": "CNY",
    "zar": "ZAR", "rand": "ZAR",
    "brl": "BRL", "reais": "BRL", "real": "BRL",
    "mxn": "MXN", "pesos": "MXN", "peso": "MXN",
    "sek": "SEK",
    "nok": "NOK",
    "dkk": "DKK",
    "pln": "PLN", "zloty": "PLN",
    "aed": "AED", "dirhams": "AED",
    "sar": "SAR", "riyals": "SAR",
    "ngn": "NGN", "naira": "NGN",
    "vnd": "VND", "dong": "VND",
}

ALL_CURRENCY_CODES = set(CURRENCY_PREFIX_ALIASES.values()) | set(CURRENCY_SUFFIX_ALIASES.values())

# Combined lookup used by BOTH the prefix scan and the suffix fallback
# below -- real files mix conventions ("$100" and "USD 100" and even
# "100USD" can all appear in the same column), so a token recognised on
# one side should be recognised on the other too, rather than each
# alias dict only working in the position it was originally written
# for. BUG FIXED after testing: "USD 4200" (a 3-letter CODE used as a
# prefix, not just a symbol) used to fall through both checks entirely
# and return (None, None, None) -- losing the number, not just the
# currency, which is a worse failure than an unrecognised token would
# be. Checking every known token (symbol OR code OR word) as a
# potential prefix fixes this.
ALL_CURRENCY_TOKENS = {**CURRENCY_PREFIX_ALIASES, **CURRENCY_SUFFIX_ALIASES}

# NOTE on ambiguity, same "be explicit rather than silently guess"
# principle as Module 45's ton/tonne note:
# - Bare "$" defaults to USD. It's ALSO used natively by CAD, AUD, SGD,
#   HKD, NZD, MXN and others -- but each of those has its own distinct
#   compound prefix here (C$, A$, S$, HK$, NZ$) that takes priority
#   when present, so bare "$" only falls through to USD when there's no
#   more specific marker. A file that uses UNPREFIXED "$" to mean, say,
#   Mexican pesos would be silently mis-read as USD -- worth a quick
#   sanity check with the client if a file's origin suggests this.
# - Bare "kr" (Swedish/Norwegian/Danish krona/krone) is deliberately
#   NOT included at all -- unlike "$", there's no separate compound
#   marker to disambiguate SEK/NOK/DKK from each other, so guessing any
#   one of them would be a real coin-flip. A value like "100 kr" will
#   come through as an UNRECOGNISED currency token instead, which
#   surfaces the ambiguity as a question rather than a silent guess.
# - "¥" defaults to JPY, not CNY, despite both using the same symbol --
#   CNY has its own unambiguous 3-letter codes (CNY/RMB/yuan) as a
#   suffix, so this only affects a bare "¥" with no other cue.

# Column-NAME keyword signals, same STRONG/WEAK split as Module 45 and
# for the same reason -- "amount" or "total" alone is a much weaker
# signal than "price" or "salary", since the former show up on
# obviously non-monetary columns too (e.g. "Amount_of_Items").
STRONG_CURRENCY_NAME_KEYWORDS = ["price", "cost", "salary", "revenue", "payment", "fee", "income", "fare"]
WEAK_CURRENCY_NAME_KEYWORDS = ["amount", "total", "value", "charge", "budget", "spend", "currency"]


# ---------------------------------------------------------------------------
# 2. Single-value quantity parser
# ---------------------------------------------------------------------------

# Prefix tokens, longest first, so "c$" is tried before the bare "$"
# that's a substring of it would otherwise match instead. Scans the
# MERGED token dict (symbols AND codes/words), not just the
# symbol-only prefix dict -- see the ALL_CURRENCY_TOKENS note above.
_PREFIX_TOKENS_BY_LENGTH = sorted(ALL_CURRENCY_TOKENS.keys(), key=len, reverse=True)

# Same loosened comma-grouping fix as Module 45 (accepts ANY comma
# grouping, not just strict Western 3-digit groups, then strips commas
# entirely) -- money amounts are exactly as likely to hit the
# Indian-lakh-style "5,50,500" case as any other number.
_LEADING_NUMBER_RE = re.compile(r"^\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*(.*)$")


def parse_currency_value(raw) -> tuple:
    """
    Extracts an (amount, currency_code, raw_token) triple from a single
    cell value -- same shape as Module 45's parse_quantity(), so the
    eventual conversion module can reuse that module's patterns
    directly.

      "$100"      -> (100.0, "USD", "$")
      "100 USD"   -> (100.0, "USD", "usd")
      "€50.5"     -> (50.5,  "EUR", "€")
      "Rs. 500"   -> (500.0, "INR", "rs.")
      "42"        -> (42.0,  None,  None)     -- bare number, no currency
      "100 kr"    -> (100.0, None,  "kr")     -- currency-like token, but
                                                  ambiguous / unrecognised
                                                  (see the "kr" note above)
      "N/A"       -> (None,  None,  None)     -- no number found at all

    Checks for a PREFIX symbol/token first (currency symbols
    conventionally precede the amount), then falls back to the SUFFIX
    pattern (word-form codes/names conventionally follow it) -- the
    reverse check order from Module 45, which only ever needed to
    check suffixes, since weight/length units always come after the
    number.
    """
    if pd.isna(raw):
        return (None, None, None)

    if isinstance(raw, (int, float)):
        return (float(raw), None, None)

    text = str(raw).strip()
    if text == "":
        return (None, None, None)

    # BUG FIXED after real-file testing: a currency code wrapped in
    # parentheses or brackets ("70 (JPY)") was falling through
    # entirely -- the closing "(" and ")" characters became part of
    # the remainder ("(jpy)"), which doesn't match the alias table's
    # bare "jpy" entry. Stripping these wrapper characters up front
    # (not just trimming whitespace) fixes this regardless of where
    # the code sits: "70 (JPY)" and "(JPY) 70" both now work.
    # KNOWN MINOR LIMITATION: this also strips genuine accounting
    # negative-number notation, e.g. "(100)" meaning -100 -- rare
    # enough in this context (currency amounts in a cleaning pipeline)
    # that recognising currency codes correctly was judged the more
    # valuable default; flag if accounting-style negatives ever show
    # up in a real client file.
    text = re.sub(r"[()\[\]]", "", text).strip()
    if text == "":
        return (None, None, None)

    # --- Try a known PREFIX first ---
    text_lower = text.lower()
    for token in _PREFIX_TOKENS_BY_LENGTH:
        if text_lower.startswith(token):
            remainder = text[len(token):].strip()
            match = _LEADING_NUMBER_RE.match(remainder)
            if match:
                number_str = match.group(1)
                try:
                    number = float(number_str.replace(",", ""))
                    return (number, ALL_CURRENCY_TOKENS[token], token)
                except ValueError:
                    pass  # fall through to the generic path below

    # --- No recognised prefix -- try NUMBER-then-SUFFIX instead ---
    match = _LEADING_NUMBER_RE.match(text)
    if not match:
        return (None, None, None)

    number_str, remainder = match.groups()
    try:
        number = float(number_str.replace(",", ""))
    except ValueError:
        return (None, None, None)

    remainder = remainder.strip().lower().rstrip(".")
    if not remainder:
        return (number, None, None)

    canonical = ALL_CURRENCY_TOKENS.get(remainder)
    return (number, canonical, remainder)


# ---------------------------------------------------------------------------
# 3. Column-level detection
# ---------------------------------------------------------------------------

def _name_signal(col_name: str) -> dict:
    name_lower = col_name.lower()
    strong = any(kw in name_lower for kw in STRONG_CURRENCY_NAME_KEYWORDS)
    weak = any(kw in name_lower for kw in WEAK_CURRENCY_NAME_KEYWORDS)

    # A currency code/symbol embedded directly in the column name
    # itself (e.g. "Price_USD", "Salary (GBP)") tells us the IMPLIED
    # currency for every bare number in that column -- same pattern as
    # Module 45's implied_unit_from_name.
    implied_currency = None
    for token, code in {**CURRENCY_PREFIX_ALIASES, **CURRENCY_SUFFIX_ALIASES}.items():
        if len(token) <= 1:
            continue  # skip bare symbols like "$" for name matching -- too noisy
        if re.search(rf"(?:^|[_\s(]){re.escape(token)}(?:$|[_\s)])", name_lower):
            implied_currency = code
            break

    return {"strong": strong, "weak": weak, "implied_currency_from_name": implied_currency}


def detect_currency_columns(df: pd.DataFrame, sample_size: int = 200) -> dict:
    """
    Returns {column_name: info_dict}, same "only surface columns that
    actually look monetary" principle as Module 45's
    detect_measurement_columns() -- ordinary columns (IDs, names,
    plain counts) are simply absent from the result.

    info_dict contains:
      confidence               0.0-1.0, same thresholds/reasoning as
                                Module 45
      detected_currencies      {code: count} from values that DID carry
                                a recognisable currency token
      implied_currency_from_name  currency implied by the column name
                                itself, or None
      bare_number_count / bare_number_examples   values with no
                                currency token at all
      unrecognised_currency_samples   values with a currency-like token
                                we don't recognise (e.g. "100 kr")
      n_sampled
    """
    results = {}

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        sample = sample.sample(n=min(sample_size, len(sample)), random_state=0) if len(sample) > sample_size else sample

        parsed = [parse_currency_value(v) for v in sample]
        n = len(parsed)

        numbers = [p[0] for p in parsed]
        codes = [p[1] for p in parsed]

        numeric_rate = sum(nm is not None for nm in numbers) / n
        currency_hits = [c for c in codes if c in ALL_CURRENCY_CODES]
        currency_rate = len(currency_hits) / n

        bare_number_count = sum(
            nm is not None and code is None and tok is None
            for nm, code, tok in parsed
        )
        bare_number_examples = [
            raw for raw, (nm, code, tok) in zip(sample, parsed)
            if nm is not None and code is None and tok is None
        ][:5]
        unrecognised_currency_samples = [
            raw for raw, (nm, code, tok) in zip(sample, parsed)
            if nm is not None and tok is not None and code is None
        ][:5]

        name_sig = _name_signal(str(col))

        confidence = 0.0
        is_currency_column = False

        # --- Decision order, strongest evidence first (mirrors Module 45) ---
        if currency_rate >= 0.3:
            is_currency_column = True
            confidence = 0.95 if currency_rate >= 0.6 else 0.75
        elif numeric_rate >= 0.8 and name_sig["strong"]:
            is_currency_column = True
            confidence = 0.6
        elif numeric_rate >= 0.8 and name_sig["weak"]:
            is_currency_column = True
            confidence = 0.35

        if not is_currency_column:
            continue

        detected_currencies = {}
        for c in currency_hits:
            detected_currencies[c] = detected_currencies.get(c, 0) + 1

        results[col] = {
            "confidence": round(confidence, 2),
            "detected_currencies": detected_currencies,
            "implied_currency_from_name": name_sig["implied_currency_from_name"],
            "bare_number_count": bare_number_count,
            "bare_number_examples": bare_number_examples,
            "unrecognised_currency_samples": unrecognised_currency_samples,
            "n_sampled": n,
        }

    return results


def needs_standardization(info: dict) -> bool:
    """
    Same shared trigger condition as Module 45's function of the same
    name, and used the same way: a currency column needs conversion
    work if it has ANY recognised currency token in its values (mixed
    OR single-currency-as-text), or bare numbers with no name-implied
    currency to fall back on.
    """
    return bool(info["detected_currencies"]) or (
        info["bare_number_count"] > 0 and not info.get("implied_currency_from_name")
    )


# ---------------------------------------------------------------------------
# 4. Per-value unresolved-currency detail (for case-by-case resolution)
# ---------------------------------------------------------------------------
# Added after real-file testing showed the ORIGINAL "one shared question
# per column" design breaking down: a column can have several DIFFERENT
# unresolved values that each need a DIFFERENT answer (a typo'd code, a
# bare number, an unfamiliar symbol) -- assuming one blanket currency
# applies to all of them is often simply wrong. These two functions
# support asking about each DISTINCT problem value individually, with
# the row it came from as a clue and a fuzzy-matched suggestion where
# a real typo is likely.

# Candidate tokens for fuzzy matching -- WORD-FORM tokens only (3+
# characters). Bare single/two-character symbols ($, £, €, kr) are
# excluded on purpose: fuzzy-matching something like "3" or "rs"
# against a huge candidate pool produces nonsense matches far too
# easily, whereas a genuine typo of a currency CODE or WORD ("gpb",
# "dolars") is long enough for similarity matching to be meaningful.
_FUZZY_CANDIDATES = sorted(
    {token for token in ALL_CURRENCY_TOKENS if len(token) >= 3}
)


def _suggest_likely_currency(raw_token: str):
    """
    Best-effort fuzzy suggestion for an unrecognised currency-like
    token, e.g. "gpb" -> "GBP" (transposed letters -- a very plausible
    typo). Returns a currency CODE (for use as a dropdown default) or
    None if nothing is a close enough match.

    NEVER auto-applied -- this only ever supplies a PRE-FILLED DEFAULT
    for a question the client still has to see and confirm/override,
    same "suggest, don't silently guess" principle as every other
    default in this project. cutoff=0.6 was chosen specifically because
    it catches "gpb"/"gbp" (ratio ~0.67, a letter transposition) while
    still requiring real similarity -- tightened rather than loosened
    if it ever starts suggesting implausible matches on real data.
    """
    if not raw_token or len(raw_token) < 3:
        return None

    matches = difflib.get_close_matches(raw_token.lower(), _FUZZY_CANDIDATES, n=1, cutoff=0.6)
    if not matches:
        return None

    return ALL_CURRENCY_TOKENS[matches[0]]


def find_unresolved_currency_details(df: pd.DataFrame, col: str, info: dict) -> list:
    """
    Enumerates every DISTINCT problem value in `col` (bare numbers with
    no name-implied currency to fall back on, OR values with a
    currency-like token we don't recognise) -- NOT capped at a handful
    of samples like detect_currency_columns()'s bare_number_examples /
    unrecognised_currency_samples, which exist only to illustrate the
    problem in a column-level summary. This function is the full list,
    meant to drive one question PER distinct value.

    Groups by distinct RAW VALUE, not by row -- if "60 GPB" appears in
    3 rows, that's one entry with count=3, not three separate
    questions, since the same typo almost certainly means the same
    thing everywhere it appears. Deliberately NOT grouped further than
    that (e.g. not merged with other unresolved values), since two
    DIFFERENT problem values very plausibly need two DIFFERENT answers
    -- that's the whole point of this function existing.

    Returns a list of dicts, one per distinct unresolved value:
      raw_value            the original cell value
      count                 how many rows contain this exact value
      example_row_index     index of the FIRST row containing it
      example_row           that row as a dict {column: value}, for
                             context -- "the clue can be in the cell"
                             (e.g. a Country column hinting at the
                             right currency)
      suggested_currency    a fuzzy-match guess (see
                             _suggest_likely_currency), or None
      is_bare               True if NO currency-like text was present
                             at all (e.g. "50") -- False if there WAS
                             a currency-like token, just an
                             unrecognised one (e.g. "60 GPB", a typo).
                             This distinction matters: a client-supplied
                             DEFAULT currency (Module 11's has_default_
                             currency question) only ever applies to
                             is_bare=True values -- an ambiguous/typo'd
                             token is a different kind of problem a
                             blanket default can't safely resolve, and
                             always needs case-by-case attention
                             regardless of whether a default exists.
    """
    implied_currency = info.get("implied_currency_from_name")
    if implied_currency:
        return []  # bare numbers already resolved via the implied currency -- nothing ambiguous

    series = df[col]
    seen = {}  # raw_value (as string key) -> detail dict, preserving first-seen order

    for idx, raw in series.items():
        if pd.isna(raw):
            continue
        number, currency_code, raw_token = parse_currency_value(raw)
        if number is None or currency_code is not None:
            continue  # either unparseable (genuinely missing) or already resolved

        key = str(raw)
        if key not in seen:
            example_row = df.loc[idx].to_dict()
            seen[key] = {
                "raw_value": raw,
                "count": 0,
                "example_row_index": idx,
                "example_row": example_row,
                "suggested_currency": _suggest_likely_currency(raw_token) if raw_token else None,
                "is_bare": raw_token is None,
            }
        seen[key]["count"] += 1

    return list(seen.values())


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "Price": ["$100", "150 USD", "89.99", "€75", "£60", "120", "R$45", "100 kr"],
        "Salary_GBP": [50000, 62000, 48000, "£55,000", 71000, None, 49500, 60000],
        "Product_Code": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
        "Total_Items": [4, 3, 2, 5, 4, 3, 2, 1],  # decoy -- weak keyword "total" but not monetary
        "Revenue": ["1,000", "2,500", "3,00,000", "5000", "USD 4200", "€3100", "1500", "2200"],
    })

    print("=== Test dataframe ===")
    print(test_df)

    print("\n=== parse_currency_value() spot checks ===")
    for val in ["$100", "100 USD", "€50.5", "Rs. 500", "42", "100 kr", "N/A", "C$75", "5,50,000"]:
        print(f"  {val!r:15} -> {parse_currency_value(val)}")

    print("\n=== detect_currency_columns() ===")
    detected = detect_currency_columns(test_df)
    for col, info in detected.items():
        print(f"\nColumn: {col}")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print(f"  needs_standardization: {needs_standardization(info)}")

    print("\n=== Columns NOT flagged (should be Product_Code) ===")
    not_flagged = [c for c in test_df.columns if c not in detected]
    print(not_flagged)
