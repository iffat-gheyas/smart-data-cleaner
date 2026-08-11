"""
Module: Email Standardisation -- Sub-step A (Detection & Parsing)
----------------------------------------------------------------------
Sibling to Module 52 (phone). Uses the `email_validator` library for
RFC-aware syntax checking and normalisation, rather than a hand-rolled
regex -- email address syntax has enough real edge cases (quoted local
parts, internationalised domains, etc.) that a well-tested library is
the right tool, same reasoning as Module 52's use of phonenumbers.

DNS/deliverability checking is explicitly DISABLED
(check_deliverability=False) -- this module only validates SYNTAX
(is this a well-formed address), never whether the mailbox actually
exists. Checking deliverability would mean a live DNS lookup per
value during cleaning, which is slow, requires network access, and
tells you nothing about whether cleaning worked -- an address can be
perfectly well-formed and syntactically standardised even if the
mailbox turns out not to exist.

CASING NUANCE, worth being explicit about (the client's own spec asks
for this to be optional, for good reason): the DOMAIN part of an email
address is always case-insensitive and safe to lowercase
unconditionally. The LOCAL part (before the @) is, strictly per the
email spec, case-SENSITIVE -- "John.Smith@x.com" and "john.smith@x.com"
are technically different addresses, even though virtually every real
mail provider treats them the same in practice. This module therefore
offers domain-only lowercasing as the "safe, always-correct" option,
and whole-address lowercasing as a separate, explicit choice -- never
picked silently.
"""

import re

import pandas as pd
from email_validator import validate_email, EmailNotValidError

TREAT_AS_MISSING = "__treat_as_missing__"

STRONG_EMAIL_NAME_KEYWORDS = ["email", "e-mail", "e_mail"]
WEAK_EMAIL_NAME_KEYWORDS = ["mail", "contact"]

# Cheap pre-filter before invoking the real validator -- just "does this
# look roughly like local@domain.tld at all".
_LOOKS_LIKE_EMAIL_RE = re.compile(r"^\S+@\S+\.\S+$")


def _looks_email_like(raw) -> bool:
    if pd.isna(raw):
        return False
    text = str(raw).strip()
    # Same whitespace-stripping parse_email_value now applies (see its
    # docstring) -- detection and parsing need to agree on what counts
    # as "close enough", or a column with several space-riddled
    # addresses could fail to be detected as an email column at all.
    text = re.sub(r"\s+", "", text)
    return bool(_LOOKS_LIKE_EMAIL_RE.match(text))


# ---------------------------------------------------------------------------
# 1. Single-value parsing / validation
# ---------------------------------------------------------------------------

def parse_email_value(raw, lowercase_local_part: bool = False) -> dict:
    """
    Attempts to validate and normalise a single email value.

    `lowercase_local_part`: if False (default), only the domain is
    lowercased (always technically correct). If True, the WHOLE
    address is lowercased (matches real-world provider behaviour, but
    not strictly spec-guaranteed) -- an explicit client choice, never
    the silent default.

    PRE-PROCESSING, UPDATED per explicit client request after testing
    against a real messy file: ALL whitespace anywhere in the value is
    now stripped before validation, not just whitespace touching the
    "@" (the original, narrower rule). Real data showed whitespace
    scattered throughout -- inside the domain ("Hotmail.Co. Uk",
    "YAHOO.  CO.   UK"), inside the local part ("SARAH. Williamss"),
    even doubled ("hasrris.  Jackson") -- and the client's instruction
    was explicit: remove it.

    WORTH BEING HONEST ABOUT: this is a genuine behaviour change from
    the earlier design (which deliberately treated "john smith@x.com"
    -- a space INSIDE a word, away from the "@" -- as a real ambiguity
    needing manual review, since the space might have been meant as a
    "." or shouldn't be there at all). Blanket-stripping now turns
    that into "johnsmith@x.com" automatically. That's the explicitly
    requested behaviour, but it means a rare case where the space
    really did belong (unlikely for an email address, but not
    impossible) would no longer get flagged for a human to check.

    Returns a dict:
      raw             the original value
      is_valid        True if this is a syntactically well-formed
                      address
      normalized      the standardised address (all whitespace
                      removed, domain lowercased) if valid, else None
      reason          human-readable explanation when NOT valid
    """
    result = {"raw": raw, "is_valid": False, "normalized": None, "reason": None}

    if pd.isna(raw):
        result["reason"] = "empty value"
        return result

    text = str(raw).strip()
    if text == "":
        result["reason"] = "empty value"
        return result

    text = re.sub(r"\s+", "", text)

    try:
        validated = validate_email(text, check_deliverability=False)
    except EmailNotValidError as e:
        result["reason"] = str(e)
        return result

    result["is_valid"] = True
    normalized = validated.normalized
    if lowercase_local_part:
        normalized = normalized.lower()
    result["normalized"] = normalized
    return result


# ---------------------------------------------------------------------------
# 2. Column-level detection
# ---------------------------------------------------------------------------

def _name_signal(col_name: str) -> dict:
    name_lower = col_name.lower()
    strong = any(kw in name_lower for kw in STRONG_EMAIL_NAME_KEYWORDS)
    weak = any(kw in name_lower for kw in WEAK_EMAIL_NAME_KEYWORDS)
    return {"strong": strong, "weak": weak}


def detect_email_columns(df: pd.DataFrame, sample_size: int = 200) -> dict:
    """
    Returns {column_name: info_dict} for columns worth treating as
    email addresses. Unlike phone numbers, email syntax validity
    doesn't depend on any external context (no "region" needed), so
    this DOES run the real validator during detection -- a much
    simpler situation than Module 52's.

    info_dict:
      confidence          0.0-1.0, same thresholds as other modules
      valid_count / invalid_count / n_sampled
      invalid_examples    a few (raw, reason) pairs for problem values
    """
    results = {}

    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        sample = sample.sample(n=min(sample_size, len(sample)), random_state=0) if len(sample) > sample_size else sample

        email_like = [v for v in sample if _looks_email_like(v)]
        content_rate = len(email_like) / len(sample)

        name_sig = _name_signal(str(col))

        is_email_column = False
        confidence = 0.0
        # BUG FIXED after testing: this used to REQUIRE content_rate >=
        # 0.6 even for a column literally named "Email" -- meaning a
        # column with several genuinely malformed addresses (exactly
        # the realistic, messy-data case this whole feature exists
        # for) could fail to be recognised as an email column AT ALL,
        # so none of the standardisation/review mechanism ever ran.
        # A strong name match ("Email", "E-mail") is now enough on its
        # own to flag the column -- confidence still scales with how
        # much of the content actually looks right, so a genuinely
        # empty or unrelated "Email"-named column still gets a lower
        # score (and, below EMAIL_CONFIRMATION_THRESHOLD, a "is this
        # really email?" confirmation question) rather than being
        # silently assumed either way.
        if name_sig["strong"]:
            is_email_column = True
            if content_rate >= 0.6:
                confidence = 0.95
            elif content_rate >= 0.3:
                confidence = 0.75
            else:
                confidence = 0.5
        elif content_rate >= 0.6 and name_sig["weak"]:
            is_email_column = True
            confidence = 0.75
        elif content_rate >= 0.8:
            is_email_column = True
            confidence = 0.6

        if not is_email_column:
            continue

        parsed = [parse_email_value(v) for v in sample]
        valid_count = sum(1 for p in parsed if p["is_valid"])
        invalid_examples = [(p["raw"], p["reason"]) for p in parsed if not p["is_valid"]][:5]

        results[col] = {
            "confidence": confidence,
            "valid_count": valid_count,
            "invalid_count": len(sample) - valid_count,
            "n_sampled": len(sample),
            "invalid_examples": invalid_examples,
        }

    return results


if __name__ == "__main__":
    print("=== parse_email_value() spot checks ===")
    tests = [
        "JOHN.SMITH@GMAIL.COM",
        " JOHN.SMITH @ GMAIL.COM ".replace(" @ ", "@"),
        "john.gmail.com",
        "john@",
        "@gmail.com",
        "john smith@gmail.com",
    ]
    for t in tests:
        print(f"  {t!r:30} -> {parse_email_value(t)}")
        print(f"  {t!r:30} (lowercase local) -> {parse_email_value(t, lowercase_local_part=True)}")

    print("\n=== detect_email_columns() ===")
    test_df = pd.DataFrame({
        "Email": ["JOHN.SMITH@GMAIL.COM", "jane.doe@yahoo.com", "bob@lee", "sara@khan.com", "tom hill@x.com", "ann@bell.co.uk"],
        "Website": ["example.com", "test.org", "site.net", "shop.io", "blog.co", "page.dev"],
        "Customer_Name": ["John Smith", "Jane Doe", "Bob Lee", "Sara Khan", "Tom Hill", "Ann Bell"],
    })
    detected = detect_email_columns(test_df)
    for col, info in detected.items():
        print(f"\nColumn: {col}")
        for k, v in info.items():
            print(f"  {k}: {v}")

    print("\n=== Columns NOT flagged (should be Website, Customer_Name) ===")
    print([c for c in test_df.columns if c not in detected])
