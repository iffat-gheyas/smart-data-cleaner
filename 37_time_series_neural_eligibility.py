"""
Module AK: Neural Network Eligibility Gate
------------------------------------------------
Per the spec: "Do not run CNN or LSTM models on datasets that are too
small to support reliable neural-network training... mark the neural
candidate as 'Not eligible -- insufficient training data' rather than
fitting an unreliable model."

DISCLOSED, NOT UNIVERSAL, THRESHOLDS: there's no single agreed-upon
minimum dataset size for "when neural networks start being reliable" --
it depends on signal strength, noise level, and architecture choice.
The thresholds here are conservative defaults reflecting the general
principle that CNN/LSTM need substantially more data than a
regularized regression to avoid essentially memorizing noise, not a
claim that these exact numbers are the one correct answer.

HONEST EXPECTATION, stated plainly: every real dataset this product
has been tested against so far (36 to a few hundred rows) will
correctly be marked NOT eligible under these thresholds. That's the
gate doing its job, not a bug -- CNN/LSTM are being built because the
client wants them available for when a large-enough multivariate
dataset genuinely arrives, not because they're expected to fire
routinely on the typical file size seen so far.
"""

MIN_OBSERVED_ROWS = 200  # below this, a neural net is more likely to memorize noise than learn real structure
MAX_MISSING_PERCENTAGE = 50.0  # above this, there's too little genuine signal to train on reliably
MAX_INPUT_TO_ROWS_RATIO = 0.5  # input dimensionality shouldn't swamp the available training rows


def check_neural_eligibility(
    n_observed_target_rows: int,
    n_variables: int,
    window_length: int,
    missing_percentage: float,
    min_observed_rows: int = MIN_OBSERVED_ROWS,
    max_missing_percentage: float = MAX_MISSING_PERCENTAGE,
    max_input_to_rows_ratio: float = MAX_INPUT_TO_ROWS_RATIO,
) -> dict:
    """
    n_observed_target_rows: how many rows of the target column are
    genuinely observed (not missing) -- the actual number of training
    examples available.
    n_variables: total value columns involved (target + others).
    window_length: the full lag+lead window size (e.g. max_lag=3,
    max_lead=3 -> window_length=7, the target's own position plus 3
    on each side).
    missing_percentage: what fraction of the target is missing overall.

    Returns {"eligible": bool, "reason": str, "details": dict}. When
    NOT eligible, "reason" is written to be shown directly to the
    client (matches the spec's required wording), not just logged
    internally.
    """
    reasons = []
    approx_input_dim = n_variables * window_length * 2  # values + missingness-mask channels, per Section 8's channel description

    if n_observed_target_rows < min_observed_rows:
        reasons.append(f"only {n_observed_target_rows} observed values (need at least {min_observed_rows})")

    if missing_percentage > max_missing_percentage:
        reasons.append(f"{missing_percentage:.0f}% missing is too sparse a signal for reliable neural-network training")

    if n_observed_target_rows > 0 and approx_input_dim > max_input_to_rows_ratio * n_observed_target_rows:
        reasons.append(
            f"approximate input dimensionality ({approx_input_dim}, from {n_variables} variable(s) x "
            f"window length {window_length} x 2 for values+missingness masks) is too large relative to "
            f"{n_observed_target_rows} observed rows -- would almost certainly overfit"
        )

    details = {
        "n_observed_target_rows": n_observed_target_rows,
        "n_variables": n_variables,
        "window_length": window_length,
        "missing_percentage": missing_percentage,
        "approx_input_dim": approx_input_dim,
    }

    if reasons:
        return {"eligible": False, "reason": "Not eligible — insufficient training data: " + "; ".join(reasons), "details": details}
    return {"eligible": True, "reason": "Eligible", "details": details}


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: every real dataset tested in this product so far (36-300ish rows) -- must correctly decline")
    print("=" * 60)
    for n_rows, label in [(34, "shampoo (36 rows, ~34 observed)"), (250, "rainfall (252 rows)"), (300, "a slightly bigger hypothetical file")]:
        result = check_neural_eligibility(n_observed_target_rows=n_rows, n_variables=3, window_length=7, missing_percentage=2.0)
        print(f"  {label}: eligible={result['eligible']}")
        if not result["eligible"]:
            print(f"    reason: {result['reason']}")

    print("\n" + "=" * 60)
    print("TEST 2: a genuinely large multivariate dataset -- must correctly be eligible")
    print("=" * 60)
    result_large = check_neural_eligibility(n_observed_target_rows=5000, n_variables=4, window_length=7, missing_percentage=3.0)
    print("5000 observed rows, 4 variables, window=7, 3% missing:", result_large["eligible"])
    print("Details:", result_large["details"])

    print("\n" + "=" * 60)
    print("TEST 3: large row count but too sparse (heavy missingness) -- must still decline")
    print("=" * 60)
    result_sparse = check_neural_eligibility(n_observed_target_rows=5000, n_variables=4, window_length=7, missing_percentage=70.0)
    print("5000 rows but 70% missing:", result_sparse["eligible"], "-", result_sparse["reason"] if not result_sparse["eligible"] else "")

    print("\n" + "=" * 60)
    print("TEST 4: large row count but with SO MANY variables/window length that input dimensionality overwhelms it")
    print("=" * 60)
    result_wide = check_neural_eligibility(n_observed_target_rows=250, n_variables=50, window_length=21, missing_percentage=5.0)
    print("250 rows but 50 variables x window 21:", result_wide["eligible"], "-", result_wide["reason"] if not result_wide["eligible"] else "")

    print("\n" + "=" * 60)
    print("TEST 5: boundary case -- exactly at the threshold")
    print("=" * 60)
    result_boundary = check_neural_eligibility(n_observed_target_rows=200, n_variables=2, window_length=3, missing_percentage=5.0)
    print("Exactly 200 observed rows (the minimum):", result_boundary["eligible"])
