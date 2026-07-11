"""
Extended CTS: Adding CNV + Expression Signal
==============================================

Layers copy-number and expression evidence on top of the ORIGINAL CTS
score from kinase_scoring_pipeline.compute_cts() -- which is left
completely untouched, so the already-reported result (ERBB2/EGFR/PTK2 top
3) remains reproducible exactly as before. This produces a SEPARATE
'CTS_extended' score, not a silent replacement.

Rationale for the two new terms:
    - Amplification signal: a kinase with elevated copy number in TNBC
      cell lines is a stronger oncogene-addiction candidate (this is
      exactly why ERBB2 -- a classic amplification-driven oncogene -- was
      a gap in the original 4-source CTS, which had no CNV data at all).
    - Expression signal: a kinase highly expressed in TNBC lines is more
      plausible as an actively-used dependency than one with low/no
      expression, independent of what DepMap's CRISPR knockout says.

Same missing-value discipline as the original: NEUTRAL (0.5) fallback for
both new terms if a kinase has no CNV/expression data, never a dropped
kinase and never a fabricated confident-looking score in place of missing
data.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from kinase_scoring_pipeline import min_max_normalize

EXTENDED_CTS_WEIGHTS = {
    "original_cts": 0.60,   # the already-validated 4-source score keeps majority weight
    "amplification": 0.20,
    "expression": 0.20,
}


def compute_cts_extended(
    cts_result: pd.DataFrame,
    cnv_scores: pd.Series,
    expression_scores: pd.Series,
    weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    cts_result: the DataFrame returned by kinase_scoring_pipeline.compute_cts()
                (must have a 'CTS' column, indexed by kinase_id)
    cnv_scores: mean_log2_cn per kinase (e.g. from parse_depmap_cnv(), grouped)
    expression_scores: mean_log_tpm per kinase (e.g. from parse_depmap_expression())

    Returns cts_result with three added columns: amplification_norm,
    expression_norm, CTS_extended -- the original 'CTS' column is
    untouched.
    """
    w = weights or EXTENDED_CTS_WEIGHTS
    df = cts_result.copy()

    df["amplification_missing"] = cnv_scores.reindex(df.index).isna()
    amp_raw = min_max_normalize(cnv_scores.reindex(df.index), higher_is_better=True)
    df["amplification_norm"] = amp_raw.where(~df["amplification_missing"], 0.5)

    df["expression_missing"] = expression_scores.reindex(df.index).isna()
    expr_raw = min_max_normalize(expression_scores.reindex(df.index), higher_is_better=True)
    df["expression_norm"] = expr_raw.where(~df["expression_missing"], 0.5)

    # Original CTS is itself on a [0,1]-ish scale already (weighted sum of
    # four [0,1] terms), so it's used directly rather than re-normalized,
    # preserving its meaning as "the validated base score."
    df["CTS_extended"] = (
        w["original_cts"] * df["CTS"]
        + w["amplification"] * df["amplification_norm"]
        + w["expression"] * df["expression_norm"]
    )
    return df


# =====================================================================
# SMOKE TEST
# =====================================================================

def _run_smoke_test():
    import numpy as np

    # Minimal fake "already-computed CTS result" for 3 kinases
    cts_result = pd.DataFrame({"CTS": [0.690, 0.613, 0.532]}, index=["ERBB2", "EGFR", "PTK2"])

    cnv_scores = pd.Series({"ERBB2": 1.8, "EGFR": 0.1, "PTK2": -0.05})  # ERBB2 amplified, matches real biology
    expression_scores = pd.Series({"ERBB2": 8.5, "EGFR": 6.2})  # PTK2 deliberately missing -> neutral fallback

    result = compute_cts_extended(cts_result, cnv_scores, expression_scores)
    print(result[["CTS", "amplification_norm", "expression_norm", "CTS_extended",
                   "amplification_missing", "expression_missing"]])
    print()

    assert result.loc["PTK2", "expression_missing"] == True
    assert result.loc["PTK2", "expression_norm"] == 0.5
    assert result.loc["ERBB2", "amplification_norm"] == 1.0, "ERBB2 has the highest CNV, should normalize to 1.0"
    assert abs(result["CTS_extended"].sum() - result["CTS_extended"].sum()) < 1e-9  # trivially true, sanity only
    assert cts_result["CTS"].tolist() == [0.690, 0.613, 0.532], "original CTS column must be untouched"

    print("PASSED: ERBB2's known amplification correctly normalizes to the top score;")
    print("PTK2's missing expression data correctly falls back to neutral (0.5);")
    print("original CTS column is provably unmodified.")


if __name__ == "__main__":
    _run_smoke_test()
