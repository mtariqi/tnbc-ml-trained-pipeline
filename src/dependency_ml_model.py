"""
Multi-Omic Kinase Dependency Prediction (the Phase 2 "learned model" idea)
============================================================================

Trains a supervised model to predict CRISPR dependency probability for a
(kinase, cell-line) pair from multi-omic features -- expression, copy
number, damaging-mutation status, and subtype. This is a genuine step
toward the original grant proposal's Phase 2 roadmap item: "replacing the
current heuristic with a trained model."

WHY GROUPED CROSS-VALIDATION, NOT RANDOM K-FOLD:
    Each cell line contributes ~90 rows (one per kinase) to the feature
    matrix. A random train/test split would put some of a cell line's
    kinases in the training set and others in the test set for the SAME
    cell line -- the model could then trivially learn "this cell line
    tends to have high dependency values" as a shortcut, inflating
    apparent accuracy without learning anything transferable. Grouping
    the cross-validation by cell_line_id (so an entire cell line is
    either fully train or fully test) gives an honest estimate of how
    well this generalizes to a NEW, unseen cell line/patient -- the
    actually useful question.

WHY GRADIENT BOOSTING, NOT A DEEP NET:
    With ~25 TNBC cell lines x 90 kinases (~2000-2250 rows) and a handful
    of features, this is small tabular data -- squarely gradient-boosted
    trees' strong suit, and far less prone to overfitting here than a
    neural network would be at this sample size.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


# =====================================================================
# 1. BUILD THE FEATURE MATRIX
# =====================================================================

def build_feature_matrix(
    expression_df: pd.DataFrame,
    cnv_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
    dependency_df: pd.DataFrame,
    subtype_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Each *_df input is the tidy long-format output of the corresponding
    depmap_multiomic_loader.py parser (columns: cell_line_id, kinase_id,
    <value>), except subtype_df, which is the raw wide DataFrame from
    parse_depmap_subtype_matrix() (indexed by cell_line_id / ModelID).

    Returns one row per (cell_line_id, kinase_id) with columns:
    log_tpm, log2_cn, damaging_mutation, dependency_prob (the prediction
    target), plus one column per subtype category if subtype_df is given.
    Rows with a missing target (dependency_prob) are dropped -- you can't
    train or evaluate against a label that doesn't exist, unlike CTS's
    missing-value policy which is about INPUT features, not the label.
    """
    merged = dependency_df[["cell_line_id", "kinase_id", "dependency_prob"]].merge(
        expression_df[["cell_line_id", "kinase_id", "log_tpm"]],
        on=["cell_line_id", "kinase_id"], how="left",
    ).merge(
        cnv_df[["cell_line_id", "kinase_id", "log2_cn"]],
        on=["cell_line_id", "kinase_id"], how="left",
    ).merge(
        mutation_df[["cell_line_id", "kinase_id", "damaging_mutation"]],
        on=["cell_line_id", "kinase_id"], how="left",
    )

    if subtype_df is not None:
        merged = merged.merge(
            subtype_df, left_on="cell_line_id", right_index=True, how="left",
        )

    before = len(merged)
    merged = merged.dropna(subset=["dependency_prob"])
    dropped = before - len(merged)
    if dropped:
        print(f"Dropped {dropped}/{before} rows with no dependency label (can't train/evaluate without one).")

    # Missing FEATURE values (not the label) get median-imputed per column --
    # simple, defensible default for a first model; revisit if a specific
    # feature has heavy missingness, in which case investigate why rather
    # than just impute through it.
    feature_cols = [c for c in merged.columns if c not in ("cell_line_id", "kinase_id", "dependency_prob")]
    for col in feature_cols:
        if merged[col].isna().any():
            n_missing = merged[col].isna().sum()
            merged[col] = merged[col].fillna(merged[col].median())
            print(f"Imputed {n_missing} missing '{col}' values with the column median.")

    return merged


# =====================================================================
# 2. TRAIN + GROUPED CROSS-VALIDATE
# =====================================================================

def train_and_evaluate(
    feature_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Trains a GradientBoostingRegressor to predict dependency_prob from all
    other columns (except cell_line_id/kinase_id), using GroupKFold
    cross-validation grouped by cell_line_id (see module docstring for why).

    Returns a dict: {model (fit on ALL data), cv_predictions, r2, spearman_r,
    feature_importances (DataFrame sorted descending)}.
    """
    feature_cols = [c for c in feature_df.columns if c not in ("cell_line_id", "kinase_id", "dependency_prob")]
    X = feature_df[feature_cols].values
    y = feature_df["dependency_prob"].values
    groups = feature_df["cell_line_id"].values

    n_groups = feature_df["cell_line_id"].nunique()
    actual_splits = min(n_splits, n_groups)
    if actual_splits < n_splits:
        print(f"Warning: only {n_groups} unique cell lines available, reducing n_splits from {n_splits} to {actual_splits}")

    gkf = GroupKFold(n_splits=actual_splits)
    model = GradientBoostingRegressor(random_state=random_state, n_estimators=150, max_depth=3, learning_rate=0.05)

    cv_preds = cross_val_predict(model, X, y, cv=gkf, groups=groups)

    r2 = r2_score(y, cv_preds)
    rho, pval = spearmanr(y, cv_preds)

    # Fit on ALL data for the final feature-importance ranking / deployed model
    model.fit(X, y)
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "cv_predictions": cv_preds,
        "r2": r2,
        "spearman_r": rho,
        "spearman_p": pval,
        "feature_importances": importances,
        "n_cell_lines": n_groups,
        "n_rows": len(feature_df),
    }


# =====================================================================
# 3. SMOKE TEST -- SYNTHETIC DATA WITH A REAL, PLANTED SIGNAL
# =====================================================================

def _run_smoke_test():
    rng = np.random.default_rng(0)
    cell_lines = [f"ACH-{i:06d}" for i in range(15)]
    kinases = ["EGFR", "ERBB2", "PTK2", "FGFR1", "SRC"]

    rows = []
    for cl in cell_lines:
        for k in kinases:
            expr = rng.uniform(2, 10)
            cnv = rng.uniform(-1, 2)
            mut = rng.integers(0, 2)
            # PLANT A REAL, DETECTABLE SIGNAL: dependency rises with expression
            # and CNV, drops slightly if there's a damaging mutation (arbitrary
            # but consistent relationship) -- so we can verify the model
            # actually learns something, not just that the code runs.
            dep = np.clip(0.1 + 0.06*expr + 0.15*cnv - 0.1*mut + rng.normal(0, 0.05), 0, 1)
            rows.append({"cell_line_id": cl, "kinase_id": k, "log_tpm": expr,
                         "log2_cn": cnv, "damaging_mutation": mut, "dependency_prob": dep})

    dependency_df = pd.DataFrame(rows)
    expression_df = dependency_df[["cell_line_id", "kinase_id", "log_tpm"]].copy()
    cnv_df = dependency_df[["cell_line_id", "kinase_id", "log2_cn"]].copy()
    mutation_df = dependency_df[["cell_line_id", "kinase_id", "damaging_mutation"]].copy()

    print("=== build_feature_matrix() ===")
    features = build_feature_matrix(expression_df, cnv_df, mutation_df, dependency_df)
    print(features.head(), "\n")
    assert len(features) == len(cell_lines) * len(kinases)

    print("=== train_and_evaluate() ===")
    result = train_and_evaluate(features, n_splits=5)
    print(f"R^2 (grouped CV): {result['r2']:.3f}")
    print(f"Spearman rho (grouped CV): {result['spearman_r']:.3f} (p={result['spearman_p']:.2e})")
    print(f"\nFeature importances:\n{result['feature_importances']}")
    print()

    # Correctness check: with a strong planted signal, the model should
    # recover real predictive power (not just "runs without crashing").
    assert result["r2"] > 0.3, f"R^2 too low ({result['r2']:.3f}) -- model isn't learning the planted signal"
    assert result["spearman_r"] > 0.5, "Spearman correlation too low -- planted signal not recovered"
    # log_tpm and log2_cn were given the largest coefficients in the synthetic
    # generator (0.06 and 0.15 respectively) -- confirm they rank as the top
    # two most important features, not damaging_mutation.
    top_two = set(result["feature_importances"]["feature"].iloc[:2])
    assert top_two == {"log_tpm", "log2_cn"}, f"expected log_tpm/log2_cn as top features, got {top_two}"

    print("PASSED: model recovers the planted expression/CNV -> dependency relationship")
    print("(R^2 and Spearman both indicate real, non-trivial predictive power), and")
    print("correctly identifies log_tpm/log2_cn as the most important features --")
    print("exactly matching how the synthetic data was generated.")


if __name__ == "__main__":
    _run_smoke_test()
