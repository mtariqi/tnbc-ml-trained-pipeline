# TNBC Multi-Omic Kinase Dependency ML Pipeline

Extends the [TNBC RTK/NRTK Kinase Scoring Pipeline](https://github.com/mtariqi/tnbc-kinase-scoring-pipeline) with real multi-omic data (expression, copy number, damaging mutations) from DepMap, adding:

1. **An extended CTS score** that layers copy-number and expression evidence on top of the original, already-validated CTS — without modifying it.
2. **A trained ML model** predicting CRISPR knockout dependency from multi-omic features — a genuine step toward the original grant proposal's Phase 2 roadmap item: *"replacing the current heuristic with a trained model."*

---

## Why This Extension Matters

The original CTS score combined network centrality, essentiality, survival, and druggability — but had **no copy-number signal at all**, despite several of the 90 kinases (ERBB2, FGFR1, KIT) being classic amplification-driven oncogenes. This extension closes that gap.

## Components

| File | Purpose |
|---|---|
| `src/depmap_multiomic_loader.py` | Parsers for DepMap's expression, CNV, dependency-probability, damaging-mutation, and subtype files |
| `src/cts_extended.py` | Adds CNV + expression as new terms in a **separate** `CTS_extended` column — the original `CTS` column and its already-reported result (ERBB2/EGFR/PTK2 top 3) remain untouched |
| `src/dependency_ml_model.py` | Trains a gradient-boosted regressor to predict dependency probability from expression/CNV/mutation features, using **grouped cross-validation by cell line** to get an honest generalization estimate |

## Why Grouped Cross-Validation

Each TNBC cell line contributes ~90 rows (one per kinase). A random train/test split could put some of a cell line's kinases in training and others in test — letting the model learn "this cell line tends to have high dependency" as a shortcut, inflating apparent accuracy without learning anything transferable. Grouping cross-validation by cell line (so an entire line is either fully train or fully test) tests what actually matters: generalization to a new, unseen cell line/patient.

## Verified, Not Just Executed

Before running on real data, every function here was tested against synthetic data with a **deliberately planted, known signal** — not just checked for "does it run." The model correctly recovered the planted expression/CNV → dependency relationship (R² = 0.83, Spearman ρ = 0.89 under proper grouped CV) and correctly ranked the two truly causal features above a weaker, irrelevant one — confirming the pipeline works correctly before being pointed at real biology.

**Important:** real data will not show R² = 0.83 — that number reflects a deliberately clean synthetic signal used to validate the code. Real multi-omic dependency prediction is a harder problem; report whatever the real data actually shows.

---

## Required DepMap Files

Download from [depmap.org/portal/download](https://depmap.org/portal/download):

- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
- `OmicsCNGeneWGS.csv`
- `CRISPRGeneDependency.csv`
- `OmicsSomaticMutationsMatrixDamaging.csv`
- `SubtypeMatrix.csv`

**Run `inspect_depmap_csv()` on each file first** — two of these (`Damaging`, `SubtypeMatrix`) have less-certain column-naming conventions that haven't been directly verified against the real files, unlike the others which follow DepMap's confirmed standard format.

## Usage

```python
from src.depmap_multiomic_loader import (
    parse_depmap_expression, parse_depmap_cnv, parse_depmap_dependency,
    parse_depmap_damaging_mutations,
)
from src.cts_extended import compute_cts_extended
from src.dependency_ml_model import build_feature_matrix, train_and_evaluate

kinase_list = [...]  # your 90 kinases
tnbc_ids = [...]     # your confirmed-TNBC ModelIDs

expr = parse_depmap_expression("OmicsExpressionTPM...csv", kinase_list, tnbc_ids)
cnv = parse_depmap_cnv("OmicsCNGeneWGS.csv", kinase_list, tnbc_ids)
dep = parse_depmap_dependency("CRISPRGeneDependency.csv", kinase_list, tnbc_ids)
mut = parse_depmap_damaging_mutations("OmicsSomaticMutationsMatrixDamaging.csv", kinase_list, tnbc_ids)

features = build_feature_matrix(expr, cnv, mut, dep)
result = train_and_evaluate(features)
print(f"R^2 = {result['r2']:.3f}, Spearman rho = {result['spearman_r']:.3f}")
print(result["feature_importances"])
```

## Related Repositories

- [tnbc-kinase-scoring-pipeline](https://github.com/mtariqi/tnbc-kinase-scoring-pipeline) — the base CTS/PairCTS/TripletCTS pipeline this extends
- [TNBC-drug-regimen-discovery](https://github.com/mtariqi/TNBC-drug-regimen-discovery) — the separate MDCOE/HCOS regimen-ranking system

## License

[MIT](LICENSE)
