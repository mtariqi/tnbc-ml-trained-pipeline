"""
DepMap Multi-Omic Loaders
==========================

Extends the original single-file DepMap loader (parse_depmap_gene_effect,
in kinase_data_fetchers.py) to the additional files available on DepMap's
portal: copy number, expression, dependency probability, and damaging
mutations.

IMPORTANT -- FORMAT ASSUMPTIONS THAT NEED VERIFYING ON THE REAL FILES:
    - OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv and
      OmicsCNGeneWGS.csv are assumed to follow the SAME convention already
      confirmed for CRISPRGeneEffect.csv: rows = ModelID, columns =
      "SYMBOL (EntrezID)". This is DepMap's standard convention across
      their CRISPR/Omics files, but has NOT been directly verified against
      these two specific files.
    - CRISPRGeneDependency.csv is assumed to follow the identical
      ModelID x gene-column layout as CRISPRGeneEffect.csv, with values
      being a dependency PROBABILITY (0-1) rather than a Chronos score.
    - OmicsSomaticMutationsMatrixDamaging.csv and SubtypeMatrix.csv have
      LESS CERTAIN column-naming conventions (may or may not use the
      "(EntrezID)" suffix). The parser strips it if present and falls
      back to the bare column name if not, but RUN inspect_depmap_csv()
      on each real file first and compare against what you see before
      trusting the parsed output.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# =====================================================================
# 0. GENERIC INSPECTOR -- RUN THIS FIRST ON EVERY REAL FILE
# =====================================================================

def inspect_depmap_csv(path: str, n_cols: int = 8, n_rows: int = 5) -> None:
    """Print shape, index name, and a sample of column names/values so you
    can confirm this parser's format assumptions before trusting it."""
    df = pd.read_csv(path, nrows=n_rows)
    print(f"=== {path} ===")
    print(f"First column (likely the index/ModelID column): '{df.columns[0]}'")
    print(f"Sample values in first column: {df.iloc[:, 0].tolist()}")
    print(f"Total columns: {len(pd.read_csv(path, nrows=0).columns)}")
    print(f"First {n_cols} column names: {df.columns[1:n_cols+1].tolist()}")
    print(f"Sample row of values: {df.iloc[0, 1:n_cols+1].tolist()}")


# =====================================================================
# 1. GENERIC WIDE-MATRIX PARSER (ModelID rows x 'SYMBOL (ID)' columns)
# =====================================================================

def parse_depmap_wide_matrix(
    csv_path: str,
    genes: List[str],
    cell_line_ids: Optional[List[str]] = None,
    value_name: str = "value",
    strip_id_suffix: bool = True,
) -> pd.DataFrame:
    """
    Generic parser for any DepMap file shaped like CRISPRGeneEffect.csv:
    rows = ModelID, columns = gene symbols (optionally with a '(EntrezID)'
    suffix). Used as the shared implementation behind the specific
    parse_depmap_expression() / parse_depmap_cnv() / parse_depmap_dependency()
    wrappers below.

    Returns a tidy DataFrame: kinase_id, cell_line_id, <value_name>,
    plus a 'mean_<value_name>' column (averaged across cell_line_ids).
    """
    df = pd.read_csv(csv_path, index_col=0)

    if cell_line_ids is not None:
        missing = set(cell_line_ids) - set(df.index)
        if missing:
            print(f"Warning: {len(missing)} requested cell lines not found in file, skipping them.")
        df = df.loc[df.index.intersection(cell_line_ids)]

    if strip_id_suffix:
        col_lookup = {col.split(" (")[0]: col for col in df.columns}
    else:
        col_lookup = {col: col for col in df.columns}

    found_genes = [g for g in genes if g in col_lookup]
    missing_genes = [g for g in genes if g not in col_lookup]
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes not found in {csv_path}: {missing_genes}")

    sub = df[[col_lookup[g] for g in found_genes]].copy()
    sub.columns = found_genes

    tidy = sub.reset_index().melt(
        id_vars=sub.index.name or "index", var_name="kinase_id", value_name=value_name
    )
    tidy = tidy.rename(columns={sub.index.name or "index": "cell_line_id"})

    mean_col = f"mean_{value_name}"
    means = tidy.groupby("kinase_id")[value_name].mean().rename(mean_col)
    tidy = tidy.merge(means, on="kinase_id", how="left")
    return tidy


# =====================================================================
# 2. SPECIFIC WRAPPERS
# =====================================================================

def parse_depmap_expression(csv_path: str, genes: List[str], cell_line_ids=None) -> pd.DataFrame:
    """OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv -- log2(TPM+1) expression."""
    return parse_depmap_wide_matrix(csv_path, genes, cell_line_ids, value_name="log_tpm")


def parse_depmap_cnv(csv_path: str, genes: List[str], cell_line_ids=None) -> pd.DataFrame:
    """OmicsCNGeneWGS.csv -- gene-level copy number (log2 relative copy ratio)."""
    return parse_depmap_wide_matrix(csv_path, genes, cell_line_ids, value_name="log2_cn")


def parse_depmap_dependency(csv_path: str, genes: List[str], cell_line_ids=None) -> pd.DataFrame:
    """CRISPRGeneDependency.csv -- probability (0-1) that knocking out this
    gene reduces viability. A complementary metric to the Chronos essentiality
    score (parse_depmap_gene_effect in kinase_data_fetchers.py) -- more
    directly interpretable, useful as a cross-check or an ML training label."""
    return parse_depmap_wide_matrix(csv_path, genes, cell_line_ids, value_name="dependency_prob")


def parse_depmap_damaging_mutations(csv_path: str, genes: List[str], cell_line_ids=None) -> pd.DataFrame:
    """OmicsSomaticMutationsMatrixDamaging.csv -- binary (0/1) flag for
    whether each cell line carries a damaging mutation in each gene.
    NOTE: column-naming convention for this file is less certain than the
    others -- run inspect_depmap_csv() on it first."""
    return parse_depmap_wide_matrix(csv_path, genes, cell_line_ids, value_name="damaging_mutation")


def parse_depmap_subtype_matrix(csv_path: str, cell_line_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    SubtypeMatrix.csv -- one-hot encoded subtype labels per ModelID.
    Returns the raw (not melted) DataFrame, indexed by ModelID, since
    subtype columns are categories, not genes -- use directly as ML features
    rather than reshaping through parse_depmap_wide_matrix().
    """
    df = pd.read_csv(csv_path, index_col=0)
    if cell_line_ids is not None:
        missing = set(cell_line_ids) - set(df.index)
        if missing:
            print(f"Warning: {len(missing)} requested cell lines not found in file, skipping them.")
        df = df.loc[df.index.intersection(cell_line_ids)]
    return df


# =====================================================================
# 3. SMOKE TESTS -- SYNTHETIC FILES MATCHING THE DOCUMENTED FORMAT
# =====================================================================

def _make_synthetic_wide_file(path: str, cell_lines: List[str], genes: List[str], value_range, seed: int):
    rng = np.random.default_rng(seed)
    cols = [f"{g} ({1000+i})" for i, g in enumerate(genes)]
    df = pd.DataFrame(rng.uniform(*value_range, size=(len(cell_lines), len(genes))),
                       index=cell_lines, columns=cols)
    df.index.name = "ModelID"
    df.to_csv(path)


def _run_smoke_tests():
    cell_lines = ["ACH-000768", "ACH-000223", "ACH-000288"]  # MDAMB231, HCC1937, BT549
    genes = ["EGFR", "ERBB2", "PTK2"]

    _make_synthetic_wide_file("/tmp/test_expr.csv", cell_lines, genes, (0, 10), seed=1)
    _make_synthetic_wide_file("/tmp/test_cnv.csv", cell_lines, genes, (-1, 2), seed=2)
    _make_synthetic_wide_file("/tmp/test_dep.csv", cell_lines, genes, (0, 1), seed=3)
    _make_synthetic_wide_file("/tmp/test_mut.csv", cell_lines, genes, (0, 1), seed=4)

    print("=== inspect_depmap_csv() ===")
    inspect_depmap_csv("/tmp/test_expr.csv")
    print()

    print("=== parse_depmap_expression() ===")
    expr = parse_depmap_expression("/tmp/test_expr.csv", genes, cell_line_ids=cell_lines)
    print(expr.head(), "\n")
    assert set(expr["kinase_id"].unique()) == set(genes)
    assert expr["cell_line_id"].nunique() == len(cell_lines)

    print("=== parse_depmap_cnv() ===")
    cnv = parse_depmap_cnv("/tmp/test_cnv.csv", genes, cell_line_ids=cell_lines)
    print(cnv.head(), "\n")

    print("=== parse_depmap_dependency() ===")
    dep = parse_depmap_dependency("/tmp/test_dep.csv", genes, cell_line_ids=cell_lines)
    print(dep.head(), "\n")
    assert dep["dependency_prob"].between(0, 1).all()

    print("=== parse_depmap_damaging_mutations() ===")
    mut = parse_depmap_damaging_mutations("/tmp/test_mut.csv", genes, cell_line_ids=cell_lines)
    print(mut.head(), "\n")

    # missing-gene handling
    print("=== Missing-gene handling (asks for a gene not in the file) ===")
    expr2 = parse_depmap_expression("/tmp/test_expr.csv", genes + ["NOT_A_REAL_GENE"], cell_line_ids=cell_lines)
    assert "NOT_A_REAL_GENE" not in expr2["kinase_id"].unique()
    print("Correctly excluded, with a warning, rather than silently producing NaN rows.\n")

    print("All multi-omic loader smoke tests passed against synthetic, format-matched files.")
    print("Run inspect_depmap_csv() on each REAL file before trusting parsed output --")
    print("column-naming conventions for the mutation and subtype files specifically")
    print("have not been directly verified.")


if __name__ == "__main__":
    _run_smoke_tests()
