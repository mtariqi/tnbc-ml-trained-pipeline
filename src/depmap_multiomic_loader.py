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
    Generic parser for DepMap wide-format files. Handles TWO layouts,
    auto-detected:

    Layout A (CRISPRGeneEffect.csv, CRISPRGeneDependency.csv, expression,
    CNV): rows = ModelID directly as the index, columns = gene symbols
    (optionally '(EntrezID)' suffix).

    Layout B (OmicsSomaticMutationsMatrixDamaging.csv/Hotspot.csv):
    ModelID is a NAMED COLUMN, not the index -- the actual index is a
    meaningless row number, and there are leading metadata columns
    (SequencingID, ModelConditionID, IsDefaultEntryForModel,
    IsDefaultEntryForMC) before the gene columns start. This was
    confirmed against the real file (previously assumed to match Layout
    A, which was wrong and would have silently indexed by row number).
    When this layout is detected, rows are also filtered to
    IsDefaultEntryForModel == 'Yes' if that column is present, to avoid
    double-counting a model with multiple sequencing entries.

    Returns a tidy DataFrame: kinase_id, cell_line_id, <value_name>,
    plus a 'mean_<value_name>' column (averaged across cell_line_ids).
    """
    raw = pd.read_csv(csv_path)

    if "ModelID" in raw.columns:
        # Layout B
        if "IsDefaultEntryForModel" in raw.columns:
            before = len(raw)
            raw = raw[raw["IsDefaultEntryForModel"] == "Yes"]
            if len(raw) < before:
                print(f"Filtered to IsDefaultEntryForModel=='Yes': {before} -> {len(raw)} rows.")
        df = raw.set_index("ModelID")
        metadata_cols = [c for c in ("SequencingID", "ModelConditionID", "IsDefaultEntryForModel", "IsDefaultEntryForMC")
                         if c in df.columns]
        df = df.drop(columns=metadata_cols)
    else:
        # Layout A
        df = raw.set_index(raw.columns[0])

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
# 1b. MUTATION MATRIX PARSER -- DIFFERENT STRUCTURE, CONFIRMED AGAINST REAL FILE
# =====================================================================

def parse_depmap_mutation_matrix(
    csv_path: str,
    genes: List[str],
    cell_line_ids: Optional[List[str]] = None,
    value_name: str = "mutation_flag",
) -> pd.DataFrame:
    """
    Parser for OmicsSomaticMutationsMatrixDamaging.csv and
    OmicsSomaticMutationsMatrixHotspot.csv specifically -- CONFIRMED to
    have a DIFFERENT structure than CRISPRGeneEffect.csv/CRISPRGeneDependency.csv:

        - 'ModelID' is a NAMED COLUMN, not the row index (the actual row
          index is just a meaningless sequential integer).
        - There are 4 other metadata columns before the gene columns start:
          SequencingID, ModelConditionID, IsDefaultEntryForModel,
          IsDefaultEntryForMC.
        - A single ModelID can have MULTIPLE rows (repeat sequencing runs /
          model conditions) -- IsDefaultEntryForModel marks which single
          row is the canonical one to use ('Yes'/'No').

    Using parse_depmap_wide_matrix() (built for the OTHER convention) on
    this file would silently treat the metadata columns as genes and use
    a meaningless integer as the cell-line identifier -- this function
    exists specifically to avoid that.
    """
    df = pd.read_csv(csv_path)

    if "ModelID" not in df.columns or "IsDefaultEntryForModel" not in df.columns:
        raise ValueError(
            "Expected 'ModelID' and 'IsDefaultEntryForModel' columns -- this file's format "
            "may differ from what was confirmed. Run inspect_depmap_csv() and compare."
        )

    before = len(df)
    df = df[df["IsDefaultEntryForModel"] == "Yes"]
    print(f"Filtered to {len(df)}/{before} rows marked as the default entry per model "
          f"(dropping duplicate/non-canonical sequencing runs).")

    metadata_cols = {"SequencingID", "ModelID", "ModelConditionID", "IsDefaultEntryForModel", "IsDefaultEntryForMC"}
    gene_col_lookup = {c.split(" (")[0]: c for c in df.columns if c not in metadata_cols}

    if cell_line_ids is not None:
        missing = set(cell_line_ids) - set(df["ModelID"])
        if missing:
            print(f"Warning: {len(missing)} requested cell lines not found in file, skipping them.")
        df = df[df["ModelID"].isin(cell_line_ids)]

    found_genes = [g for g in genes if g in gene_col_lookup]
    missing_genes = [g for g in genes if g not in gene_col_lookup]
    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes not found in {csv_path}: {missing_genes}")

    sub = df[["ModelID"] + [gene_col_lookup[g] for g in found_genes]].copy()
    sub.columns = ["cell_line_id"] + found_genes

    tidy = sub.melt(id_vars="cell_line_id", var_name="kinase_id", value_name=value_name)
    mean_col = f"mean_{value_name}"
    means = tidy.groupby("kinase_id")[value_name].mean().rename(mean_col)
    tidy = tidy.merge(means, on="kinase_id", how="left")
    return tidy




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
    CONFIRMED structure: ModelID is a named column (not the row index),
    with 4 metadata columns before the gene columns start, and possible
    duplicate rows per model -- see parse_depmap_mutation_matrix()."""
    return parse_depmap_mutation_matrix(csv_path, genes, cell_line_ids, value_name="damaging_mutation")


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
