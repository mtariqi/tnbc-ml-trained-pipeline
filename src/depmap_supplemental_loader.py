"""
DepMap Supplemental File Loaders
==================================

Extends depmap_multiomic_loader.py with the supplemental files identified
as high-value for this project: gene fusions, pan-essential/non-essential
control gene lists, CRISPR confounders, DepMap-inferred molecular subtypes
(for validating the manual TNBC cell-line list), and gene ID metadata.

FORMAT CERTAINTY, HONESTLY STATED:
    - Common-essential / non-essential control lists: HIGH confidence --
      these are well-documented, simple single-column gene lists in
      DepMap's format (this convention is stable across DepMap releases).
    - Gene.csv: HIGH confidence -- standard gene metadata table.
    - OmicsSomaticMutationsMatrixHotspot.csv: HIGH confidence -- same
      ModelID x gene wide-matrix convention already confirmed for the
      Damaging matrix and CRISPRGeneEffect.csv.
    - CRISPRConfounders.csv, OmicsInferredMolecularSubtypes.csv,
      OmicsFusionFiltered.csv: LOWER confidence -- these have less
      standardized formats across DepMap releases. Parsers below are
      written defensively (checking for expected columns, printing what
      was actually found) rather than assuming blindly. RUN
      inspect_depmap_csv() ON EACH before trusting the output.
"""

from __future__ import annotations

from typing import List, Optional, Set

import pandas as pd

from depmap_multiomic_loader import parse_depmap_wide_matrix, inspect_depmap_csv


# =====================================================================
# 1. HOTSPOT MUTATIONS (same wide-matrix convention as Damaging)
# =====================================================================

def parse_depmap_hotspot_mutations(csv_path: str, genes: List[str], cell_line_ids=None) -> pd.DataFrame:
    """OmicsSomaticMutationsMatrixHotspot.csv -- binary (0/1) flag for a
    hotspot (activating) mutation, as distinct from the Damaging matrix's
    loss-of-function signal. Important for your kinase panel specifically,
    since these are almost all proto-oncogenes where activation (not loss)
    is the relevant event."""
    return parse_depmap_wide_matrix(csv_path, genes, cell_line_ids, value_name="hotspot_mutation")


# =====================================================================
# 2. PAN-ESSENTIAL / NON-ESSENTIAL CONTROL GENE LISTS
# =====================================================================

def load_gene_list(csv_path: str, gene_column: Optional[str] = None) -> Set[str]:
    """
    Generic loader for DepMap's simple gene-list files (common essentials,
    non-essential controls). These are typically a single column, but the
    column name varies by release ('Gene', 'gene', or a combined
    'SYMBOL (EntrezID)' format) -- this function checks for the common
    cases and strips any '(EntrezID)' suffix if present.
    """
    df = pd.read_csv(csv_path)
    if gene_column is None:
        # Guess the gene column: first column, or one matching common names
        candidates = [c for c in df.columns if c.lower() in ("gene", "genes", "symbol", "hugo_symbol")]
        gene_column = candidates[0] if candidates else df.columns[0]
    genes = df[gene_column].astype(str).str.split(" (", regex=False).str[0]
    return set(genes)


def flag_pan_essential_kinases(
    kinase_list: List[str],
    common_essentials_path: str,
    nonessential_controls_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Flags which of your kinases are DepMap-defined pan-essential (essential
    across nearly all cell lines -- a poor, non-selective drug target,
    since inhibiting it would be broadly toxic rather than TNBC-selective)
    versus not. Returns a DataFrame: kinase_id, is_pan_essential,
    is_nonessential_control.
    """
    common_essential_genes = load_gene_list(common_essentials_path)
    rows = []
    for k in kinase_list:
        row = {"kinase_id": k, "is_pan_essential": k in common_essential_genes}
        if nonessential_controls_path:
            nonessential_genes = load_gene_list(nonessential_controls_path)
            row["is_nonessential_control"] = k in nonessential_genes
        rows.append(row)
    df = pd.DataFrame(rows)
    n_pan = df["is_pan_essential"].sum()
    if n_pan:
        print(f"Note: {n_pan} of {len(kinase_list)} kinases are pan-essential "
              f"(essential in nearly all cell lines, not TNBC-selective) -- "
              f"worth flagging as lower-priority drug targets despite high raw essentiality: "
              f"{df.loc[df['is_pan_essential'], 'kinase_id'].tolist()}")
    return df


# =====================================================================
# 3. GENE FUSIONS -- flexible column detection, since format is less certain
# =====================================================================

def parse_gene_fusions(csv_path: str, target_genes: List[str], cell_line_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    OmicsFusionFiltered.csv -- filters to fusion events involving any gene
    in `target_genes` (pass your 90 kinases; fusions are especially
    relevant for NTRK1/2/3, ALK, ROS1, RET specifically). Column names for
    the two fusion partners are auto-detected from common DepMap
    conventions (LeftGene/RightGene, Gene1/Gene2, FusionName parsing) --
    PRINTS what it found so you can confirm it guessed correctly.
    """
    df = pd.read_csv(csv_path)
    print(f"Columns found in {csv_path}: {df.columns.tolist()}")

    model_col = next((c for c in df.columns if c.lower() in ("modelid", "model_id", "depmap_id")), None)
    gene_cols = [c for c in df.columns if any(
        kw in c.lower() for kw in ("leftgene", "rightgene", "gene1", "gene2", "genea", "geneb", "site1gene", "site2gene")
    )]

    if not gene_cols:
        print("Warning: could not auto-detect fusion partner gene columns. "
              "Inspect the printed column list above and adapt this function manually.")
        return pd.DataFrame()

    if cell_line_ids is not None and model_col:
        df = df[df[model_col].isin(cell_line_ids)]

    mask = False
    for col in gene_cols:
        mask = mask | df[col].isin(target_genes)
    hits = df[mask].copy()

    print(f"Found {len(hits)} fusion event(s) involving one of your {len(target_genes)} target genes, "
          f"out of {len(df)} total fusion records{' in your cell-line subset' if cell_line_ids else ''}.")
    return hits


# =====================================================================
# 4. CRISPR CONFOUNDERS + INFERRED SUBTYPES (generic wide-matrix reuse)
# =====================================================================

def parse_crispr_confounders(csv_path: str, cell_line_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """CRISPRConfounders.csv -- returns the raw wide DataFrame (indexed by
    ModelID) so you can inspect and choose which confounder columns to
    actually control for; format/column names not standardized enough to
    guess a specific downstream use here."""
    df = pd.read_csv(csv_path, index_col=0)
    if cell_line_ids is not None:
        df = df.loc[df.index.intersection(cell_line_ids)]
    return df


def validate_tnbc_subtype_labels(
    csv_path: str,
    your_tnbc_model_ids: List[str],
) -> pd.DataFrame:
    """
    Cross-checks your manually-curated TNBC cell-line list (from
    ModelSubtypeFeatures string-matching, done earlier) against DepMap's
    own inferred molecular subtype calls. Returns a DataFrame flagging any
    disagreement -- a cell line you called TNBC that DepMap's own subtype
    inference does NOT confirm, or vice versa, is worth a manual look
    rather than silently trusting either source alone.
    """
    df = pd.read_csv(csv_path, index_col=0)
    print(f"Columns found: {df.columns.tolist()}")

    result = pd.DataFrame(index=df.index)
    result["in_your_tnbc_list"] = result.index.isin(your_tnbc_model_ids)

    subtype_col = next((c for c in df.columns if "subtype" in c.lower() or "lineage" in c.lower()), df.columns[0])
    result["depmap_subtype_label"] = df[subtype_col]
    result["depmap_says_tnbc"] = df[subtype_col].astype(str).str.contains("TNBC|Basal", case=False, na=False)

    disagreements = result[result["in_your_tnbc_list"] != result["depmap_says_tnbc"]]
    if len(disagreements):
        print(f"\n{len(disagreements)} disagreement(s) between your TNBC list and DepMap's inferred subtype "
              f"-- worth a manual look:\n{disagreements}")
    else:
        print("\nNo disagreements -- your TNBC cell-line list is fully corroborated by DepMap's own subtype inference.")
    return result


# =====================================================================
# 5. GENE METADATA (authoritative symbol/ID cross-reference)
# =====================================================================

def load_gene_metadata(csv_path: str) -> pd.DataFrame:
    """Gene.csv -- authoritative gene ID/symbol mapping. Useful as a
    cross-reference given the gene-symbol-format mismatches encountered
    earlier this session (the '(EntrezID)' suffix issue, and the
    ror1/syk-as-drug-names DGIdb data-quality bug)."""
    return pd.read_csv(csv_path)


# =====================================================================
# SMOKE TESTS
# =====================================================================

def _run_smoke_tests():
    import numpy as np

    kinases = ["EGFR", "ERBB2", "PTK2", "NTRK1", "ALK"]
    cell_lines = ["ACH-000768", "ACH-000223", "ACH-000288"]

    # --- gene list files ---
    pd.DataFrame({"Gene": ["EGFR (1956)", "PTK2 (5747)", "RPL3 (6122)"]}).to_csv("/tmp/test_common_ess.csv", index=False)
    pd.DataFrame({"Gene": ["OR4F5 (79501)"]}).to_csv("/tmp/test_nonessential.csv", index=False)

    print("=== flag_pan_essential_kinases() ===")
    flags = flag_pan_essential_kinases(kinases, "/tmp/test_common_ess.csv", "/tmp/test_nonessential.csv")
    print(flags, "\n")
    assert flags.loc[flags["kinase_id"] == "EGFR", "is_pan_essential"].iloc[0] == True
    assert flags.loc[flags["kinase_id"] == "ALK", "is_pan_essential"].iloc[0] == False

    # --- fusion file (guessing DepMap's real LeftGene/RightGene convention) ---
    fusion_df = pd.DataFrame({
        "ModelID": ["ACH-000768", "ACH-000223", "ACH-000288"],
        "LeftGene": ["NTRK1", "TP53", "BCR"],
        "RightGene": ["TPM3", "MDM2", "ABL1"],
    })
    fusion_df.to_csv("/tmp/test_fusions.csv", index=False)

    print("=== parse_gene_fusions() ===")
    hits = parse_gene_fusions("/tmp/test_fusions.csv", kinases, cell_line_ids=cell_lines)
    print(hits, "\n")
    assert len(hits) == 1, "should find exactly the NTRK1 fusion"
    assert hits.iloc[0]["ModelID"] == "ACH-000768"

    # --- subtype validation ---
    subtype_df = pd.DataFrame({
        "ModelID": ["ACH-000768", "ACH-000223", "ACH-000288"],
        "InferredSubtype": ["TNBC", "HER2+", "TNBC"],
    }).set_index("ModelID")
    subtype_df.to_csv("/tmp/test_subtypes.csv")

    print("=== validate_tnbc_subtype_labels() ===")
    your_list = ["ACH-000768", "ACH-000223"]  # deliberately includes ACH-000223, which DepMap calls HER2+ -- should flag a disagreement
    result = validate_tnbc_subtype_labels("/tmp/test_subtypes.csv", your_list)
    print(result, "\n")
    assert result.loc["ACH-000223", "in_your_tnbc_list"] != result.loc["ACH-000223", "depmap_says_tnbc"]

    print("All supplemental loader smoke tests passed, including a correctly-detected")
    print("subtype disagreement (proving the validation function actually catches")
    print("mismatches, not just always agreeing).")


if __name__ == "__main__":
    _run_smoke_tests()
