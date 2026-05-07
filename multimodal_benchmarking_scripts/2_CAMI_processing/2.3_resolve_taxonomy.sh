#!/bin/bash
#
# resolve_taxonomy.sh
#
# Resolves NCBI tax_ids in a CAMI gsa_mapping.tsv file to full taxonomic
# lineages using taxonkit, and produces a clean per-tax_id lineage table.
#
# Usage:
#   ./resolve_taxonomy.sh <gsa_mapping.tsv> [output_prefix]
#
# Example:
#   ./resolve_taxonomy.sh gsa_mapping.tsv my_dataset
#   # Produces:
#   #   my_dataset_unique_taxids.txt
#   #   my_dataset_taxid_lineages.tsv
#   #   my_dataset_rank_summary.txt
#
# Defaults output_prefix to the input filename without extension.
#
# Requires:
#   - taxonkit (with NCBI taxdump available at ~/.taxonkit or $TAXONKIT_DB)

set -euo pipefail

# ---- Argument parsing ----
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <gsa_mapping.tsv> [output_prefix]" >&2
    exit 1
fi

INPUT="$1"
PREFIX="${2:-$(basename "${INPUT%.*}")}"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: input file not found: $INPUT" >&2
    exit 1
fi

if ! command -v taxonkit &> /dev/null; then
    echo "Error: taxonkit not found in PATH" >&2
    exit 1
fi

# ---- Output filenames ----
TAXIDS_FILE="${PREFIX}_unique_taxids.txt"
LINEAGES_FILE="${PREFIX}_taxid_lineages.tsv"
RANK_SUMMARY="${PREFIX}_rank_summary.txt"

echo "Input:           $INPUT"
echo "Output prefix:   $PREFIX"
echo

# ---- Step 1: extract unique tax_ids ----
echo "[1/3] Extracting unique tax_ids from column 3..."
cut -f3 "$INPUT" | tail -n +2 | sort -u > "$TAXIDS_FILE"
N_TAXIDS=$(wc -l < "$TAXIDS_FILE")
echo "      Found $N_TAXIDS unique tax_ids -> $TAXIDS_FILE"
echo

# ---- Step 2: resolve lineages with rank info ----
echo "[2/3] Resolving lineages with taxonkit..."
{
    echo -e "tax_id\trank\tdomain\tphylum\tclass\torder\tfamily\tgenus\tspecies"
    taxonkit lineage -r "$TAXIDS_FILE" \
        | taxonkit reformat -f "{k}\t{p}\t{c}\t{o}\t{f}\t{g}\t{s}" \
        | awk -F'\t' 'BEGIN{OFS="\t"} {print $1,$3,$4,$5,$6,$7,$8,$9,$10}'
} > "$LINEAGES_FILE"
echo "      Wrote lineage table -> $LINEAGES_FILE"
echo

# ---- Step 3: rank summary ----
echo "[3/3] Summarizing rank distribution..."
{
    echo "Rank distribution of unique tax_ids:"
    echo "------------------------------------"
    tail -n +2 "$LINEAGES_FILE" | cut -f2 | sort | uniq -c | sort -rn
    echo
    echo "Tax_ids missing species-level assignment:"
    awk -F'\t' 'NR>1 && $9 == ""' "$LINEAGES_FILE" | wc -l
} | tee "$RANK_SUMMARY"
echo
echo "Done. Outputs:"
echo "  $TAXIDS_FILE"
echo "  $LINEAGES_FILE"
echo "  $RANK_SUMMARY"
