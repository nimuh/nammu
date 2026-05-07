library(data.table)

setwd("~/anonymous/simulation_short_read/2018.08.15_09.49.32_sample_0/contigs/")


genes_proteins <- fread("prodigal_output.csv")




mapping  <- fread("gsa_mapping.tsv")
lineages <- fread("cami_marine_taxid_lineages.tsv")  # adjust filename
result   <- merge(mapping, lineages, by = "tax_id", all.x = TRUE)

# Tier 1: species-level
tier1 <- result[rank == "species"]

# Tier 2: genus-level (novel species)
tier2 <- result[rank == "genus"]

# Tier 3: coarser ranks
tier3 <- result[rank %in% c("order", "family", "class")]

cat("Tier 1 (species):", nrow(tier1), "contigs\n")
cat("Tier 2 (genus):  ", nrow(tier2), "contigs\n")
cat("Tier 3 (coarse): ", nrow(tier3), "contigs\n")

# Distribution of contigs per species in tier 1
tier1[, .N, by = species][order(-N)][1:20]  # top 20 most abundant


cat("Tier 1 (species-resolved tax_ids) breakdown:\n")
cat("  Unique tax_ids: ", tier1[, uniqueN(tax_id)],  "\n")
cat("  Unique species: ", tier1[, uniqueN(species)], "\n")
cat("  Unique genera:  ", tier1[, uniqueN(genus)],   "\n")
cat("  Unique families:", tier1[, uniqueN(family)],  "\n")
cat("  Unique orders:  ", tier1[, uniqueN(order)],   "\n")
cat("  Unique classes: ", tier1[, uniqueN(class)],   "\n")
cat("  Unique phyla:   ", tier1[, uniqueN(phylum)],  "\n")
cat("  Unique domains: ", tier1[, uniqueN(domain)],  "\n")
cat("  Total contigs:  ", nrow(tier1),                "\n")



genes_with_tax <- merge(
  genes_proteins,
  result,
  by.x = "contig_ID",                  # column in genes_proteins
  by.y = "#anonymous_contig_id",        # column in result
  all.x = TRUE                         # keep all genes, even if contig has no tax info
)

# Sanity check
nrow(genes_proteins)        # original gene count
nrow(genes_with_tax)        # should match (assuming no duplicate contigs in result)
genes_with_tax[is.na(tax_id), .N]  # genes whose contig wasn't in mapping


genes_filtered <- genes_with_tax[
  !is.na(species) &
    species != "" &
    !grepl("unidentified|uncultured|unclassified|environmental sample",
           species, ignore.case = TRUE)
]



# Summary
cat("Final dataset:\n")
cat("  Genes:           ", nrow(genes_filtered), "\n")
cat("  Unique contigs:  ", genes_filtered[, uniqueN(contig_ID)], "\n")
cat("  Unique species:  ", genes_filtered[, uniqueN(species)], "\n")
cat("  Unique genera:   ", genes_filtered[, uniqueN(genus)], "\n")
cat("  Unique phyla:    ", genes_filtered[, uniqueN(phylum)], "\n")




fwrite(genes_filtered, "genes_in_contigs.csv", sep = ",")




contigs_taxonomy <- unique(
  genes_filtered[, .(contig_ID, tax_id, genome_id, rank,
                    phylum, class, order, family, genus, species)]
)

fwrite(contigs_taxonomy, "contigs_taxonomy.csv", sep = ",")

# Sanity check
nrow(contigs_taxonomy)



library(Biostrings)



contigs_fa <- readDNAStringSet("contigs_min1kb.fasta")

# Read your taxonomy CSV
contigs_taxonomy <- fread("contigs_taxonomy.csv")

# Filter FASTA to matching contigs
keep_ids <- contigs_taxonomy$contig_ID
contigs_filtered_fa <- contigs_fa[names(contigs_fa) %in% keep_ids]

# Sanity check
length(contigs_filtered_fa)               # should be ~102,007
length(setdiff(keep_ids, names(contigs_fa)))  # should be 0 if all IDs are present

# Write out
writeXStringSet(contigs_filtered_fa, "contigs_filtered_postprodigal.fasta")






