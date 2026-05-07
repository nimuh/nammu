library(data.table)
library(ggplot2)
library(Biostrings)

setwd("~/anonymous/simulation_short_read/2018.08.15_09.49.32_sample_0/contigs/")

# ---- Load data ----
contigs_taxonomy <- fread("contigs_taxonomy.csv")
contigs_fa       <- readDNAStringSet("contigs_filtered_postprodigal.fasta")

# Build per-contig length table
length_dt <- data.table(
  contig_ID = names(contigs_fa),
  length    = width(contigs_fa)
)

# Join lengths with taxonomy
contig_info <- merge(contigs_taxonomy, length_dt, by = "contig_ID")

# Common theme
my_theme <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"))


# ---- Plot 1: Contig length distribution (log scale) ----
p1 <- ggplot(contig_info, aes(x = length)) +
  geom_histogram(bins = 60, fill = "steelblue", color = "white") +
  scale_x_log10(labels = scales::label_comma()) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Contig length distribution",
       subtitle = paste0(nrow(contig_info), " contigs, log-scale x-axis"),
       x = "Contig length (bp, log10)",
       y = "Number of contigs") +
  my_theme

ggsave("plot_contig_lengths.png", p1, width = 8, height = 5, dpi = 150)


# ---- Plot 2: Cumulative bases captured by length cutoff ----
sorted_lengths <- sort(contig_info$length, decreasing = TRUE)
cumulative <- data.table(
  length_threshold = sorted_lengths,
  cumulative_bases = cumsum(as.numeric(sorted_lengths)),
  n_contigs        = seq_along(sorted_lengths)
)

p2 <- ggplot(cumulative, aes(x = length_threshold, y = cumulative_bases / 1e6)) +
  geom_line(color = "darkred", size = 0.7) +
  scale_x_log10(labels = scales::label_comma()) +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Cumulative bases vs. minimum contig length",
       subtitle = "Useful for choosing a length filter threshold",
       x = "Minimum contig length (bp, log10)",
       y = "Total bases retained (Mb)") +
  my_theme

ggsave("plot_cumulative_bases.png", p2, width = 8, height = 5, dpi = 150)


# ---- Plot 3: Contigs per phylum ----
phylum_counts <- contig_info[, .N, by = phylum][order(-N)]

p3 <- ggplot(phylum_counts,
             aes(x = reorder(phylum, N), y = N)) +
  geom_col(fill = "seagreen") +
  coord_flip() +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Contigs per phylum",
       subtitle = paste0(nrow(phylum_counts), " phyla represented"),
       x = NULL, y = "Number of contigs") +
  my_theme

ggsave("plot_contigs_per_phylum.png", p3, width = 8, height = 5, dpi = 150)


# ---- Plot 4: Taxonomic breadth at each rank ----
breadth <- data.table(
  rank  = factor(c("Phylum", "Class", "Order",
                   "Family", "Genus", "Species"),
                 levels = c("Phylum", "Class", "Order",
                            "Family", "Genus", "Species")),
  count = c(
            contig_info[, uniqueN(phylum)],
            contig_info[, uniqueN(class)],
            contig_info[, uniqueN(order)],
            contig_info[, uniqueN(family)],
            contig_info[, uniqueN(genus)],
            contig_info[, uniqueN(species)])
)

p4 <- ggplot(breadth, aes(x = rank, y = count)) +
  geom_col(fill = "darkorange") +
  geom_text(aes(label = count), vjust = -0.4, size = 4) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Taxonomic breadth across contig set",
       subtitle = "Number of unique taxa at each rank",
       x = NULL, y = "Unique taxa") +
  my_theme

ggsave("plot_taxonomic_breadth.png", p4, width = 8, height = 5, dpi = 150)


# ---- Plot 5: Length distribution by phylum (top phyla only) ----
top_phyla <- phylum_counts[1:min(8, .N), phylum]
contig_top <- contig_info[phylum %in% top_phyla]
contig_top[, phylum := factor(phylum, levels = top_phyla)]

p5 <- ggplot(contig_top, aes(x = phylum, y = length)) +
  geom_boxplot(fill = "steelblue", alpha = 0.6, outlier.size = 0.3) +
  scale_y_log10(labels = scales::label_comma()) +
  coord_flip() +
  labs(title = "Contig length by phylum (top 8 phyla)",
       subtitle = "Log-scale; box = IQR, line = median",
       x = NULL, y = "Contig length (bp, log10)") +
  my_theme

ggsave("plot_length_by_phylum.png", p5, width = 8, height = 6, dpi = 150)


# ---- Plot 6: Top 20 most abundant species ----
species_counts <- contig_info[, .N, by = species][order(-N)][1:20]

p6 <- ggplot(species_counts, aes(x = reorder(species, N), y = N)) +
  geom_col(fill = "purple4") +
  coord_flip() +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Top 20 species by contig count",
       x = NULL, y = "Number of contigs") +
  my_theme +
  theme(axis.text.y = element_text(size = 9))

ggsave("plot_top_species.png", p6, width = 9, height = 6, dpi = 150)

cat("All plots written to current directory.\n")
