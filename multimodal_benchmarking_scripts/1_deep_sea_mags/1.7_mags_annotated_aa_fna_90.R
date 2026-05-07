library(dplyr)

# Set working directory
setwd("~/anonymous/deep_sea_MAGs_public/")

# Load the full Prodigal nucleotide data
full_data <- read.csv("MAGs_combined_prodigal_fna.csv", header = FALSE, stringsAsFactors = FALSE,
                      col.names = c("Sequence_ID", "Start", "End", "Strand", 
                                    "ID", "Partial", "Start_Type", 
                                    "RBS_Motif", "RBS_Spacer", 
                                    "GC_Content", "Sequence_nucleotide"))

# File prefixes to iterate over
cycles <- c(90)

# Loop through each file
for (cycle in cycles) {
  input_file <- paste0("MAGS_annotated_", cycle, ".csv")
  output_file <- paste0("MAGS_annotated_", cycle, "_aa_fna.csv")
  
  # Read in annotated file
  mags_df <- read.csv(input_file)
  
  # Merge with prodigal nucleotide data
  merged_df <- left_join(mags_df, full_data, by = c("Sequence_ID", "Start", "End", "Strand", 
                                                    "ID", "Partial", "Start_Type", 
                                                    "RBS_Motif", "RBS_Spacer", 
                                                    "GC_Content"))
  
  cat("\nColumns in", input_file, ":\n")
  cat(paste(colnames(merged_df), collapse = ", "), "\n\n")
  
  merged_df <- select(merged_df, CSMAG_ID, Sequence_ID, contig_ID, Start, End, Strand, ID, Partial, Start_Type, 
                      RBS_Motif, RBS_Spacer, GC_Content, Sequence, Sequence_nucleotide, Gene, Percent_Identity, 
                      Alignment_Length, Mismatches, Gap_Openings, Query_Start, Query_End, Subject_Start, 
                      Subject_End, E_value, Bit_Score, KO, Length, Domains, Modules)
  

  write.csv(merged_df, output_file, row.names = FALSE)
}





