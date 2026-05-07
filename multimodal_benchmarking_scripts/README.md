# multimodal benchmarking

## 1. deep sea metagenome-assembled genomes 
In order to run the workflow for this task, the non-redundant metagenome-assembed genomes from Han et al., 2023 (10.1038/s41597-023-02521-4) need to be downloaded. They can be accessed through the paper's corresponding figshare repository (https://doi.org/10.6084/m9.figshare.22568107). 

### 1.1_prodigal_mags.sh 
script running prodigal to predict open reading frames for the nonredundant MAGs, output in faa format. 
### 1.2_prodigal_fna_mags.sh 
script running prodigal to predict open reading frames for nonredundant MAGs, output in fna format. 
### 1.3_faa_to_csv.sh 
converting faa output to .csv format
### 1.4_fna_to_csv.sh 
converting fna output to .csv format
### 1.5_diamond_run.sh 
script to run DIAMOND (https://doi.org/10.1038/s41592-021-01101-x) against the KEGG prokaryotic database. Requires KEGG to run. 
### 1.6_mags_annotated_90.R
filtering ORF annotation by diamond and formatting
### 1.7_mags_annotated_aa_fna.R 
ORF annotations via KEGG including ORFs in both amino acid and nucleic acid format

## 2. CAMI processing
In order to run the workflow for this task, the contigs and their corresponding taxa from sample 0 in 'simulated short read' from Critical Assessment of Metagenome Interpretation (CAMI) is required, and can be downloaded at https://cami-challenge.org. 

### 2.1_filter_prodigal.txt 
command used to filter contig file >1 Kbp and the command used to run prodigal against the updated filtered file
### 2.2_prodigal_processing.py 
script used to process prodigal output after open reading frame (ORF) predictions
### 2.3 resolve_taxonomy.sh 
script used to format and reconcile taxonomy of contigs 
### 2.4_processing_prodigal_output.R 
R script used to process prodigal output
### 2.5_prodigal_output_plots.R
Script to visualize contigs and their ORFs post processing 

## benchmark.py 
script used for benchmarking across tasks 

## create_mags.py
script used to reassemble contigs into mixed-modality schemes 


