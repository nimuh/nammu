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
### mags_annotated_90.R
filtering ORF annotation by diamond and formatting
### mags_annotated_aa_fna.R 
ORF annotations via KEGG including ORFs in both amino acid and nucleic acid format
