#!/bin/bash

# Set the input and output directories
input_dir="/anonymous/"
output_dir="/anonymous/"

# Make sure the output directory exists
mkdir -p "$output_dir"

# Loop through each .fasta file in the input directory
for mag_file in "$input_dir"/*.fa; do
  # Extract the base name of the MAG file (without path and extension)
  mag_name=$(basename "$mag_file" .fasta)

  # Define the Prodigal output file name
  output_file="$output_dir/${mag_name}_prodigal.faa"

  # Run the prodigal command (using double quotes for variable expansion)
  echo "prodigal -i \"$mag_file\" -a \"$output_file\" -p meta"
done

