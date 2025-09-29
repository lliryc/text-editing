#!/usr/bin/env python3
"""
Script to combine dev-edits.txt and qalb14-dev-subword-level-areta-13.txt
to create a TSV file with columns (subedit, class_label).
"""

import sys
import os

def combine_files(dev_edits_file, areta_file, output_file):
    """
    Combine two files with matching first columns to create output with (subedit, class_label).
    
    Args:
        dev_edits_file: Path to dev-edits.txt (subword, subedit)
        areta_file: Path to qalb14-dev-subword-level-areta-13.txt (subword, class_label)
        output_file: Path to output TSV file
    """
    
    # Read dev-edits.txt and store subedits
    subedits = []
    with open(dev_edits_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')  # Only strip newline, not all whitespace
            if line:  # Non-empty line
                parts = line.split('\t')
                if len(parts) >= 2:
                    subedits.append(parts[1])  # Second column is subedit
                else:
                    print(f"Warning: Unexpected format in dev-edits.txt: {line}")
                    subedits.append('')  # Add empty string for missing data
            else:  # Empty line
                subedits.append('')
    
    print(f"Read {len(subedits)} subedits from {dev_edits_file}")
    
    # Read areta file and combine with subedits
    class_labels = []
    with open(areta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')  # Only strip newline, not all whitespace
            if line:  # Non-empty line
                parts = line.split('\t')
                if len(parts) >= 2:
                    class_labels.append(parts[1])  # Second column is class_label
                else:
                    print(f"Warning: Unexpected format in areta file: {line}")
                    class_labels.append('')  # Add empty string for missing data
            else:  # Empty line
                class_labels.append('')
    
    print(f"Read {len(class_labels)} class labels from {areta_file}")
    
    # Determine the minimum length to avoid index errors
    min_length = min(len(subedits), len(class_labels))
    print(f"Combining {min_length} lines")
    
    # Write combined data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(min_length):
            f.write(f"{subedits[i]}\t{class_labels[i]}\n")
    
    print(f"Successfully created {output_file} with {min_length} lines")
    
    # Show first few lines as preview
    print("\nFirst 10 lines of output:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"{i+1:2d}: {line.strip()}")

if __name__ == "__main__":
    # File paths
    dev_edits_file = "data/msa-ged/modeling-13/pnx_sep/nopnx/dev/dev-edits.txt"
    areta_file = "data/msa-ged/modeling-13/pnx_sep/nopnx/dev/qalb14-dev-subword-level-areta-13.txt"
    output_file = "subedit_class_label.tsv"
    
    # Check if input files exist
    if not os.path.exists(dev_edits_file):
        print(f"Error: {dev_edits_file} not found")
        sys.exit(1)
    
    if not os.path.exists(areta_file):
        print(f"Error: {areta_file} not found")
        sys.exit(1)
    
    # Combine the files
    combine_files(dev_edits_file, areta_file, output_file)
