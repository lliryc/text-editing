#!/usr/bin/env python3
"""
Script to extract subedits (first column) from subedit_class_label.tsv
and create a file with subedits only.
"""

import sys
import os

def extract_subedits(input_file, output_file):
    """
    Extract subedits (first column) from the combined TSV file.
    
    Args:
        input_file: Path to subedit_class_label.tsv
        output_file: Path to output file with subedits only
    """
    
    subedits = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')  # Only strip newline
            if line:  # Non-empty line
                parts = line.split('\t')
                if len(parts) >= 1:
                    subedits.append(parts[0])  # First column is subedit
                else:
                    subedits.append('')  # Empty line
            else:  # Empty line
                subedits.append('')
    
    print(f"Read {len(subedits)} subedits from {input_file}")
    
    # Write subedits to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for subedit in subedits:
            f.write(f"{subedit}\n")
    
    print(f"Successfully created {output_file} with {len(subedits)} lines")
    
    # Show first few lines as preview
    print("\nFirst 10 lines of output:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"{i+1:2d}: {line.strip()}")

if __name__ == "__main__":
    # File paths
    input_file = "subedit_class_label.tsv"
    output_file = "subedits_only.txt"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)
    
    # Extract subedits
    extract_subedits(input_file, output_file)



