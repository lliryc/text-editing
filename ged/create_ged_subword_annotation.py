import pandas as pd
import argparse
import traceback
from camel_tools.utils.charsets import UNICODE_PUNCT_CHARSET
import string
import re


def merge_annotations(areta_annotation_word_level_df, edits_annotation_subword_level_df, output_label_column, output_areta_annotation_subword_level_file, pnx=False):
  # create datasource for the new data frame from tuples list
  subword_error_list = []
  
  if len(areta_annotation_word_level_df) == 0 or len(edits_annotation_subword_level_df) == 0:
    raise ValueError("Empty input data frames")
  
  subword_pos = 0
  word_pos = 0
  
  current_word = areta_annotation_word_level_df.at[word_pos, "Input"]
  current_error_label = areta_annotation_word_level_df.at[word_pos, output_label_column]
  
  while subword_pos < len(edits_annotation_subword_level_df) and word_pos < len(areta_annotation_word_level_df):
    
     if edits_annotation_subword_level_df.at[subword_pos, "Input"] == '=':
        subword_pos += 1
        subword_error_list.append(('=', 'UC'))
        continue
      
     if edits_annotation_subword_level_df.at[subword_pos, "Input"] == '' or \
     edits_annotation_subword_level_df.at[subword_pos, "Input"] == '\n' or \
      pd.isna(edits_annotation_subword_level_df.at[subword_pos, "Input"])  :
        subword_pos += 1
        subword_error_list.append((None, None))
        continue
      
     original_subword = edits_annotation_subword_level_df.at[subword_pos, "Input"]
     
     current_subword = original_subword

     if current_subword.startswith('0') and len(current_subword) >= 2 and current_subword.isdigit():
       print(f"Found subword starting with 0 and consisting of at least 2 digits: '{current_subword}'")
       current_subword = current_subword[1:]

     current_subword = current_subword.replace('##', '')
     
     if current_word is None:
       while areta_annotation_word_level_df.at[word_pos, 'Input'] == '#ERROR!':
          word_pos += 1
          if word_pos == len(areta_annotation_word_level_df):
            break
       if word_pos == len(areta_annotation_word_level_df):
         break
       current_word = areta_annotation_word_level_df.at[word_pos, 'Input']
       current_error_label = areta_annotation_word_level_df.at[word_pos, output_label_column]
       
     if current_subword == current_word[:len(current_subword)]:
       subword_error_list.append((original_subword, current_error_label))
       subword_pos += 1
       current_word = current_word[len(current_subword):]
       if current_word == '':
         word_pos += 1
         current_word = None
         current_error_label = None
     else:
       # Skip mismatched entries instead of failing
       print(f"Warning: Skipping mismatch at word_pos={word_pos}, subword_pos={subword_pos}")
       print(f"Current word='{repr(current_word)}', current_subword='{repr(current_subword)}'")
       
       if (not current_subword.isdigit()) and current_subword in current_word:
         subword_pos += 1
         word_pos += 1
         subword_error_list.append((original_subword, current_error_label))
         current_word = None
         current_error_label = None
         continue
       
       # Try to advance both positions to find the next match
       elif (current_word.isdigit() and current_subword.isdigit()) and (int(current_word) == int(current_subword)):
         subword_pos += 1
         word_pos += 1
         subword_error_list.append((original_subword, current_error_label))
         current_word = None
         current_error_label = None

       else:
        subword_error_list.append((original_subword, 'UC'))
        subword_pos += 1

  with open(output_areta_annotation_subword_level_file, 'w') as f:
    for subword, error_label in subword_error_list:
      if subword is None:
        f.write('\n')
      else:
        f.write(f"{subword}\t{error_label}\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_areta_annotation_word_level", default="/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-ged/raw/dev/qalb14-dev-word-level-areta.tsv", type=str, required=True)
  parser.add_argument("--input_edits_annotation_subword_level", default="/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-gec/modeling/qalb14/pnx_sep/qalb14-nopnx/dev.txt", type=str, required=True)
  parser.add_argument("--output_areta_annotation_subword_level", default="/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-ged/modeling-13/pnx_sep/nopnx/dev/qalb14-dev-subword-level-areta-13.txt", type=str, required=True)
  parser.add_argument("--output_label_column", default="Error Type (Coarse)", type=str, required=True)
  args = parser.parse_args()

  areta_annotation_word_level_df = pd.read_csv(args.input_areta_annotation_word_level, sep="\t", quoting=3)
  areta_annotation_word_level_df = areta_annotation_word_level_df.dropna(subset=["Input"])
  # Remove rows where Input column contains only whitespace/tabs
  areta_annotation_word_level_df = areta_annotation_word_level_df[~areta_annotation_word_level_df["Input"].str.strip().eq("")]
  areta_annotation_word_level_df = areta_annotation_word_level_df[["Input", "Edit"] + [args.output_label_column]]
  # Reset index to ensure consecutive indices after filtering
  areta_annotation_word_level_df = areta_annotation_word_level_df.reset_index(drop=True)
  
  edits_annotation_subword_level_df = pd.read_csv(args.input_edits_annotation_subword_level, sep="\t", names=["Input"], header=None, quoting=3, skip_blank_lines=False)

  merge_annotations(areta_annotation_word_level_df, edits_annotation_subword_level_df, args.output_label_column, args.output_areta_annotation_subword_level)
