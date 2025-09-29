import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ARETA_13_LABELS_PREDICTED = "data/msa-ged/testing-13/pnx_sep/nopnx/qalb14-dev-subword-level-areta-13-labels.txt"
ARETA_13_SUBWORDS_LABELS_TRUE = "data/msa-ged/modeling-13/pnx_sep/nopnx/dev/qalb14-dev-subword-level-areta-13.txt"

ARETA_43_LABELS_PREDICTED = "data/msa-ged/testing-43/pnx_sep/nopnx/qalb14-dev-subword-level-areta-43-labels.txt"
ARETA_43_SUBWORDS_LABELS_TRUE = "data/msa-ged/modeling-43/pnx_sep/nopnx/dev/qalb14-dev-subword-level-areta-43.txt"

if __name__ == "__main__":
  # ARETA 13 processing
  areta_13_subwords_labels_true_df = pd.read_csv(ARETA_13_SUBWORDS_LABELS_TRUE, sep="\t", header=None, quoting=3, names=['Input', 'Label'], skip_blank_lines=True)
  areta_13_labels_predicted_df = pd.read_csv(ARETA_13_LABELS_PREDICTED, sep="\t", header=None, quoting=3, names=['Predicted'], skip_blank_lines=True)
  areta_13_subwords_labels_true_df['Predicted'] = areta_13_labels_predicted_df['Predicted'].values
  
  # ARETA 13 metrics calculation
  areta_13_accuracy = accuracy_score(areta_13_subwords_labels_true_df['Label'], areta_13_subwords_labels_true_df['Predicted'])
  areta_13_precision = precision_score(areta_13_subwords_labels_true_df['Label'], areta_13_subwords_labels_true_df['Predicted'], average='macro', zero_division=0)
  areta_13_recall = recall_score(areta_13_subwords_labels_true_df['Label'], areta_13_subwords_labels_true_df['Predicted'], average='macro', zero_division=0)
  areta_13_f1 = f1_score(areta_13_subwords_labels_true_df['Label'], areta_13_subwords_labels_true_df['Predicted'], average='macro', zero_division=0)
  
  print(f"ARETA 13 Accuracy (qalb14): {areta_13_accuracy}")
  print(f"ARETA 13 Precision (qalb14): {areta_13_precision}")
  print(f"ARETA 13 Recall (qalb14): {areta_13_recall}")
  print(f"ARETA 13 F1 (qalb14): {areta_13_f1}")
  
  print("-" * 100)
  
  # ARETA 43 processing
  areta_43_labels_predicted_df = pd.read_csv(ARETA_43_LABELS_PREDICTED, sep="\t", header=None, quoting=3, names=['Predicted'], skip_blank_lines=True)
  areta_43_subwords_labels_true_df = pd.read_csv(ARETA_43_SUBWORDS_LABELS_TRUE, sep="\t", header=None, quoting=3, names=['Input', 'Label'], skip_blank_lines=True)
  areta_43_subwords_labels_true_df['Predicted'] = areta_43_labels_predicted_df['Predicted'].values
  
  # ARETA 43 metrics calculation
  areta_43_accuracy = accuracy_score(areta_43_subwords_labels_true_df['Label'], areta_43_subwords_labels_true_df['Predicted'])
  areta_43_precision = precision_score(areta_43_subwords_labels_true_df['Label'], areta_43_subwords_labels_true_df['Predicted'], average='macro', zero_division=0)
  areta_43_recall = recall_score(areta_43_subwords_labels_true_df['Label'], areta_43_subwords_labels_true_df['Predicted'], average='macro', zero_division=0)
  areta_43_f1 = f1_score(areta_43_subwords_labels_true_df['Label'], areta_43_subwords_labels_true_df['Predicted'], average='macro', zero_division=0)
  
  print(f"ARETA 43 Accuracy (qalb14): {areta_43_accuracy}")
  print(f"ARETA 43 Precision (qalb14): {areta_43_precision}")
  print(f"ARETA 43 Recall (qalb14): {areta_43_recall}")
  print(f"ARETA 43 F1 (qalb14): {areta_43_f1}")
  

