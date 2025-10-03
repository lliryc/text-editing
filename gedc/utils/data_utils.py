import logging
from typing import List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


def read_examples_from_file(file_path):
    guid_index = 1
    examples = {'subwords': [], 'edits': [], 'areta43': [], 'areta13': []}

    with open(file_path, encoding="utf-8") as f:
        words = []
        edits_labels = []
        areta13_labels = []
        areta43_labels = []
        for line in f:
          line = line.strip()
          splits = line.split("\t")
          words.append(splits[0].strip())
          if len(splits) > 3:
              edits_labels.append(splits[1].strip())
              areta43_labels.append(splits[2].strip())
              areta13_labels.append(splits[3].strip())
          else:

              edits_labels.append("K*") # place holder
              areta43_labels.append("UC") # place holder
              areta13_labels.append("UC") # place holder
        if words:
            examples['subwords'].append(words)
            examples['edits_labels'].append(edits_labels)
            examples['areta43_labels'].append(areta43_labels)
            examples['areta13_labels'].append(areta13_labels)

    dataset = Dataset.from_dict(examples)
    return dataset



def process(examples, edits_label_list: List[str], areta13_label_list: List[str], areta43_label_list: List[str], tokenizer: PreTrainedTokenizer):

    edits_label_map = {label: i for i, label in enumerate(edits_label_list)}
    areta13_label_map = {label: i for i, label in enumerate(areta13_label_list)}
    areta43_label_map = {label: i for i, label in enumerate(areta43_label_list)}

    examples_tokens = [words for words in examples['subwords']]

    examples_edits_labels = [labels for labels in examples['edits_labels']]
    examples_areta13_labels = [labels for labels in examples['areta13_labels']]
    examples_areta43_labels = [labels for labels in examples['areta43_labels']]

    tokenized_inputs = [tokenizer.encode_plus(tokens, max_length=512, truncation=True) for tokens in examples_tokens]

    tokenized_inputs = {'subwords': [ex_tokens for ex_tokens in examples_tokens],
                        'edits_labels': [ex_labels for ex_labels in examples_edits_labels],
                        'areta13_labels': [ex_labels for ex_labels in examples_areta13_labels],
                        'areta43_labels': [ex_labels for ex_labels in examples_areta43_labels],
                        'input_ids': [ex['input_ids'] for ex in tokenized_inputs],
                        'token_type_ids': [ex['token_type_ids'] for ex in tokenized_inputs],
                        'attention_mask': [ex['attention_mask'] for ex in tokenized_inputs]
                        }

    # Process labels for each type
    edits_labels = []
    areta13_labels = []
    areta43_labels = []

    for i in range(len(examples_edits_labels)):
        # Process edits labels
        edits_label_ids = []
        for word_idx in range(len(examples_edits_labels[i])):
            label = examples_edits_labels[i][word_idx]
            if label not in edits_label_map:
                edits_label_ids.append(edits_label_map['K*'])
            else:
                edits_label_ids.append(edits_label_map[label])
        
        if len(edits_label_ids) != len(tokenized_inputs['input_ids'][i]) - 2:
            assert len(edits_label_ids) > len(tokenized_inputs['input_ids'][i]) - 2
            edits_label_ids = edits_label_ids[:len(tokenized_inputs['input_ids'][i]) - 2]

        edits_label_ids = [-100] + edits_label_ids + [-100]
        assert len(edits_label_ids) == len(tokenized_inputs['input_ids'][i])
        edits_labels.append(edits_label_ids)

        # Process areta13 labels
        areta13_label_ids = []
        for word_idx in range(len(examples_areta13_labels[i])):
            label = examples_areta13_labels[i][word_idx]
            if label not in areta13_label_map:
                areta13_label_ids.append(areta13_label_map.get('K*', 0))  # Use 0 as default if K* not found
            else:
                areta13_label_ids.append(areta13_label_map[label])
        
        if len(areta13_label_ids) != len(tokenized_inputs['input_ids'][i]) - 2:
            assert len(areta13_label_ids) > len(tokenized_inputs['input_ids'][i]) - 2
            areta13_label_ids = areta13_label_ids[:len(tokenized_inputs['input_ids'][i]) - 2]

        areta13_label_ids = [-100] + areta13_label_ids + [-100]
        assert len(areta13_label_ids) == len(tokenized_inputs['input_ids'][i])
        areta13_labels.append(areta13_label_ids)

        # Process areta43 labels
        areta43_label_ids = []
        for word_idx in range(len(examples_areta43_labels[i])):
            label = examples_areta43_labels[i][word_idx]
            if label not in areta43_label_map:
                areta43_label_ids.append(areta43_label_map.get('K*', 0))  # Use 0 as default if K* not found
            else:
                areta43_label_ids.append(areta43_label_map[label])
        
        if len(areta43_label_ids) != len(tokenized_inputs['input_ids'][i]) - 2:
            assert len(areta43_label_ids) > len(tokenized_inputs['input_ids'][i]) - 2
            areta43_label_ids = areta43_label_ids[:len(tokenized_inputs['input_ids'][i]) - 2]

        areta43_label_ids = [-100] + areta43_label_ids + [-100]
        assert len(areta43_label_ids) == len(tokenized_inputs['input_ids'][i])
        areta43_labels.append(areta43_label_ids)

    # Add all label types to the output
    tokenized_inputs['edits_labels'] = edits_labels
    tokenized_inputs['areta43_labels'] = areta43_labels
    tokenized_inputs['areta13_labels'] = areta13_labels

    return tokenized_inputs


def get_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        labels = f.read().splitlines()

    return labels


