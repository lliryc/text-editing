import logging
from typing import List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


def read_examples_from_file(file_path):
    guid_index = 1
    examples = {'subwords': [], 'edits': []}

    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples['subwords'].append(words)
                    examples['edits'].append(labels)
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split("\t")
                words.append(splits[0].strip())
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    # This is needed to get around the Trainer evaluation
                    # labels.append("K") # No Compress Experiment 
                    labels.append("K*") # place holder
        if words:
            examples['subwords'].append(words)
            examples['edits'].append(labels)

    dataset = Dataset.from_dict(examples)
    return dataset



def process(examples, edits_label_list: List[str], areta13_label_list: List[str], areta43_label_list: List[str], tokenizer: PreTrainedTokenizer):

    edits_label_map = {label: i for i, label in enumerate(edits_label_list)}
    areta13_label_map = {label: i for i, label in enumerate(areta13_label_list)}
    areta43_label_map = {label: i for i, label in enumerate(areta43_label_list)}

    examples_tokens = [words for words in examples['subwords']]

    examples_edits_labels = [labels for labels in examples['edits']]
    examples_areta13_labels = [labels for labels in examples['areta13']]
    examples_areta43_labels = [labels for labels in examples['areta43']]

    tokenized_inputs = [tokenizer.encode_plus(tokens, max_length=512, truncation=True) for tokens in examples_tokens]

    tokenized_inputs = {'subwords': [ex_tokens for ex_tokens in examples_tokens],
                        'edits': [ex_labels for ex_labels in examples_edits_labels],
                        'areta13': [ex_labels for ex_labels in examples_areta13_labels],
                        'areta43': [ex_labels for ex_labels in examples_areta43_labels],
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
    tokenized_inputs['areta13_labels'] = areta13_labels
    tokenized_inputs['areta43_labels'] = areta43_labels

    return tokenized_inputs


def get_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        labels = f.read().splitlines()

    return labels


