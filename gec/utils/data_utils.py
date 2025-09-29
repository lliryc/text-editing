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


    labels = []

    for i, ex_labels in enumerate(examples_edits_labels):
        label_ids = []
        for word_idx in range(len(ex_labels)):
            label = ex_labels[word_idx]
            if label not in edits_label_map:
                # label_ids.append(label_map['K']) # No Compress Exp
                label_ids.append(edits_label_map['K*'])
            else:
                label_ids.append(edits_label_map[label])
        
        if len(label_ids) != len(tokenized_inputs['input_ids'][i]) - 2:
            assert len(label_ids) > len(tokenized_inputs['input_ids'][i]) - 2
            label_ids = label_ids[:len(tokenized_inputs['input_ids'][i]) - 2]

        label_ids = [-100] + label_ids + [-100]

        assert len(label_ids) == len(tokenized_inputs['input_ids'][i])

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels

    return tokenized_inputs


def get_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        labels = f.read().splitlines()

    return labels


