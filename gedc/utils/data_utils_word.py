import logging
from typing import List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


def read_examples_from_file_words(file_path):
    guid_index = 1
    examples = {'words': [], 'edits': []}

    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples['words'].append(words)
                    examples['edits'].append(labels)
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split("\t")
                words.append(splits[0].replace("\n", ""))
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    # This is needed to get around the Trainer evaluation
                    # labels.append("K") # No Compress Exp
                    labels.append("K*") # place holder 
        if words:
            examples['words'].append(words)
            examples['edits'].append(labels)

    dataset = Dataset.from_dict(examples)
    return dataset


def process_words(examples, label_list: List[str], tokenizer: PreTrainedTokenizer):

    label_map = {label: i for i, label in enumerate(label_list)}

    examples_tokens = [words for words in examples['words']]

    tokenized_inputs = tokenizer(examples_tokens, max_length=512, truncation=True, is_split_into_words=True)

    labels = []
    examples_labels = [labels for labels in examples['edits']]

    for i, ex_labels in enumerate(examples_labels):

        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label = ex_labels[word_idx]
                if label not in label_map:
                    # label_ids.append(label_map['K']) # No Compress Exp
                    label_ids.append(label_map['K*'])
                else:
                    label_ids.append(label_map[label])

            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        if len(label_ids) != len(word_ids):
            label_ids = label_ids[1:-1] # taking out the padding for the special tokens
            label_ids = label_ids[:len(word_ids) - 2] # truncating up to len(word_ids) without special tokens
            label_ids = [-100] + label_ids + [-100] # putting the padding for special tokens back

        # assert len(label_ids) == len(word_ids)
        labels.append(label_ids)


    tokenized_inputs['labels'] = labels

    return tokenized_inputs

