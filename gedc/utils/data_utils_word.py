import logging
from typing import List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


def read_examples_from_file_words(file_path):
    guid_index = 1
    examples = {'words': [], 'edits': [], 'areta13': [], 'areta43': []}

    with open(file_path, encoding="utf-8") as f:
        words = []
        edits_labels = []
        areta13_labels = []
        areta43_labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples['words'].append(words)
                    examples['edits'].append(edits_labels)
                    examples['areta13'].append(areta13_labels)
                    examples['areta43'].append(areta43_labels)
                    guid_index += 1
                    words = []
                    edits_labels = []
                    areta13_labels = []
                    areta43_labels = []
            else:
                splits = line.split("\t")
                words.append(splits[0].replace("\n", ""))
                if len(splits) >= 4:
                    edits_labels.append(splits[1].replace("\n", ""))
                    areta13_labels.append(splits[2].replace("\n", ""))
                    areta43_labels.append(splits[3].replace("\n", ""))
                elif len(splits) > 1:
                    # Fallback for files with only edit labels
                    edits_labels.append(splits[-1].replace("\n", ""))
                    areta13_labels.append("K*")  # placeholder
                    areta43_labels.append("K*")  # placeholder
                else:
                    # Examples could have no label for mode = "test"
                    # This is needed to get around the Trainer evaluation
                    edits_labels.append("K*")  # place holder
                    areta13_labels.append("K*")  # place holder
                    areta43_labels.append("K*")  # place holder
        if words:
            examples['words'].append(words)
            examples['edits'].append(edits_labels)
            examples['areta13'].append(areta13_labels)
            examples['areta43'].append(areta43_labels)

    dataset = Dataset.from_dict(examples)
    return dataset


def process_words(examples, edits_label_list: List[str], areta13_label_list: List[str], areta43_label_list: List[str], tokenizer: PreTrainedTokenizer):

    edits_label_map = {label: i for i, label in enumerate(edits_label_list)}
    areta13_label_map = {label: i for i, label in enumerate(areta13_label_list)}
    areta43_label_map = {label: i for i, label in enumerate(areta43_label_list)}

    examples_tokens = [words for words in examples['words']]

    tokenized_inputs = tokenizer(examples_tokens, max_length=512, truncation=True, is_split_into_words=True)

    # Process labels for each type
    edits_labels = []
    areta13_labels = []
    areta43_labels = []

    examples_edits_labels = [labels for labels in examples['edits']]
    examples_areta13_labels = [labels for labels in examples['areta13']]
    examples_areta43_labels = [labels for labels in examples['areta43']]

    for i in range(len(examples_edits_labels)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        
        # Process edits labels
        edits_label_ids = []
        areta13_label_ids = []
        areta43_label_ids = []

        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                edits_label_ids.append(-100)
                areta13_label_ids.append(-100)
                areta43_label_ids.append(-100)

            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                # Process edits label
                edits_label = examples_edits_labels[i][word_idx]
                if edits_label not in edits_label_map:
                    edits_label_ids.append(edits_label_map.get('K*', 0))
                else:
                    edits_label_ids.append(edits_label_map[edits_label])

                # Process areta13 label
                areta13_label = examples_areta13_labels[i][word_idx]
                if areta13_label not in areta13_label_map:
                    areta13_label_ids.append(areta13_label_map.get('K*', 0))
                else:
                    areta13_label_ids.append(areta13_label_map[areta13_label])

                # Process areta43 label
                areta43_label = examples_areta43_labels[i][word_idx]
                if areta43_label not in areta43_label_map:
                    areta43_label_ids.append(areta43_label_map.get('K*', 0))
                else:
                    areta43_label_ids.append(areta43_label_map[areta43_label])

            else:
                edits_label_ids.append(-100)
                areta13_label_ids.append(-100)
                areta43_label_ids.append(-100)

            previous_word_idx = word_idx

        # Handle truncation for all label types
        if len(edits_label_ids) != len(word_ids):
            edits_label_ids = edits_label_ids[1:-1]  # taking out the padding for the special tokens
            edits_label_ids = edits_label_ids[:len(word_ids) - 2]  # truncating up to len(word_ids) without special tokens
            edits_label_ids = [-100] + edits_label_ids + [-100]  # putting the padding for special tokens back

        if len(areta13_label_ids) != len(word_ids):
            areta13_label_ids = areta13_label_ids[1:-1]
            areta13_label_ids = areta13_label_ids[:len(word_ids) - 2]
            areta13_label_ids = [-100] + areta13_label_ids + [-100]

        if len(areta43_label_ids) != len(word_ids):
            areta43_label_ids = areta43_label_ids[1:-1]
            areta43_label_ids = areta43_label_ids[:len(word_ids) - 2]
            areta43_label_ids = [-100] + areta43_label_ids + [-100]

        edits_labels.append(edits_label_ids)
        areta13_labels.append(areta13_label_ids)
        areta43_labels.append(areta43_label_ids)

    # Add all label types to the output
    tokenized_inputs['edits_labels'] = edits_labels
    tokenized_inputs['areta13_labels'] = areta13_labels
    tokenized_inputs['areta43_labels'] = areta43_labels

    return tokenized_inputs

