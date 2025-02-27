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



def process(examples, label_list: List[str], tokenizer: PreTrainedTokenizer):

    label_map = {label: i for i, label in enumerate(label_list)}

    examples_tokens = [words for words in examples['subwords']]

    examples_labels = [labels for labels in examples['edits']]

    tokenized_inputs = [tokenizer.encode_plus(tokens) for tokens in examples_tokens]

    tokenized_inputs = {'subwords': [ex_tokens for ex_tokens in examples_tokens],
                        'edits': [ex_labels for ex_labels in examples_labels],
                        'input_ids': [ex['input_ids'] for ex in tokenized_inputs],
                        'token_type_ids': [ex['token_type_ids'] for ex in tokenized_inputs],
                        'attention_mask': [ex['attention_mask'] for ex in tokenized_inputs]
                        }


    labels = []

    for i, ex_labels in enumerate(examples_labels):
        label_ids = []
        for word_idx in range(len(ex_labels)):
            label = ex_labels[word_idx]
            if label not in label_map:
                # label_ids.append(label_map['K']) # No Compress Exp
                label_ids.append(label_map['K*'])
            else:
                label_ids.append(label_map[label])
        
        label_ids = [-100] + label_ids + [-100]

        assert len(label_ids) == len(tokenized_inputs['input_ids'][i])

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels

    return tokenized_inputs


def get_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        labels = f.read().splitlines()

    return labels


class TokenClassificationDataset(torch.utils.data.Dataset):
    """A wrapper class for prediction dataset."""
    def __init__(self, examples, labels, tokenizer):
        self.tokenizer = tokenizer
        self.features = self.process_examples(examples, labels,
                                             pad_token_label_id=-100)

    def process_examples(self, examples, labels, pad_token_label_id=-100):
        label_map = {label: i for i, label in enumerate(labels)}

        examples_tokens = [words for words in examples['subwords']]

        examples_labels = [labels for labels in examples['edits']]

        featurized_inputs = []

        for ex_id, (example_tokens, example_labels) in enumerate(zip(examples_tokens, examples_labels)):
            tokens = []
            label_ids = []

            for word_tokens, label in zip(example_tokens, example_labels):
                if len(word_tokens) > 0:
                    tokens.append(word_tokens)

                    if label not in label_map: # OOV during test
                        label_ids.append([label_map['K']] * (len(word_tokens)))
                    else:
                        label_ids.append([label_map[label]] * (len(word_tokens)))

            token_segments = []
            token_segment = []
            label_ids_segments = []
            label_ids_segment = []
            num_word_pieces = 0
            seg_seq_length = self.tokenizer.model_max_length - 2

            for idx, word_pieces in enumerate(tokens):
                if num_word_pieces + len(word_pieces) > seg_seq_length:
                    # convert to ids and add special tokens

                    input_ids = self.tokenizer.convert_tokens_to_ids(token_segment)
                    input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

                    label_ids_segment = [pad_token_label_id] + label_ids_segment + [pad_token_label_id]


                    features = {'input_ids': input_ids,
                                'attention_mask': [1] * len(input_ids),
                                'token_type_ids': [0] * len(input_ids),
                                'labels': label_ids_segment,
                                'sent_id': ex_id
                                }

                    featurized_inputs.append(features)

                    token_segments.append(token_segment)
                    label_ids_segments.append(label_ids_segment)
                    token_segment = list(word_pieces)
                    label_ids_segment = list(label_ids[idx])
                    num_word_pieces = len(word_pieces)
                else:
                    token_segment.extend(word_pieces)
                    label_ids_segment.extend(label_ids[idx])
                    num_word_pieces += len(word_pieces)

            if len(token_segment) > 0:
                input_ids = self.tokenizer.convert_tokens_to_ids(token_segment)
                input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]


                label_ids_segment = [pad_token_label_id] + label_ids_segment + [pad_token_label_id]

                features = {'input_ids': input_ids,
                            'attention_mask': [1] * len(input_ids),
                            'token_type_ids': [0] * len(input_ids),
                            'labels': label_ids_segment,
                            'sent_id': ex_id
                            }

                featurized_inputs.append(features)

                token_segments.append(token_segment)
                label_ids_segments.append(label_ids_segment)

        return featurized_inputs

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

