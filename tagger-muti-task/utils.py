import logging
import os
from enum import Enum
from typing import List, Union
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from dataclasses import dataclass


logger = logging.getLogger(__name__)


def read_examples_from_file(file_path):
    guid_index = 1
    examples = {'subwords': [], 'edits': [], 'bin': []}

    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        binary_labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples['subwords'].append(words)
                    examples['edits'].append(labels)
                    examples['bin'].append(binary_labels)
                    guid_index += 1
                    words = []
                    labels = []
                    binary_labels = []
            else:
                splits = line.split("\t")
                words.append(splits[0].strip())
                if len(splits) > 1:
                    labels.append(splits[1].strip())
                    binary_labels.append(splits[2].strip())
                else:
                    # Examples could have no label for mode = "test"
                    # This is needed to get around the Trainer evaluation
                    labels.append("K*") # place holder
                    binary_labels.append("C")
        if words:
            examples['subwords'].append(words)
            examples['edits'].append(labels)
            examples['bin'].append(binary_labels)

    dataset = Dataset.from_dict(examples)
    return dataset



def process(examples, label_list: List[str], bin_label_list: List[str], 
            tokenizer: PreTrainedTokenizer):

    label_map = {label: i for i, label in enumerate(label_list)}

    bin_label_map = {label: i for i, label in enumerate(bin_label_list)}

    examples_tokens = [words for words in examples['subwords']]

    examples_labels = [labels for labels in examples['edits']]

    examples_binary_labels = [labels for labels in examples['bin']]

    tokenized_inputs = [tokenizer.encode_plus(tokens) for tokens in examples_tokens]

    tokenized_inputs = {'subwords': [ex_tokens for ex_tokens in examples_tokens],
                        'edits': [ex_labels for ex_labels in examples_labels],
                        'bin': [ex_bin_labels for ex_bin_labels in examples_binary_labels],
                        'input_ids': [ex['input_ids'] for ex in tokenized_inputs],
                        'token_type_ids': [ex['token_type_ids'] for ex in tokenized_inputs],
                        'attention_mask': [ex['attention_mask'] for ex in tokenized_inputs]
                        }


    labels = []
    bin_labels = []

    for i, (ex_labels, ex_bin_labels) in enumerate(zip(examples_labels, examples_binary_labels)):
        label_ids = []
        label_bin_ids = []

        for word_idx in range(len(ex_labels)):
            label = ex_labels[word_idx]
            bin_label = ex_bin_labels[word_idx]
            if label not in label_map:
                label_ids.append(label_map['K*'])
            else:
                label_ids.append(label_map[label])
            label_bin_ids.append(bin_label_map[bin_label])

        label_ids = [-100] + label_ids + [-100]
        label_bin_ids = [-100] + label_bin_ids + [-100]
        assert len(label_ids) == len(tokenized_inputs['input_ids'][i]) == len(label_bin_ids)

        labels.append(label_ids)
        bin_labels.append(label_bin_ids)

    tokenized_inputs['labels'] = labels
    tokenized_inputs['binary_labels'] = bin_labels

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


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

@dataclass
class DataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        binary_labels = [feature["binary_labels"] for feature in features] if "binary_labels" in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k not in ["labels", "binary_labels"]} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None and binary_labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch["labels"] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
            batch["binary_labels"] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in binary_labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]
            batch["binary_labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in binary_labels
            ]

        batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        batch["binary_labels"] = torch.tensor(batch["binary_labels"], dtype=torch.int64)

        return batch
    

def load_ged_labels(path):
    sent_labels = []
    labels = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                sent_labels.append(line)
            else:
                labels.append(sent_labels)
                sent_labels = []
        
        if sent_labels:
            labels.append(sent_labels)
    return labels
