import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from gedc.utils.data_utils import get_labels, process, read_examples_from_file
from gedc.utils.data_utils_word import process_words, read_examples_from_file_words

from gedc.model import BertForTokenMultiLabelClassification
import os
import sys
import json

from gedc.utils.postprocess import remove_pnx, pnx_tokenize, space_clean
from edits.edit import SubwordEdit

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are
    going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="aubmindlab/bert-base-arabertv02",
        metadata={"help": "Path to pretrained model or model identifier from "
                          "huggingface.co/models"}
    )
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if "
                                        "not the same as model_name"}
    )
    
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if "
                                        "not the same as model_name"}
    )

    use_fast: bool = field(default=False, metadata={"help": "Set this flag to "
                                                            "use fast "
                                                            "tokenization."})

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the "
                                        "pretrained models downloaded from s3"}
    )

    add_class_weights: bool = field(
        default=False, metadata={"help": "Whether to weigh classes during "
                                        "training or not."}
    )

    topk_pred: bool = field(
        default=False, metadata={"help": "Whether to get topk predictions or not"}
    )

    output_probs: bool = field(
        default=False, metadata={"help": "Whether to output prediction probabilities or not"}
    )

    debug_mode: bool = field(
        default=False, metadata={"help": "Run in debug mode or not"}
    )

    pred_threshold: float = field(
        default=None, metadata={"help": "Top prediction threshold before defaulting to K*"}
    )

    input_unit: str = field(
        default='subword-level', metadata={"help": "The input unit of the edit: subword-level or word-level"}
    )
    
    task: str = field(
        default='msa-ged', metadata={"help": "The task type: msa-ged or da-ged"}
    )

    continue_train: bool = field(
        default=False, metadata={"help": "Whether to continue training or not."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    tokenized_data_path: str = field(
        default='/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-gedc/modeling-mh/pnx_sep/nopnx/train/qalb14-train-subword-level-multi-heads.txt',
        metadata={"help": "The input file path."}
    )
    tokenized_raw_data_path: str = field(
        default='/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-gedc/modeling-mh/pnx_sep/nopnx/dev/qalb14-dev-subword-level-multi-heads.raw.txt', 
        metadata={"help": "The input file path."}
    )
    edit_labels: Optional[str] = field(
        default='/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-gedc/modeling-mh/pnx_sep/nopnx/train/edit-labels.txt',
        metadata={"help": "Path to a file containing all edit labels."},
    )
    areta13_labels: Optional[str] = field(
        default='/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-gedc/modeling-mh/pnx_sep/nopnx/train/areta13-labels.txt',
        metadata={"help": "Path to a file containing all areta13 labels."},
    )
    areta43_labels: Optional[str] = field(
        default='/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-gedc/modeling-mh/pnx_sep/nopnx/train/areta43-labels.txt',
        metadata={"help": "Path to a file containing all areta43 labels."},
    )
    label_pred_output_file: Optional[str] = field(
        default='ouput_data_dir', 
        metadata={"help": "Label predictions output file."}
    )
    rewrite_pred_output_file: Optional[str] = field(
        default=None, metadata={"help": "Rewrite predictions output file."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments,
                               DataTrainingArguments,
                               TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a
        # json file, let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
                                                    json_file=os.path.abspath(
                                                                 sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    training_args.no_cuda=True
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists "
            "and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=(logging.INFO if training_args.local_rank in [-1, 0]
               else logging.WARN),
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, "
        "16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    edit_labels = get_labels(data_args.edit_labels)
    areta13_labels = get_labels(data_args.areta13_labels)
    areta43_labels = get_labels(data_args.areta43_labels)
    edits_label_map: Dict[int, str] = {i: label for i, label in enumerate(edit_labels)}
    areta13_label_map: Dict[int, str] = {i: label for i, label in enumerate(areta13_labels)}
    areta43_label_map: Dict[int, str] = {i: label for i, label in enumerate(areta43_labels)}
    edits_num_labels = len(edit_labels)
    areta13_num_labels = len(areta13_labels)
    areta43_num_labels = len(areta43_labels)

    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name
            else model_args.model_name_or_path),
        edits_num_labels=edits_num_labels,
        areta13_num_labels=areta13_num_labels,
        areta43_num_labels=areta43_num_labels,
        id2edits_label=edits_label_map,
        id2areta13_label=areta13_label_map,
        id2areta43_label=areta43_label_map,
        cache_dir=model_args.cache_dir,
    )
    
    config.edits_num_labels = 4800
    config.areta13_num_labels = 13
    config.areta43_num_labels = 43
    
    config.loss_weight_edits = 1.0
    config.loss_weight_areta13 = 1.0
    config.loss_weight_areta43 = 1.0
    
    config.label_smoothing = 0.0
    
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name
            else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=False if model_args.input_unit == 'subword-level' else True, 
        model_max_length=512
    )

    edits_class_weights = None
    areta13_class_weights = None
    areta43_class_weights = None

    if training_args.do_train:
        if model_args.input_unit == 'subword-level':
            train_dataset = read_examples_from_file(file_path=data_args.tokenized_data_path)
        else:
            train_dataset = read_examples_from_file_words(file_path=data_args.tokenized_data_path)

        if model_args.add_class_weights:
            edits_class_weights = compute_class_weights(training_data=train_dataset, weight_threshold=500,
                                                  labels_map={label: i for i, label in enumerate(edit_labels)})
            areta13_class_weights = compute_class_weights(training_data=train_dataset, weight_threshold=500,
                                                  labels_map={label: i for i, label in enumerate(areta13_labels)})
            areta43_class_weights = compute_class_weights(training_data=train_dataset, weight_threshold=500,
                                                  labels_map={label: i for i, label in enumerate(areta43_labels)})

        if model_args.input_unit == 'subword-level':
            train_dataset = train_dataset.map(process,
                        fn_kwargs={"edits_label_list": edit_labels, "areta13_label_list": areta13_labels, "areta43_label_list": areta43_labels, "tokenizer": tokenizer},
                        batched=True,
                        desc="Running tokenizer on train dataset"
                        )
        else:
            train_dataset = train_dataset.map(process_words,
                        fn_kwargs={"edits_label_list": edit_labels, "areta13_label_list": areta13_labels, "areta43_label_list": areta43_labels, "tokenizer": tokenizer},
                        batched=True,
                        desc="Running tokenizer on train dataset"
                        )

    model = BertForTokenMultiLabelClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        edits_class_weights=edits_class_weights,
        areta13_class_weights=areta13_class_weights,
        areta43_class_weights=areta43_class_weights
    )


    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=(model_args.model_name_or_path
                        if os.path.isdir(model_args.model_name_or_path)
                        else None)
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)


    # Predict
    if training_args.do_predict:
        if model_args.input_unit == 'subword-level':
            test_dataset =  read_examples_from_file(file_path=data_args.tokenized_data_path)
            raw_test_dataset =  read_examples_from_file(file_path=data_args.tokenized_raw_data_path)
            test_dataset = test_dataset.map(process,
                                            fn_kwargs={"edits_label_list": edit_labels, "areta13_label_list": areta13_labels, "areta43_label_list": areta43_labels, "tokenizer": tokenizer},
                                            batched=True,
                                            desc=f"Running tokenizer on {data_args.tokenized_data_path}")
        else:
            test_dataset =  read_examples_from_file_words(file_path=data_args.tokenized_data_path)
            raw_test_dataset =  read_examples_from_file_words(file_path=data_args.tokenized_raw_data_path)
            test_dataset = test_dataset.map(process_words,
                                            fn_kwargs={"edits_label_list": edit_labels, "areta13_label_list": areta13_labels, "areta43_label_list": areta43_labels, "tokenizer": tokenizer},
                                            batched=True,
                                            desc=f"Running tokenizer on {data_args.tokenized_data_path}")


        pred_edits, preds_probs = predict(model=model, test_dataset=test_dataset,
                                          collate_fn=data_collator,
                                          topk_pred=model_args.topk_pred,
                                          pred_threshold=model_args.pred_threshold,
                                          label_map=edits_label_map,
                                          batch_size=training_args.per_device_eval_batch_size)

        if data_args.label_pred_output_file:
            label_pred_output_file = os.path.join(training_args.output_dir,
                                                data_args.label_pred_output_file)

            if not model_args.debug_mode:
                if trainer.is_world_process_zero():
                    with open(label_pred_output_file, "w") as writer:
                        if model_args.output_probs:
                            for example, example_probs in zip(pred_edits, preds_probs):
                                for label, prob in zip(example, example_probs):
                                    writer.write(f'{label}\t{prob}')
                                    writer.write('\n')
                                writer.write('\n')
                        else:
                            for example in pred_edits:
                                for label in example:
                                    writer.write(label)
                                    writer.write('\n')
                                writer.write('\n')

        if model_args.topk_pred and model_args.debug_mode:
            pred_rewrites = rewrite_topk(subwords=test_dataset['subwords'], edits=pred_edits)
            rewrite_pred_output_file = os.path.join(training_args.output_dir,
                                                    data_args.rewrite_pred_output_file)
            with open(rewrite_pred_output_file, "w", encoding="utf-8") as writer:
                writer.write('Top 1 Pred\tTop 2 Pred\tTop 3 Pred\tTop 4 Pred\tTop 5 Pred\t\tTop 1 Prob\t'
                             f'Top 2 Prob\tTop 3 Prob\tTop 4 Prob\tTop 5 Prob\tEdits\t\n')

                for i in range(len(pred_rewrites)):
                    sent_edits, sent_probs, sent_rewrites = pred_edits[i], preds_probs[i], pred_rewrites[i]
                    assert len(sent_edits) == len(sent_probs) == len(sent_rewrites)
                    for rewrites, edits, probs in zip(sent_rewrites, sent_edits, sent_probs):
                        probs_output = '\t'.join([f'<s>{prob:.2f}<s>' for prob in probs])
                        rewrites_output = '\t'.join([f'<s>{rewrite}<s>' for rewrite in rewrites])
                        writer.write(f'{rewrites_output}\t\t{probs_output}\t{edits}')
                        writer.write('\n')
                    writer.write('\n')

        else:
            detok_pred_rewrites, pred_rewrites, non_app_edits = rewrite(subwords=(raw_test_dataset['subwords']
                                                                                 if model_args.input_unit == 'subword-level'
                                                                                 else  raw_test_dataset['words']),
                                                                        edits=pred_edits)

            # Clean generated output by separating pnx and extra white space
            if model_args.task == 'msa-gedc':
                detok_pred_rewrites = pnx_tokenize(detok_pred_rewrites)
            else:
                detok_pred_rewrites = space_clean(detok_pred_rewrites)

            rewrite_pred_output_file = os.path.join(training_args.output_dir,
                                                    data_args.rewrite_pred_output_file)

            with open(rewrite_pred_output_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(detok_pred_rewrites))
                writer.write("\n")

            with open(rewrite_pred_output_file+'.subwords', "w", encoding="utf-8") as writer:
                for subwords in pred_rewrites:
                    for subword in subwords:
                        writer.write(subword)
                        writer.write('\n')
                    writer.write('\n')

            with open(rewrite_pred_output_file+'.na_edits', "w", encoding="utf-8") as writer:
                for edit in non_app_edits:
                    writer.write(json.dumps(edit, ensure_ascii=False))
                    writer.write("\n")

            # removing the pnx from the generated outputs
            detok_pred_rewrites_nopnx = remove_pnx(detok_pred_rewrites)

            with open(rewrite_pred_output_file+'.nopnx', "w", encoding="utf-8") as writer:
                writer.write("\n".join(detok_pred_rewrites_nopnx))
                writer.write("\n")



def predict(model, test_dataset, collate_fn, label_map, topk_pred=True,
            batch_size=32, pred_threshold=None):
    logger.info(f"***** Running Prediction *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Batch size = {batch_size}")

    if 'subwords' in test_dataset.column_names:
        _test_dataset = test_dataset.remove_columns(['subwords', 'edits'])
    elif 'words' in test_dataset.column_names:
         _test_dataset = test_dataset.remove_columns(['words', 'edits'])

    data_loader = DataLoader(_test_dataset,
                             batch_size=batch_size,
                             shuffle=False, drop_last=False, collate_fn=collate_fn)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    preds = []
    preds_probs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            inputs = {'input_ids': batch['input_ids'],
                      'token_type_ids': batch['token_type_ids'],
                      'attention_mask': batch['attention_mask']}

            label_ids = batch['labels']

            logits = model(**inputs)[0]

            if topk_pred:
                predictions, probs = _align_predictions_topk(logits,
                                                             label_ids,
                                                             label_map,
                                                             k=5)

            else:
                predictions, probs = _align_predictions(logits,
                                                        label_ids,
                                                        label_map,
                                                        pred_threshold=pred_threshold)


            preds.extend(predictions)
            preds_probs.extend(probs)

    assert len(preds) == len(preds_probs)
    return preds, preds_probs


def _align_predictions(predictions, label_ids, label_map, pred_threshold=None):
    """Aligns the predictions of the model with the inputs and it takes
    care of getting rid of the padding token.
    Args:
        predictions (:obj:`np.ndarray`): The predictions of the model
        label_ids (:obj:`np.ndarray`): The label ids of the inputs.
            They will always be the ids of Os since we're dealing with a
            test dataset. Note that label_ids are also padded.
    Returns:
        :obj:`list` of :obj:`list` of :obj:`str`: The predicted labels for
        all the sentences in the batch
    """

    batch_size, seq_len, num_labels = predictions.shape
    predictions = torch.nn.functional.softmax(predictions, dim=-1)

    preds = torch.argmax(predictions, dim=-1).cpu().numpy()
    probs, topk_preds = torch.topk(predictions, k=5)
    probs = probs.cpu().numpy()
    topk_preds = topk_preds.cpu().numpy()
    probs = probs[:, :, 0]
    assert probs.shape == preds.shape


    preds_list = [[] for _ in range(batch_size)]
    probs_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                if pred_threshold:
                    if probs[i][j] < pred_threshold:
                        preds_list[i].append('K*')
                    else:
                        preds_list[i].append(label_map[preds[i][j]])
                else:
                    preds_list[i].append(label_map[preds[i][j]])

                probs_list[i].append(probs[i][j])

    return preds_list, probs


def _align_predictions_topk(predictions, label_ids, label_map, k=10):
    """Aligns the predictions of the model with the inputs and it takes
    care of getting rid of the padding token.
    Args:
        predictions (:obj:`np.ndarray`): The predictions of the model
        label_ids (:obj:`np.ndarray`): The label ids of the inputs.
            They will always be the ids of Os since we're dealing with a
            test dataset. Note that label_ids are also padded.
    Returns:
        :obj:`list` of :obj:`list` of :obj:`str`: The predicted labels for
        all the sentences in the batch
    """
    batch_size, seq_len, num_labels = predictions.shape
    predictions = torch.nn.functional.softmax(predictions, dim=-1)

    topk_probs, topk_indices = torch.topk(predictions, k=k, dim=-1)

    batch_size, seq_len = topk_indices.shape[0], topk_indices.shape[1]
    preds_list = [[] for _ in range(batch_size)]
    probs_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append([label_map[idx.item()] for idx in topk_indices[i][j]])
                probs_list[i].append([prob.cpu().item() for prob in topk_probs[i][j]])

    return preds_list, probs_list


def applicable_preds(sentences, preds):
    applicable_preds = []

    assert len(sentences) == len(preds)
    for subwords, subwords_edits in zip(sentences, preds):
        assert len(subwords) == len(subwords_edits)
        app_edits = []
        for subword, edits in zip(subwords, subwords_edits):
            applicable_edit = None
            for edit in edits:
                subword_edit = SubwordEdit(subword, edit)
                if subword_edit.is_applicable(subword):
                    applicable_edit = edit
                    break

            if applicable_edit is None:
                import pdb; pdb.set_trace()
                applicable_edit = 'K'

            app_edits.append(applicable_edit)

        assert len(app_edits) == len(subwords)
        applicable_preds.append(app_edits)

    assert len(applicable_preds) == len(sentences)
    return applicable_preds


def rewrite(subwords, edits):
    assert len(subwords) == len(edits)

    rewritten_sents = []
    rewritten_sents_merge = []
    non_app_edits = []

    for i, (sent_subwords, sent_edits) in enumerate(zip(subwords, edits)):
        if len(sent_subwords) != len(sent_edits):
            # In case the predicted edits are less then the subwords 
            # because of truncation, add keeps
            assert len(sent_subwords) > len(sent_edits)
            sent_edits += ['K*'] * (len(sent_subwords) - len(sent_edits))

        # assert len(sent_subwords) == len(sent_edits)
        rewritten_sent = []

        for subword, edit in zip(sent_subwords, sent_edits):
            edit = SubwordEdit(subword=subword, raw_subword=subword, edit=edit)

            if edit.is_applicable(subword):
                rewritten_subword = edit.apply(subword)
                rewritten_sent.append(rewritten_subword)
            else:
                non_app_edits.append({'subword': subword, 'edit': edit.to_json_str()})
                rewritten_sent.append(subword)

        rewritten_sents_merge.append(resolve_merges(rewritten_sent, sent_edits))
        rewritten_sents.append(rewritten_sent)


    detok_rewritten_sents = [detokenize_sent(sent) for sent in rewritten_sents_merge]
    return detok_rewritten_sents, rewritten_sents, non_app_edits



def rewrite_topk(subwords, edits):
    assert len(subwords) == len(edits)
    rewritten_sents = []

    for i, (sent_subwords, sent_edits) in enumerate(zip(subwords, edits)):
        assert len(sent_subwords) == len(sent_edits)
        rewritten_sent = []

        for subword, topk_edits in zip(sent_subwords, sent_edits):
            rewritten_subword = []

            for edit in topk_edits:
                edit = SubwordEdit(subword, edit)

                if edit.is_applicable(subword):
                    res = edit.apply(subword)
                else:
                    res = 'NA'

                rewritten_subword.append(res)

            rewritten_sent.append(rewritten_subword)
        assert len(rewritten_sent) == len(sent_edits)

        rewritten_sents.append(rewritten_sent)

    return rewritten_sents


def detokenize_sent(sent):
    detokenize_sent = []
    for subword in sent:
        if subword.startswith('##'):
            detokenize_sent[-1] = detokenize_sent[-1] + subword.replace('##', '')
        else:
            detokenize_sent.append(subword)

    return ' '.join(detokenize_sent)


def resolve_merges(sent, edits):
    _sent = []
    assert len(sent) == len(edits)
    for subword, edit in zip(sent, edits):
        if edit.startswith('M'):
            if len(_sent) > 0:
                _sent[-1] = _sent[-1] + subword
            else:
                _sent.append(subword)
        else:
            _sent.append(subword)
    return _sent


def compute_class_weights(training_data, weight_threshold, labels_map):
    from collections import Counter
    labels = [edit for ex in training_data for edit in ex['edits']]
    labels_freqs = Counter(labels)

    # Calculate total number of samples
    total_samples = sum(labels_freqs.values())

    # Initialize class weights
    class_weights = {}

    for cls, freq in labels_freqs.items():
        if freq < weight_threshold:
            # Assign weight inversely proportional to frequency
            class_weights[cls] = total_samples / freq
        else:
            # Assign default weight of 1.0 for well-represented classes
            class_weights[cls] = 1.0
    # log scaling
    log_weights = {cls: 1 + np.log(weight) for cls, weight in class_weights.items()}

    log_weights_tensor = torch.zeros(len(labels_map))
    for cls, weight in log_weights.items():
        log_weights_tensor[labels_map[cls]] = weight

    return log_weights_tensor



if __name__ == "__main__":
    main()
