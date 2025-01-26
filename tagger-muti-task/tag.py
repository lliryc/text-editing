import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils import Split, get_labels, process, TokenClassificationDataset, DataCollatorForTokenClassification, read_examples_from_file
from model import BertForTokenClassificationSingleLabel
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from edits.edit import SubwordEdit
from m2scorer import m2scorer
from util.postprocess import remove_pnx, pnx_tokenize


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are
    going to fine-tune from.
    """

    model_name_or_path: str = field(
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

    # If you want to tweak more attributes on your tokenizer, you should do it
    # in a distinct script, or just modify its tokenizer_config.json.

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

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files "
                          "for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels."},
    )
    binary_labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all binary labels."},
    )
    pred_mode: Optional[str] = field(
        default=None, metadata={"help": "Prediction mode to get the actual "
                                        "token predictions on dev or test."}
    )
    label_pred_output_file: Optional[str] = field(
        default=None, metadata={"help": "Label predictions output file."}
    )
    rewrite_pred_output_file: Optional[str] = field(
        default=None, metadata={"help": "Rewrite predictions output file."}
    )
    m2_edits: Optional[str] = field(
        default=None, metadata={"help": "M2 edits file."}
    )
    m2_edits_nopnx: Optional[str] = field(
        default=None, metadata={"help": "M2 edits file without punctuation."}
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

    labels = get_labels(data_args.labels)
    binary_labels = get_labels(data_args.binary_labels)

    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    binary_label_map: Dict[int, str] = {i: label for i, label in enumerate(binary_labels)}

    num_labels = len(labels)
    num_binary_labels = len(binary_labels)

    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name
            else model_args.model_name_or_path),
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )

    config.update({
        'num_binary_labels': num_binary_labels,
        'binary_id2label': binary_label_map,
        'binary_label2id': {label: i for i, label in enumerate(binary_labels)}
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name
            else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=False,
        model_max_length=512
    )

    if training_args.do_train:
        train_dataset = read_examples_from_file(data_dir=data_args.data_dir, mode=Split.train)
        train_dataset = train_dataset.map(process,
                    fn_kwargs={"label_list": labels, "bin_label_list": binary_labels, "tokenizer": tokenizer},
                    batched=True,
                    desc="Running tokenizer on train dataset"
                    )

    model = BertForTokenClassificationSingleLabel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
                                                       padding=True)

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
        pred_mode = data_args.pred_mode

        if pred_mode == "test":
            pred_data = Split.test
        elif pred_mode == "dev":
            pred_data = Split.dev
        elif pred_mode == "train":
            pred_data = Split.train
        elif pred_mode == "tune":
            pred_data = Split.tune
        elif pred_mode == "test_L1":
            pred_data = Split.test_L1
        elif pred_mode == "test_L2":
            pred_data = Split.test_L2
        elif pred_mode == "pred":
            pred_data = Split.pred

        test_dataset =  read_examples_from_file(data_dir=data_args.data_dir, mode=pred_data)

        test_dataset = test_dataset.map(process,
                                        fn_kwargs={"label_list": labels, "bin_label_list": binary_labels, "tokenizer": tokenizer},
                                        batched=True,
                                        desc=f"Running tokenizer on {pred_mode} dataset")

        pred_edits = predict(model=model, test_dataset=test_dataset,
                             collate_fn=data_collator,
                             topk_pred=model_args.topk_pred,
                             label_map=label_map,
                             batch_size=training_args.per_device_eval_batch_size)


        label_pred_output_file = os.path.join(training_args.output_dir,
                                              data_args.label_pred_output_file)

        if trainer.is_world_process_zero():
            with open(label_pred_output_file, "w") as writer:
                for example in pred_edits:
                    for label in example:
                        writer.write(label)
                        writer.write('\n')
                    writer.write('\n')

        detok_pred_rewrites, pred_rewrites, non_app_edits = rewrite(subwords=test_dataset['subwords'], edits=pred_edits)

        # Clean generated output by separating pnx and extra white space
        detok_pred_rewrites = pnx_tokenize(detok_pred_rewrites)
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

        # running the m2 evaluation
        logger.info("*** Running M2 Evaluation ***")
        m2scorer.evaluate(rewrite_pred_output_file, data_args.m2_edits,
                          timeout=30)

        # running the m2 evaluation without pnx
        logger.info("*** Running M2 Evaluation (No Pnx) ***")
        m2scorer.evaluate(rewrite_pred_output_file+'.nopnx', data_args.m2_edits_nopnx,
                          timeout=30)



def predict(model, test_dataset, collate_fn, label_map, topk_pred=True, batch_size=32):
    logger.info(f"***** Running Prediction *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Batch size = {batch_size}")


    data_loader = DataLoader(test_dataset.remove_columns(['subwords', 'edits', 'bin']),
                             batch_size=batch_size,
                             shuffle=False, drop_last=False, collate_fn=collate_fn)

    sent_ids = None
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            inputs = {'input_ids': batch['input_ids'],
                      'token_type_ids': batch['token_type_ids'],
                      'attention_mask': batch['attention_mask']}

            label_ids = batch['labels']
            # sent_ids = (batch['sent_id'] if sent_ids is None
            #             else torch.cat((sent_ids, batch['sent_id'])))

            logits = model(**inputs)[0]

            if topk_pred:
                predictions = _align_predictions_topk(logits, label_ids, label_map, k=10)
            else:
                predictions = _align_predictions(logits.cpu().numpy(),
                                                 label_ids.cpu().numpy(),
                                                 label_map)


            preds.extend(predictions)

    # Collating the predicted labels based on the sentence ids
    # sent_ids = sent_ids.cpu().numpy()
    # final_preds_list = [[] for _ in range(len(set(sent_ids)))]
    # for i, id in enumerate(sent_ids):
    #     final_preds_list[id].extend(preds[i])

    # return final_preds_list
    if topk_pred:
        _preds = applicable_preds(test_dataset['subwords'], preds)
        return _preds

    return preds


def _align_predictions(predictions, label_ids, label_map):
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
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list


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
    topk_vals, topk_indices = torch.topk(predictions, k=k, dim=-1)

    batch_size, seq_len = topk_indices.shape[0], topk_indices.shape[1]
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append([label_map[idx.item()] for idx in topk_indices[i][j]])

    return preds_list


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
        assert len(sent_subwords) == len(sent_edits)
        rewritten_sent = []

        for subword, edit in zip(sent_subwords, sent_edits):
            edit = SubwordEdit(subword, edit)

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


if __name__ == "__main__":
    main()
