from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score
    )
import argparse
import pandas as pd


def eval(gold_labels, preds_labels, labels):
    """Main evaluation function

    Args:
        gold_labels (list of list of str): gold labels
        preds_labels (list of list of str): prediction labels
        labels (list): list of output labels

    Returns:
        evaluation metrics as a dict.
    """

    flatten_gold = [label for sublist in gold_labels for label in sublist]
    flatten_pred = [label for sublist in preds_labels for label in sublist]

    metrics = compute_metrics(flatten_gold, flatten_pred, labels)
    return metrics


def compute_metrics(gold_list, preds_list, labels):
    """Computes the evaluation metrics"""
    gold_detection = [('E' if label != 'K' else label) for label in gold_list]
    pred_detection = [('E' if label != 'K' else label) for label in preds_list]


    detection_accuracy = accuracy_score(y_true=gold_detection, y_pred=pred_detection)
    edits_accuracy = accuracy_score(y_true=gold_list, y_pred=preds_list)

    return {'detect_acc': detection_accuracy,
            'edit_acc': edits_accuracy}


def read_data(path):
    """Reads gold and predicted labels

    Args:
        path (str): path of a tsv file containing gold and predicted labels

    Returns:
        tuple of gold and predicted labels respectively.
    """
    pred_tags = []
    gold_tags = []
    all_gold_tags = []
    all_pred_tags = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) > 1:
                gold, pred = line[1], line[2]
                gold_tags.append(gold)
                pred_tags.append(pred)
            else:
                all_gold_tags.append(gold_tags)
                all_pred_tags.append(pred_tags)
                gold_tags = []
                pred_tags = []

        if pred_tags and gold_tags:
            all_gold_tags.append(gold_tags)
            all_pred_tags.append(pred_tags)

    return all_gold_tags, all_pred_tags


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--labels')
    parser.add_argument('--output')

    args = parser.parse_args()
    labels = get_labels(args.labels)
    gold_tags, pred_tags = read_data(args.data)

    metrics = eval(gold_tags, pred_tags, labels)

    with open(args.output+'.txt', "w") as f:
        for metric in metrics:
            f.write(f'{metric} : {metrics[metric]}\n')
