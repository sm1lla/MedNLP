import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import EvalPrediction

from .helpers import get_class_labels


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(y_pred, y_true):
    # compute metrics
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
        "f1": f1_macro_average,
        "accuracy": accuracy,
        "all_symptoms_metric": all_symptoms_metric(y_pred, y_true),
        "two_way_metric": two_way_metric(y_pred, y_true),
    }
    sklearn_metrics = classification_report(
        y_pred=y_pred, y_true=y_true, target_names=get_class_labels(), output_dict=True
    )

    metrics.update(sklearn_metrics)

    return metrics


def compute_metrics(p: EvalPrediction, threshold: float = 0.5):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    result = multi_label_metrics(y_pred=y_pred, y_true=p.label_ids)
    return result


def all_symptoms_metric(y_pred, y_true):
    # Check whether all labels exactly match with the gold standard labels
    num_examples = len(y_pred)
    boolean_array = y_pred == y_true
    boolean_array = boolean_array.all(axis=1)

    num_true_values = np.count_nonzero(boolean_array)
    result = num_true_values / num_examples * 100
    return result


def each_symptoms_metric(y_pred, y_true):
    # Each symptom: Check whether a single label matches with the corresponding gold standard label.

    num_examples = len(y_pred)
    boolean_array = y_pred == y_true
    boolean_array = boolean_array.T
    num_true_values = np.count_nonzero(boolean_array, axis=1)
    result = num_true_values / num_examples * 100
    result = result.tolist()  # cant log numpy array
    return result


def two_way_metric(y_pred, y_true):
    # Check whether the input contains at least one ADE or not

    num_examples = len(y_pred)
    boolean_array = (y_pred.any(axis=1)) == (y_true.any(axis=1))

    num_true_values = np.count_nonzero(boolean_array)
    result = num_true_values / num_examples * 100
    return result
