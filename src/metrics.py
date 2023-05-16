import numpy as np


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
