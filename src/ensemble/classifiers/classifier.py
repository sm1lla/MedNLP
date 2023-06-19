import time

import numpy as np
import torch

import wandb
from src.metrics import multi_label_metrics
from src.utils import (
    add_section_to_metric_log,
    configure_wandb_without_cfg,
    finish_wandb,
)


class Classifier:
    def __init__(
        self,
        estimators: list,
        project_name: str,
        name: str,
        group_name: str,
        use_wandb: bool,
    ):
        self.estimators = estimators
        self.estimator_val_metrics = [0] * len(estimators)
        self.project_name = project_name
        self.name = name
        self.group_name = group_name
        self.use_wandb = use_wandb

    def predict(self, x: list):
        predictions = np.array()

        for estimator in self.estimators:
            try:
                predictions.append(estimator.predict(x))
            except Exception as e:
                print("Error: model unable to predict")
                print(e)
        predictions = self.preprocess_predictions(predictions)

        results = self.classify(predictions)

        return results

    def predict_on_dataset(self, on_test_data: bool = False):
        st = time.time()
        predictions = []
        probs = []
        labels = None
        for estimator in self.estimators:
            (
                est_predictions,
                est_probs,
                est_labels,
            ) = estimator.get_predictions_and_labels_on_datast(on_test_data)
            predictions.append(est_predictions)
            probs.append(est_probs)
            if labels is not None:
                assert np.array_equal(labels, est_labels)
            labels = est_labels

        predictions, probs = self.preprocess(predictions,probs)

        results = self.classify(predictions, probs)

        et = time.time()
        elapsed_time = et - st
        print("prediction time:", elapsed_time, "seconds")

        return results, labels

    def train(self):
        st = time.time()
        for estimator in self.estimators:
            single_st = time.time()
            estimator.train()
            single_et = time.time()
            single_elapsed_time = single_et - single_st
            print("\n\n")
            print(f"Training completed for model = {estimator.name}")
            print("training time:", single_elapsed_time, "seconds")
            print("\n\n")
        et = time.time()
        elapsed_time = et - st
        print("Total training time:", elapsed_time, "seconds")

    def train_on_selected_data(self, datasets: list):
        st = time.time()

        for estimator, dataset in zip(self.estimators, datasets):
            single_st = time.time()
            estimator.train_on_selected_data(dataset)
            single_et = time.time()
            single_elapsed_time = single_et - single_st
            print("\n\n")
            print(f"Training completed for model = {estimator.name}")
            print("training time:", single_elapsed_time, "seconds")
            print("\n\n")

        et = time.time()
        elapsed_time = et - st
        print("Total training time:", elapsed_time, "seconds")

    def validate(self, on_test_data: bool = False):
        # get predictions
        results, labels = self.predict_on_dataset(on_test_data)

        # calculate metrics
        metrics = multi_label_metrics(results, labels)

        if self.use_wandb and on_test_data:
            configure_wandb_without_cfg(self.project_name, self.name, self.group_name)
            wandb.log(add_section_to_metric_log("test", metrics))

        return metrics

    def validate_estimators(self, on_test_data: bool = False):
        for idx, estimator in enumerate(self.estimators):
            if not on_test_data:
                self.estimator_val_metrics[idx] = estimator.validate(on_test_data)
            else:
                estimator.validate(on_test_data)
        if not on_test_data:
            print(self.estimator_val_metrics)

    def classify(self, predictions: list, probs: list):
        raise NotImplementedError("Implement in subclass")

    def preprocess(self, predictions: list, probs: list):
        return np.asarray(predictions), torch.stack(probs).numpy()
