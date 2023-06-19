import math

import numpy as np

from src.ensemble.classifiers.classifier import Classifier


class MajorityVoteClassifier(Classifier):
    def __init__(
        self,
        estimators: list,
        project_name: str,
        name: str,
        group_name: str,
        use_wandb: bool,
    ):
        super().__init__(
            estimators, project_name, "MajorityVote" + name, group_name, use_wandb
        )
        self.size_of_majority = math.ceil(len(estimators) / 2)

    def classify(self, predictions: np.array, probs: np.array):
        counted_predictions = np.sum(predictions, axis=0)

        result = np.where(counted_predictions >= self.size_of_majority, 1, 0)

        return result
