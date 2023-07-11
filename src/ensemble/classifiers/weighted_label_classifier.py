import numpy as np

from src.ensemble.classifiers.classifier import Classifier


class WeightVoteClassifier(Classifier):
    def __init__(
        self,
        estimators: list,
        project_name: str,
        name: str,
        group_name: str,
        use_wandb: bool,
    ):
        super().__init__(
            estimators, project_name, "WeightVote" + name, group_name, use_wandb
        )
        self.weights = None

    def load_weight_list(self):
        self.validate_estimators()
        f1_scores = [
            tupel["eval_micro avg.f1-score"] for tupel in self.estimator_val_metrics
        ]

        indices_highest_elements = np.argsort(f1_scores, axis=0)[::-1]

        weights = np.arange(5)[::-1] + 1
        self.weights = weights[indices_highest_elements]

        print(self.weights)

    def classify(self, predictions: np.array, probs: np.array):
        if self.weights == None:
            self.load_weight_list()

        for idx in range(len(self.estimators)):
            predictions[idx] = predictions[idx] * self.weights[idx]

        counted_predictions = np.sum(predictions, axis=0) / np.sum(self.weights, axis=0)

        result = np.where(counted_predictions >= 0.5, 1, 0)
        return result
