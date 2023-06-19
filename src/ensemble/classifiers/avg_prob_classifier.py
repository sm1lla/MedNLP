import numpy as np

from src.ensemble.classifiers.classifier import Classifier


class AvgProbClassifier(Classifier):
    def __init__(
        self,
        estimators: list,
        threshold: float,
        project_name: str,
        name: str,
        group_name: str,
        use_wandb: bool,
    ):
        super().__init__(
            estimators, project_name, "AvgProbVote" + name, group_name, use_wandb
        )
        self.threshold = threshold

    def classify(self, predictions: np.array, probs: np.array):
        averaged_probs = np.sum(probs, axis=0) / len(self.estimators)
        result = np.where(averaged_probs >= self.threshold, 1, 0)

        # result = max(ensemble_predictions,key=itemgetter(1))[0]
        return result
