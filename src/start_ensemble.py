from omegaconf import DictConfig
import os

from src.ensemble.data_techniques.kfoldcross import kfoldcross
from src.ensemble.data_techniques.default import default_data
from src.ensemble.data_techniques.bagging import bagging
from src.ensemble.classifiers.majority_vote_classifier import MajorityVoteClassifier
from src.ensemble.classifiers.avg_prob_classifier import AvgProbClassifier
from src.ensemble.classifiers.max_prob_classifier import MaxProbClassifier
from src.ensemble.classifiers.median_prob_classifier import MedianProbClassifier
from src.ensemble.classifiers.weighted_label_classifier import WeightVoteClassifier
from src.ensemble.estimators.estimator import estimator



def start_ensemble(cfg: DictConfig):

    cfg.run_name = "" if cfg.run_name =="None" else cfg.run_name
    modelInfo = str(cfg.task.ensemble_size) + cfg.task.ensemble_technique + cfg.run_name

    dataset_technique = {"kfoldCross": kfoldcross,
                         "default":default_data,
                         "bagging":bagging}

    datasets = dataset_technique[cfg.task.ensemble_technique](cfg)


    if cfg.task.ensemble_path == "None":
        cfg.task.ensemble_path = os.getcwd()

    cfg.group_name = modelInfo

    learners = []

    for num in range(1, cfg.task.ensemble_size + 1):
        model = estimator(name=modelInfo + str(num), cfg=cfg)
        learners.append(model)

    voter_name =  str(cfg.task.ensemble_size) + cfg.task.ensemble_technique

    voter = MajorityVoteClassifier(
        learners, cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb
    )

    voter2 = AvgProbClassifier(learners,cfg.threshold,cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb)
    voter3 = MaxProbClassifier(learners,cfg.threshold,cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb)
    voter4 = MedianProbClassifier(learners,cfg.threshold,cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb)
    voter5 = WeightVoteClassifier(learners,cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb)
    if cfg.task.do_train:
        voter.train_on_selected_data(datasets)

    voter.validate_estimators(on_test_data=True)
    voter2.validate(on_test_data=True)
    voter3.validate(on_test_data=True)
    voter4.validate(on_test_data=True)
    voter5.validate(on_test_data=True)
