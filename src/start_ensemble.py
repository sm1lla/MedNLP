from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from pathlib import Path
import os

from src.ensemble.data_techniques.kfoldcross import kfoldcross
from src.ensemble.data_techniques.default import default_data
from src.ensemble.data_techniques.bagging import bagging
from src.ensemble.data_techniques.shuffle import shuffle
from src.ensemble.data_techniques.partitioning import partitioning
from src.ensemble.classifiers.majority_vote_classifier import MajorityVoteClassifier
from src.ensemble.classifiers.avg_prob_classifier import AvgProbClassifier
from src.ensemble.classifiers.max_prob_classifier import MaxProbClassifier
from src.ensemble.classifiers.median_prob_classifier import MedianProbClassifier
from src.ensemble.classifiers.weighted_label_classifier import WeightVoteClassifier
from src.ensemble.estimators.estimator import estimator
from src.dataset import load_dataset_from_file




def get_default_estimators(cfg: DictConfig, modelInfo:str,datasets:list):
    learners = []

    for num,dataset in zip(range(1, cfg.task.ensemble_size + 1),datasets):
        model = estimator(name=modelInfo + str(num), cfg=cfg,dataset=dataset)
        learners.append(model)
    
    return learners

def get_multilingual_estimators(cfg: DictConfig, modelInfo:str,datasets:list):
    #ignore datasets

    learners = []
    config_directory = Path(get_original_cwd()) / "src" / "config"/"dataset"

    language = {
        0: "de",
        1:"en",
        2:"ja",
        3:"fr",
    }
    ensemble_size = cfg.task.ensemble_size - (cfg.task.ensemble_size%4) # we want for every language at least one 
    if ensemble_size<4 and (ensemble_size%4)!= 0:
        raise Exception("Multilingual Ensemble does not fulfill requirements (ensemble size should be at least languages size and dividable by it)")
    cfg.task.ensemble_size = ensemble_size

    for num in range(ensemble_size):
        dataset_config_name = language[num%4] + ".yaml"
        dataset_config_path = config_directory/dataset_config_name
        cfg.dataset = OmegaConf.load(dataset_config_path )
        
        dataset = load_dataset_from_file(cfg.dataset.path)
        
        model = estimator(name=modelInfo + language[num%4] + str(num+1), cfg=cfg, dataset=dataset)
        learners.append(model)
    
    return learners

dataset_technique = {"kfold": kfoldcross,
                     "default":default_data,
                     "bagging":bagging,
                     "shuffle":shuffle,
                     "partition":partitioning}

estimator_technique  = {
    "default":get_default_estimators,
    "multilingual": get_multilingual_estimators
}

def start_ensemble(cfg: DictConfig):

    cfg.run_name = "" if cfg.run_name =="None" else cfg.run_name
    modelInfo = str(cfg.task.ensemble_size) + cfg.task.ensemble_technique + cfg.task.ensemble_technique + cfg.run_name

    datasets = dataset_technique[cfg.task.ensemble_technique](cfg)


    if cfg.task.ensemble_path == "None":
        cfg.task.ensemble_path = os.getcwd()

    cfg.group_name = modelInfo

    estimators = estimator_technique[cfg.task.estimator_technique](cfg,modelInfo,datasets)

    voter_name =  str(cfg.task.ensemble_size) + cfg.task.ensemble_technique

    voter = MajorityVoteClassifier(
        estimators, cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb
    )

    voter2 = AvgProbClassifier(estimators,cfg.threshold,cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb)
    voter3 = MaxProbClassifier(estimators,cfg.threshold,cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb)
    voter4 = MedianProbClassifier(estimators,cfg.threshold,cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb)
    voter5 = WeightVoteClassifier(estimators,cfg.project_name, voter_name, cfg.group_name, cfg.use_wandb)

    if cfg.task.do_train:
        voter.train()
    else:
       voter.validate_estimators(on_test_data=True)
    voter.validate(on_test_data=True)
    voter2.validate(on_test_data=True)
    voter3.validate(on_test_data=True)
    voter4.validate(on_test_data=True)
    voter5.validate(on_test_data=True)
