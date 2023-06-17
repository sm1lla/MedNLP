from omegaconf import DictConfig

from .ensemble.ensemble_techniques.kfoldcross import kfoldcross


def start_ensemble(cfg: DictConfig):
    technique = {"kfold": kfoldcross}

    technique[cfg.task.ensemble_technique](cfg)
