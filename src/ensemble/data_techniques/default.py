from omegaconf import DictConfig
from src.dataset import create_dataset

def default_data(cfg: DictConfig):

    return [create_dataset(cfg.dataset.path)]*cfg.task.ensemble_size
