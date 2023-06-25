from omegaconf import DictConfig

from src.dataset import load_dataset_from_file


def default_data(cfg: DictConfig):
    return [load_dataset_from_file(cfg.dataset.path)] * cfg.task.ensemble_size
