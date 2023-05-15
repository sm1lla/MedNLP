import hydra
from omegaconf import DictConfig

from .dataset import examine_dataset
from .inference import infer
from .train import train


@hydra.main(config_path="config", config_name="config")
def cli(cfg: DictConfig):
    options = {"train": train, "infer": infer, "dataset": examine_dataset}