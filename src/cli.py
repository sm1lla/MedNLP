import hydra
from omegaconf import DictConfig

from .dataset import examine_dataset
from .evaluate import evaluate_model
from .inference import infer
from .train import train
from .utils import configure_wandb


@hydra.main(config_path="config", config_name="config")
def cli(cfg: DictConfig):
    options = {
        "train": train,
        "infer": infer,
        "dataset": examine_dataset,
        "evaluate": evaluate_model,
    }

    configure_wandb(cfg)

    options[cfg.task.name](cfg)
