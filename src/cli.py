import hydra
from omegaconf import DictConfig

from .dataset import examine_dataset
from .evaluate import evaluate_model
from .inference import infer
from .predict import predict
from .train import train
from .utils import configure_wandb


@hydra.main(config_path="config", config_name="config")
def cli(cfg: DictConfig):
    options = {
        "train": train,
        "infer": infer,
        "predict": predict,
        "dataset": examine_dataset,
        "evaluate": evaluate_model,
    }

    options[cfg.task.name](cfg)
