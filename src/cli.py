import hydra
from omegaconf import DictConfig

from .dataset import examine_dataset, print_examples_for_classnames
from .evaluate import evaluate_model
from .inference import infer
from .predict import predict
from .train import train


@hydra.main(config_path="config", config_name="config")
def cli(cfg: DictConfig):
    options = {
        "train": train,
        "infer": infer,
        "predict": predict,
        "dataset": examine_dataset,
        "evaluate": evaluate_model,
        "examples": print_examples_for_classnames
    }

    options[cfg.task.name](cfg)
