import hydra
from omegaconf import DictConfig

from .dataset import examine_dataset, print_examples_for_classnames
from .debug import debug
from .evaluate import evaluate_model
from .inference import infer
from .predict import predict
from .start_ensemble import start_ensemble
from .train import train
from .tweet_generation import generate_for_all_classes


@hydra.main(config_path="config", config_name="config")
def cli(cfg: DictConfig):
    options = {
        "train": train,
        "infer": infer,
        "predict": predict,
        "dataset": examine_dataset,
        "evaluate": evaluate_model,
        "debug": debug,
        "ensemble": start_ensemble,
        "examples": print_examples_for_classnames,
        "generate": generate_for_all_classes,
    }

    options[cfg.task.name](cfg)
