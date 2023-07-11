import hydra
from omegaconf import DictConfig

from .augmentation import translate
from .dataset import examine_dataset, print_examples_for_classnames
from .debug import debug
from .evaluate import evaluate_model
from .evaluateWandb import evaluate_wandb_run
from .inference import infer
from .output import save_predictions
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
        "evaluateWandb": evaluate_wandb_run,
        "debug": debug,
        "ensemble": start_ensemble,
        "examples": print_examples_for_classnames,
        "generate": generate_for_all_classes,
        "translate": translate,
        "output": save_predictions,
    }

    options[cfg.task.name](cfg)
