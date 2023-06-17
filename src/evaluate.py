from omegaconf import DictConfig

from .utils import configure_wandb
from .train import initialize_trainer


def evaluate_model(cfg: DictConfig,use_test:bool=False):
    if use_test:
        configure_wandb(cfg)
    trainer = initialize_trainer(cfg,use_test)

    results = trainer.evaluate()

    return results
