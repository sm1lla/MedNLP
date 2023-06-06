from omegaconf import DictConfig

from .utils import configure_wandb
from .train import initialize_trainer


def evaluate_model(cfg: DictConfig):
    configure_wandb(cfg)
    trainer = initialize_trainer(cfg)

    results = trainer.evaluate()

    print(results)
