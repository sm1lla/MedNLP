from omegaconf import DictConfig

from .train import initialize_trainer


def evaluate_model(cfg: DictConfig):
    trainer = initialize_trainer(cfg)

    results = trainer.evaluate()

    print(results)
