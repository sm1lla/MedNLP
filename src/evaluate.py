from omegaconf import DictConfig

from .utils import configure_wandb
from .train import initialize_trainer,init_trainer_with_dataset
from datasets import DatasetDict

def evaluate_model(cfg: DictConfig,use_test:bool=False):
    if use_test:
        configure_wandb(cfg)
    trainer = initialize_trainer(cfg,use_test)

    results = trainer.evaluate()

    return results

def evaluate_model_with_data(cfg: DictConfig,dataset:DatasetDict,use_test:bool=False):
    if use_test:
        configure_wandb(cfg)
    trainer = init_trainer_with_dataset(cfg=cfg,dataset=dataset,use_test=use_test)

    results = trainer.evaluate()

    return results
