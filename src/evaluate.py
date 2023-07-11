from datasets import DatasetDict
from omegaconf import DictConfig

import wandb

from .train import init_trainer_with_dataset, initialize_trainer
from .utils import add_section_to_metric_log, configure_wandb


def evaluate_model(cfg: DictConfig, use_test: bool = True):
    if use_test:
        configure_wandb(cfg)
    trainer = initialize_trainer(cfg, use_test)
    test_evaluation = trainer.evaluate(trainer.eval_dataset)
    if use_test:
        wandb.log(add_section_to_metric_log("test", test_evaluation, "eval_"))

    return test_evaluation


def evaluate_model_with_data(
    cfg: DictConfig, dataset: DatasetDict, use_test: bool = False
):
    if use_test:
        configure_wandb(cfg)
    trainer = init_trainer_with_dataset(cfg=cfg, dataset=dataset, use_test=use_test)

    results = trainer.evaluate()

    return results
