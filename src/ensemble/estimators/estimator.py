import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from datasets import DatasetDict
import wandb
from src.evaluate import evaluate_model_with_data
from src.train import train, init_trainer_with_dataset
from src.utils import (
    add_section_to_metric_log,
    configure_wandb_without_cfg,
    finish_wandb,
    get_best_checkpoint_path,
)


class estimator:
    def __init__(self, name, cfg: DictConfig, dataset:DatasetDict):
        self.name = name
        self.cfg = DictConfig(cfg)
        self.cfg.run_name = self.name
        self.dataset = dataset
        self.folder = Path(cfg.task.ensemble_path) / self.name
        if cfg.task.do_train is not True:
            self.cfg.task.model_path = get_best_checkpoint_path(self.folder)
        else:
            self.cfg.task.model_path = self.folder
        self.use_wandb = cfg.use_wandb

    def load_model(self):
        raise NotImplementedError()

    def clear_session(self):
        torch.cuda.empty_cache()

    def predict(self, texts: list):
        raise NotImplementedError()


    def train(self):
        train(self.cfg, self.dataset, self.folder)
        self.cfg.task.model_path = get_best_checkpoint_path(self.folder)

    def get_predictions_and_labels_on_datast(self, on_test_data: bool = False):
        trainer = init_trainer_with_dataset(cfg=self.cfg,dataset=self.dataset, use_test=on_test_data)
        predictions, labels, metrics = trainer.predict(trainer.eval_dataset)

        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= self.cfg.threshold)] = 1

        return predictions, probs, labels

    def validate(self, use_test: bool = False):
        if self.use_wandb and use_test:
            configure_wandb_without_cfg(
                self.cfg.project_name, self.name, self.cfg.group_name
            )
            self.cfg.use_wandb = False  # ugly workaround
            wandb.log(
                add_section_to_metric_log(
                    "test", evaluate_model_with_data(cfg=self.cfg,dataset=self.dataset, use_test=use_test), "eval_"
                )
            )
            self.cfg.use_wandb = True
        else:
            return evaluate_model_with_data(cfg=self.cfg,dataset=self.dataset, use_test=use_test)    
