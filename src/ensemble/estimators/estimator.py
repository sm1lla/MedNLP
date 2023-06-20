import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

import wandb
from src.evaluate import evaluate_model
from src.train import initialize_trainer, train
from src.utils import (
    add_section_to_metric_log,
    configure_wandb_without_cfg,
    finish_wandb,
    get_best_checkpoint_path,
)


class estimator:
    def __init__(self, name, cfg: DictConfig):
        self.name = name
        self.cfg = DictConfig(cfg)
        self.cfg.run_name = self.name
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
        train(self.cfg, self.folder)
        self.cfg.task.model_path = get_best_checkpoint_path(self.folder)

    def train_on_selected_data(self, dataset):
        train(self.cfg, dataset, self.folder)
        self.cfg.task.model_path = get_best_checkpoint_path(self.folder)

    def get_predictions_and_labels_on_datast(self, on_test_data: bool = False):
        trainer = initialize_trainer(self.cfg, on_test_data)
        predictions, labels, metrics = trainer.predict(trainer.eval_dataset)

        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= self.cfg.threshold)] = 1

        return predictions, probs, labels

    def validate(self, use_test: bool = False):
        if not use_test:
            return evaluate_model(self.cfg, use_test)
        else:
            # log to wandb
            if self.use_wandb:
                configure_wandb_without_cfg(
                    self.cfg.project_name, self.name, self.cfg.group_name
                )
                self.cfg.use_wandb = False  # ugly workaround
                wandb.log(
                    add_section_to_metric_log(
                        "test", evaluate_model(self.cfg, use_test), "eval_"
                    )
                )

                self.cfg.use_wandb = True

    def get_prediction_scores(self, image_paths: list[str]):
        raise NotImplementedError()

    def predict_from_prediction_scores(self, prediction_scores):
        raise NotImplementedError()
