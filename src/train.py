import os
from pathlib import Path

from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .dataset import create_dataset, downsample, upsample
from .helpers import get_class_labels
from .metrics import compute_metrics
from .preprocessing import tokenize
from .utils import configure_wandb


def initialize_trainer(cfg: DictConfig, use_test: bool = False):
    dataset = create_dataset(cfg.dataset.path)
    labels = get_class_labels(dataset)
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    tokenizer = tokenizer = AutoTokenizer.from_pretrained(Path(cfg.task.model_path))

    model = AutoModelForSequenceClassification.from_pretrained(
        Path(cfg.task.model_path),
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    encoded_dataset = tokenize(dataset, labels, tokenizer)

    args = TrainingArguments(
        os.getcwd(),
        evaluation_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        save_total_limit=cfg.save_total_limit,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val" if not use_test else "test"],
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, cfg.threshold),
    )

    return trainer


def train(cfg: DictConfig, dataset=None, train_folder=None):
    configure_wandb(cfg)

    if dataset == None:
        dataset = create_dataset(cfg.dataset.path)

    if cfg.upsample:
        dataset["train"] = upsample(dataset["train"])
    if cfg.downsample:
        dataset["train"] = downsample(dataset["train"])

    labels = [
        label
        for label in dataset["train"].features.keys()
        if label not in ["train_id", "text"]
    ]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.pretrained_model)

    encoded_dataset = tokenize(dataset, labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.dataset.pretrained_model,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        os.getcwd() if not train_folder else train_folder,
        evaluation_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        save_total_limit=cfg.save_total_limit,
        report_to="wandb",
        fp16=cfg.fp16,
        label_smoothing_factor=cfg.label_smoothing_factor,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, cfg.threshold),
    )

    trainer.train()
    trainer.evaluate(encoded_dataset["test"])
