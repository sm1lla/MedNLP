from pathlib import Path

import os
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .dataset import create_dataset
from .train import tokenize, compute_metrics
from .inference import get_class_labels
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from .utils import configure_wandb

def evaluate_model(cfg: DictConfig):
    
    configure_wandb(cfg)
    #do it with trainer 
    dataset = create_dataset()
    labels = get_class_labels(use_cached=False)
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
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()

    print(results)
