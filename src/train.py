import os
from datetime import datetime

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from .dataset import create_dataset
from .inference import get_class_labels
from .metrics import all_symptoms_metric, each_symptoms_metric, two_way_metric


def tokenize(dataset, labels, tokenizer):
    encoded_dataset = dataset.map(
        lambda x: preprocess_data(x, labels, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    encoded_dataset.set_format("torch")
    return encoded_dataset


def preprocess_data(examples, labels, tokenizer):
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
# TODO : get access to cfg to use threshold value from there
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
        "f1": f1_macro_average,
        "accuracy": accuracy,
        "all_symptoms_metric": all_symptoms_metric(y_pred, y_true),
        "two_way_metric": two_way_metric(y_pred, y_true),
    }
    sklearn_metrics = classification_report(
        y_pred, y_true, target_names=get_class_labels(), output_dict=True
    )

    metrics.update(sklearn_metrics)

    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def train(cfg: DictConfig):
    dataset = create_dataset(test_size=0.2)

    labels = [
        label
        for label in dataset["train"].features.keys()
        if label not in ["train_id", "text"]
    ]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

    encoded_dataset = tokenize(dataset, labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "deepset/gbert-base",
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

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
        report_to="wandb",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
