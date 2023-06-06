from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .helpers import get_class_labels


def get_predicted_classes(outputs, id2label, threshold):
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [
        id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
    ]
    return predicted_labels


def infer(cfg: DictConfig):
    labels = get_class_labels()
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

    encoding = tokenizer(cfg.task.text, return_tensors="pt")

    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    outputs = model(**encoding)

    predicted_labels = get_predicted_classes(outputs, id2label, cfg.threshold)
    print(f"Text: {cfg.task.text}")
    print(f"Prediction: {predicted_labels}")
