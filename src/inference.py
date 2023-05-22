from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .dataset import create_dataset


def get_class_labels(use_cached: bool = True):
    if use_cached:
        return [
            "nausea",
            "diarrhea",
            "fatigue",
            "vomiting",
            "loss of appetite",
            "headache",
            "fever",
            "interstitial lung disease",
            "liver damage",
            "dizziness",
            "pain",
            "alopecia",
            "analgesic asthma syndrome",
            "renal impairment",
            "hypersensitivity",
            "insomnia",
            "constipation",
            "bone marrow dysfunction",
            "abdominal pain",
            "hemorrhagic cystitis",
            "rash",
            "stomatitis",
            "other",
        ]
    else:
        dataset = create_dataset(test_size=0.2)

        labels = [
            label
            for label in dataset["train"].features.keys()
            if label not in ["train_id", "text"]
        ]
        return labels


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

    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        Path(cfg.task.checkpoint_path)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        Path(cfg.task.checkpoint_path),
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    encoding = tokenizer(cfg.task.text, return_tensors="pt")
    print(encoding.items())
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    outputs = model(**encoding)

    predicted_labels = get_predicted_classes(outputs, id2label, cfg.threshold)
    print(predicted_labels)
