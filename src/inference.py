from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .dataset import create_dataset


def get_class_labels(use_cached: bool = True):
    if use_cached:
        return [
            "C0027497:Übelkeit",
            "C0011991:Diarrhöe",
            "C0015672:Erschöpfung",
            "C0042963:Erbrechen",
            "C0003123:Anorexie",
            "C0018681:Kopfschmerzen",
            "C0015967:Fieber",
            "C0206062:Interstitielle Lungenerkrankung",
            "C0023895:Leberschädigung",
            "C0012833:Drehschwindel",
            "C0030193:Schmerz",
            "C0002170:Alopezie",
            "C0004096:Analgetisches Asthma-Syndrom",
            "C0022658:Nierenerkrankung",
            "C0020517:Hypersensibilität",
            "C0917801:Insomnie",
            "C0009806:Constipation",
            "C0005956:Knochenmarkerkrankung",
            "C0000737:Bauchschmerzen",
            "C0010692:Hämorrhagische Zystitis",
            "C0015230:Ausschlag",
            "C0149745:Stomatitis",
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


def infer(text: str, checkpoint_path: str):
    labels = get_class_labels()
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    tokenizer = tokenizer = AutoTokenizer.from_pretrained(Path(checkpoint_path))

    model = AutoModelForSequenceClassification.from_pretrained(
        Path(checkpoint_path),
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    outputs = model(**encoding)

    predicted_labels = get_predicted_classes(outputs, id2label, 0.5)
    print(predicted_labels)
