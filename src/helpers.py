from .dataset import create_dataset


def get_class_labels(dataset=None):
    if not dataset:
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
        labels = [
            label
            for label in dataset["train"].features.keys()
            if label not in ["train_id", "text"]
        ]
        return labels
