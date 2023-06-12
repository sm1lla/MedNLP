import random

import numpy as np
import pandas as pd
from datasets import Dataset
from hydra.utils import get_original_cwd
from sklearn.utils import shuffle

from src.helpers import drug_examples


def setup_dataframe(new_dataset: pd.DataFrame, columns: list[str], column_index: int):
    rows_generated = len(new_dataset)
    for column in columns:
        if column == columns[column_index]:
            new_dataset[column] = np.ones(rows_generated, dtype=int)
        elif column == "train_id":
            new_dataset.insert(0, "train_id", np.zeros(rows_generated, dtype=float))
        elif column != "text":
            new_dataset[column] = np.zeros(rows_generated, dtype=int)
    return new_dataset


def replace_placeholder(dataset_generated: pd.DataFrame):
    dataset_generated["text"] = [
        text.replace("[Medikamentenname]", random.sample(drug_examples(), 1)[0])
        for text in dataset_generated["text"]
    ]
    return dataset_generated


def add_generated_samples(dataset, column_indices: list[int], path: str, language: str):
    dataset = dataset.to_pandas()
    for column_index in column_indices:
        dataset_generated = add_samples_for_class(
            column_index, path, language, list(dataset.columns)
        )
        dataset = pd.concat([dataset, dataset_generated], ignore_index=True)

    shuffle(dataset, random_state=42)
    return Dataset.from_pandas(dataset)


def add_samples_for_class(index: int, path: str, language: str, columns: list[str]):
    dataset_generated = pd.read_csv(
        f"{get_original_cwd()}/{path}/generated_tweets_{language}_{index}.csv"
    )
    dataset_generated = setup_dataframe(dataset_generated, columns, column_index=index)

    dataset_generated = replace_placeholder(dataset_generated)
    return dataset_generated
