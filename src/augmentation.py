import random

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from sklearn.utils import shuffle
from transformers import pipeline

from src.helpers import drug_examples

from .dataset import load_dataset_from_file


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

    dataset = shuffle(dataset, random_state=42)
    return Dataset.from_pandas(dataset)


def add_samples_for_class(index: int, path: str, language: str, columns: list[str]):
    dataset_generated = pd.read_csv(f"{path}_{language}_{index}.csv")
    dataset_generated = setup_dataframe(dataset_generated, columns, column_index=index)

    dataset_generated = replace_placeholder(dataset_generated)
    return dataset_generated


def translate(cfg: DictConfig):
    dataset = load_dataset_from_file(cfg.dataset.path)["train"]
    translator = pipeline(cfg.task.mode, model="t5-base", device=0)
    results = translator([cfg.task.prefix + text for text in dataset["text"]])

    translations = [translation["translation_text"] for translation in results]

    dataset_pd = pd.DataFrame(dataset)
    dataset_pd["text"] = translations

    dataset_pd.to_csv(cfg.task.save_path)


def add_translated(dataset: DatasetDict, path: str):
    dataset = dataset.to_pandas()
    stemmed = path.split(".")[0]
    path_translated_senteces = f"{stemmed}_translated.csv"
    translated = pd.read_csv(path_translated_senteces)
    translated.drop("Unnamed: 0", axis=1, inplace=True)
    num_rows = len(translated)
    translated["train_id"] = [float(num) for num in range(10000, 10000 + num_rows)]
    translated.rename(
        columns=dict(zip(translated.columns, dataset.columns)), inplace=True
    )
    dataset = pd.concat([dataset, translated], ignore_index=True)
    dataset = shuffle(dataset, random_state=42)
    return Dataset.from_pandas(dataset)
