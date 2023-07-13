import random

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig
from sklearn.utils import shuffle
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

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


def add_generated_samples(
    dataset, column_indices: list[int], path: str, language: str, fraction: float = 1.0
):
    dataset = dataset.to_pandas()
    for column_index in column_indices:
        dataset_generated = add_samples_for_class(
            column_index, path, language, list(dataset.columns)
        )
        dataset_generated = dataset_generated.sample(frac=fraction, random_state=42)
        dataset = pd.concat([dataset, dataset_generated], ignore_index=True)

    dataset = shuffle(dataset, random_state=42)
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.remove_columns(["__index_level_0__"])
    return dataset


def add_samples_for_class(index: int, path: str, language: str, columns: list[str]):
    dataset_generated = pd.read_csv(f"{path}_de_{index}.csv")
    dataset_generated = setup_dataframe(dataset_generated, columns, column_index=index)

    dataset_generated = replace_placeholder(dataset_generated)
    return dataset_generated


def translate(cfg: DictConfig):
    dataset = load_dataset("csv", data_files={"train": cfg.task.input_path})["train"]
    dataset_pd = pd.DataFrame(dataset)
    if not cfg.task.use_t5:
        tokenizer = AutoTokenizer.from_pretrained(cfg.task.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.task.model_name)
    else:
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        dataset_pd["text"] = [cfg.task.prefix + text for text in dataset["text"]]

    results = []

    for text in dataset_pd["text"]:
        splits = text.split(".")
        splits = [
            split
            for split in splits
            if len("".join(x for x in split if x.isalpha())) > 1
        ]

        input_ids = tokenizer(
            splits,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        ).input_ids
        outputs = model.generate(input_ids, max_new_tokens=128)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs = [string + ". " for string in decoded_outputs]
        output = "".join(decoded_outputs)
        results.append(output)

    dataset_pd["text"] = results

    dataset_pd.to_csv(cfg.task.save_path, index=False)


def add_translated(
    dataset: DatasetDict,
    path: str,
    column_indices: list[int],
):
    dataset = dataset.to_pandas()
    stemmed = path.split(".")[0]
    path_translated_senteces = f"{stemmed}_translated.csv"
    translated = pd.read_csv(path_translated_senteces)
    num_rows = len(translated)
    translated["train_id"] = [float(num) for num in range(10000, 10000 + num_rows)]
    translated.rename(
        columns=dict(zip(translated.columns, dataset.columns)), inplace=True
    )

    selected = set()
    for column_index in column_indices:
        for index, row in translated.iterrows():
            if row[column_index] == 1:
                selected.add(index)

    selected_rows = translated.iloc[list(selected)]
    dataset = pd.concat([dataset, selected_rows], ignore_index=True)
    dataset = shuffle(dataset, random_state=42)
    return Dataset.from_pandas(dataset).remove_columns(["__index_level_0__"])
