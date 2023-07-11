from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from transformers import AutoTokenizer, Trainer

from .dataset import load_dataset_from_file
from .helpers import get_class_labels
from .preprocessing import tokenize
from .train import initialize_trainer


def add_columns(dataframe: pd.DataFrame, columns: list[str]):
    for column in columns:
        dataframe[column] = 0
    return dataframe


def save_predictions(cfg: DictConfig):
    # get class labels from training dataset
    class_labels = get_class_labels(load_dataset_from_file(cfg.dataset.path))

    # prepare dataframe
    test_dataset_df = pd.read_csv(cfg.task.dataset_path)
    test_dataset_df.drop(columns=["Unnamed: 0"], inplace=True)
    trainer: Trainer = initialize_trainer(cfg)

    test_dataset_df = add_columns(
        test_dataset_df,
        columns=class_labels,
    )

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(Path(cfg.task.model_path))
    test_dataset = Dataset.from_pandas(test_dataset_df)
    test_dataset_dict = DatasetDict(
        {"train": test_dataset, "val": test_dataset, "test": test_dataset}
    )
    encoded_dataset = tokenize(
        test_dataset_dict, labels=class_labels, tokenizer=tokenizer
    )

    # predict
    output = trainer.predict(encoded_dataset["test"])

    # process predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(output.predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= cfg.threshold)] = 1

    # set predicted values and save to csv
    for row_index in range(y_pred.shape[0]):
        test_dataset_df.iloc[row_index, 2:] = y_pred[row_index, :]

    test_dataset_df.to_csv(cfg.task.output_path, index=False)
