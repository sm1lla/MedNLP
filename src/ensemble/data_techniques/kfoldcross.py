import os

import pandas as pd
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from sklearn.model_selection import KFold

from src.dataset import get_pd_datasets

def create_kfold_cross_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_divisions: int,
):
    data = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Split {len(data)} Training samples in {num_divisions} partitions")
    kfolds_datasets = []

    kf = KFold(n_splits=num_divisions)
    for train_index, val_index in kf.split(data):
        print("TRAIN:", train_index, "Val:", val_index)
        train = data.iloc[train_index]
        val = data.iloc[val_index]

        # [[path1,label1],[path2,label2]...]  -> [[path1,path2...],[label1, label2...]]
        train = Dataset.from_pandas(train).remove_columns(["__index_level_0__"])
        val = Dataset.from_pandas(val).remove_columns(["__index_level_0__"])
        test = Dataset.from_pandas(test_df)
        dataset = DatasetDict({"train": train, "val": val, "test": test})

        kfolds_datasets.append(dataset)

    return kfolds_datasets


def kfoldcross(cfg: DictConfig):

    train_df, val_df, test_df = get_pd_datasets(cfg)

    kfolds_datasets = create_kfold_cross_datasets(
        train_df, val_df, test_df, cfg.task.ensemble_size
    )

    return kfolds_datasets