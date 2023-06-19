
import random 
from pathlib import Path
from src.dataset import get_pd_datasets
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def create_partitioning_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame, num_divisions:int):
    
    partion_size = int(len(train_df) / num_divisions)
    first_idx = 0
    second_idx = partion_size
    
    partition_datasets = []
    for division_idx in range(num_divisions):
        if division_idx + 1 is not num_divisions:
            train = train_df[first_idx : second_idx]
        else:
            train = train_df[first_idx:]

        first_idx +=partion_size
        second_idx +=partion_size
        # [[path1,label1],[path2,label2]...]  -> [[path1,path2...],[label1, label2...]]
        train = Dataset.from_pandas(train)
        val = Dataset.from_pandas(val_df)
        test = Dataset.from_pandas(test_df)
        dataset = DatasetDict({"train": train, "val": val, "test": test})

        partition_datasets.append(dataset)

    return partition_datasets


def partitioning(cfg):
    train_df, val_df, test_df = get_pd_datasets(cfg)
    return create_partitioning_datasets(train_df,val_df,test_df, cfg.task.ensemble_size)