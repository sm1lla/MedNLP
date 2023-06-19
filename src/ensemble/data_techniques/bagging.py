
from pathlib import Path
from src.dataset import get_pd_datasets
import pandas as pd
from sklearn.utils import resample
from datasets import Dataset, DatasetDict

def create_bagging_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_divisions: int,
):
    bagging_datasets = []
    for division_idx in range(num_divisions):

        train = resample(train_df,replace=True)

        # [[path1,label1],[path2,label2]...]  -> [[path1,path2...],[label1, label2...]]
        train = Dataset.from_pandas(train).remove_columns(["__index_level_0__"])
        val = Dataset.from_pandas(val_df)
        test = Dataset.from_pandas(test_df)
        dataset = DatasetDict({"train": train, "val": val, "test": test})

        bagging_datasets.append(dataset)

    return bagging_datasets

def bagging(cfg): 

    train_df, val_df, test_df = get_pd_datasets(cfg)
    return create_bagging_datasets(train_df,val_df,test_df, cfg.task.ensemble_size)


