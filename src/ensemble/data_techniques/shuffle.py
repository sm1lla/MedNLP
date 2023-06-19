
from pathlib import Path
from src.dataset import get_pd_datasets
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def create_shuffled_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_divisions: int,
):
    shuffled_datasets = []
    for division_idx in range(num_divisions):
        data = pd.concat([train_df, val_df], ignore_index=True)
        data = resample(data,replace=False)
        train,val = train_test_split(data, test_size=len(val_df)/len(data))

        # [[path1,label1],[path2,label2]...]  -> [[path1,path2...],[label1, label2...]]
        train = Dataset.from_pandas(train).remove_columns(["__index_level_0__"])
        val = Dataset.from_pandas(val).remove_columns(["__index_level_0__"])
        test = Dataset.from_pandas(test_df)
        dataset = DatasetDict({"train": train, "val": val, "test": test})

        shuffled_datasets.append(dataset)

    return shuffled_datasets

def shuffle(cfg): 

    train_df, val_df, test_df = get_pd_datasets(cfg)
    return create_shuffled_datasets(train_df,val_df,test_df, cfg.task.ensemble_size)


