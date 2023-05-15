import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def create_dataset(test_size: float):
    dataset = load_dataset(
        "csv",
        data_files=to_absolute_path("data/ntcir17_mednlp-sc_sm_de_train_03_04_23.csv"),
        split="train",
    )
    dataset = dataset.train_test_split(test_size=test_size, shuffle=False)
    return dataset


def count_class_occurences(train_set: pd.DataFrame, test_set: pd.DataFrame):
    sums_train = train_set.drop(["train_id", "text"], axis=1)
    sums_train.rename(
        columns=lambda x: x.split(":")[1] if x != "other" else x, inplace=True
    )
    sums_train = sums_train.sum()
    sums_test = test_set.drop(["train_id", "text"], axis=1)
    sums_test.rename(
        columns=lambda x: x.split(":")[1] if x != "other" else x, inplace=True
    )
    sums_test = sums_test.sum()
    sums = pd.DataFrame({"train": sums_train, "test": sums_test})
    return sums


def class_distribution(dataset_sums: pd.DataFrame):
    print(dataset_sums)
    percentages = pd.DataFrame()
    percentages["train %"] = round(
        (dataset_sums["train"] / dataset_sums["train"].sum()) * 100, 3
    )
    percentages["test %"] = round(
        (dataset_sums["test"] / dataset_sums["test"].sum()) * 100, 3
    )
    return percentages


def pie_chart_distibution(dataset_sums: pd.DataFrame):
    dataset_sums.plot.pie(subplots=True, legend=False, ylabel="", figsize=(17, 14))
    plt.savefig("")


def examine_dataset(cfg: DictConfig):
    dataset = create_dataset(test_size=0.2)

    # Create dataframes for train and test set
    train = dataset["train"]
    test = dataset["test"]
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    dataset_sums = count_class_occurences(train_df, test_df)

    # Print number of occurences and percentage for each class
    print(dataset_sums)
    print(class_distribution(dataset_sums))
    pie_chart_distibution(dataset_sums)
