import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle


def create_dataset(dataset_path: str, test_size: float = 0.2):
    # load data
    dataset = load_dataset(
        "csv",
        data_files=dataset_path,
        split="train",
    )
    pd_dataset = pd.DataFrame(dataset)
    labels = labels = [
        label for label in pd_dataset.columns if label not in ["train_id", "text"]
    ]

    # get tuples
    pd_dataset["label_tuple"] = pd_dataset[labels].apply(
        lambda row: "".join(row.values.astype(str)), axis=1
    )

    # split data in wether possible for stratified split or not
    counts = pd_dataset["label_tuple"].value_counts().to_dict()
    pd_dataset["tuple_count"] = pd_dataset["label_tuple"].apply(lambda x: counts[x])
    pd_dataset_stratify = pd_dataset[pd_dataset["tuple_count"] != 1]

    pd_dataset_no_stratify = pd_dataset[pd_dataset["tuple_count"] == 1]

    # split them
    stratified_train, stratified_test = train_test_split(
        pd_dataset_stratify,
        test_size=test_size,
        random_state=42,
        stratify=pd_dataset_stratify[["label_tuple"]],
    )
    unstratified_train, unstratified_test = train_test_split(
        pd_dataset_no_stratify, random_state=42, test_size=test_size
    )

    # combine them
    train = pd.concat([stratified_train, unstratified_train])
    test = pd.concat([stratified_test, unstratified_test])

    # clean them
    train = train.drop(["label_tuple", "tuple_count"], axis=1)
    test = test.drop(["label_tuple", "tuple_count"], axis=1)

    # change back to dataset class
    dataset = DatasetDict(
        {"train": dataset.from_pandas(train), "test": dataset.from_pandas(test)}
    )

    dataset = dataset.remove_columns(["__index_level_0__"])
    return dataset


def upsample(dataset: Dataset):
    dataset = dataset.to_pandas()
    classes = minority_classes(dataset)
    dataset_extension = pd.DataFrame(columns=dataset.columns)
    num_samples = 50
    for minority_class in classes:
        count = 0
        while count <= num_samples:
            for row in dataset[dataset[minority_class] == 1].iterrows():
                if count > num_samples:
                    break
                else:
                    dataset_extension = pd.concat(
                        [dataset_extension, pd.DataFrame([row[1]])], ignore_index=True
                    )
                    count += 1

    updated_dataset = pd.concat([dataset, dataset_extension], ignore_index=True)
    shuffle(updated_dataset, random_state=42)
    return Dataset.from_pandas(updated_dataset)


def downsample(dataset: Dataset):
    dataset = dataset.to_pandas()
    classes = majority_classes(dataset)
    for majority_class in classes:
        subset = dataset[dataset[majority_class] == 1]
        new_subset = resample(subset, replace=False, n_samples=300, random_state=42)
        dataset.drop(subset.index, inplace=True)

        dataset = pd.concat([dataset, new_subset], ignore_index=True)

    shuffle(dataset, random_state=42)
    return Dataset.from_pandas(dataset)


def majority_classes(dataset):
    counts = dataset.iloc[:, 2:].sum()
    classes = list(counts[counts > 300][counts < 800].index)

    return classes


def minority_classes(dataset):
    counts = dataset.iloc[:, 2:].sum()
    classes = list(counts[counts < 50].index)

    return classes


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
    plt.savefig(f"{os.getcwd()}/pie_chart.png", format="png", dpi=1200)


def examine_dataset(cfg: DictConfig):
    dataset = create_dataset(cfg.dataset.path, test_size=0.2)
    # Create dataframes for train and test set
    train = dataset["train"]
    test = dataset["test"]
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)
    print(f"training data size  = {len(train_df)}")
    print(f"test data size  = {len(test_df)}")
    dataset_sums = count_class_occurences(train_df, test_df)

    print(train_df[:100])
    # Print number of occurences and percentage for each class
    print(dataset_sums)
    print(class_distribution(dataset_sums))
    pie_chart_distibution(dataset_sums)


def plot_tuple_distribution():
    # load data
    dataset = load_dataset(
        "csv",
        data_files="data/ntcir17_mednlp-sc_sm_de_train_08_05_23.csv",
        split="train",
    )
    pd_dataset = pd.DataFrame(dataset)
    pd_dataset = pd_dataset.drop(["train_id", "text"], axis=1)
    pd_dataset.rename(
        columns=lambda x: x.split(":")[1] if x != "other" else x, inplace=True
    )
    labels = [label for label in pd_dataset.columns]

    # plot tuple_dist
    pd_dataset["label_tuple"] = pd_dataset[labels].apply(
        lambda row: "-".join(
            [labels[i] for i in range(len(labels)) if row.values[i].astype(bool)]
        ),
        axis=1,
    )
    counts = pd_dataset["label_tuple"].value_counts()
    counts.plot.pie()
    counts.to_csv("output/tuple_dist.csv")
    plt.savefig("output/tuple_dist.png")
    plt.clf()

    counts[1:].plot.pie()
    plt.savefig("output/tuple_dist_without_full_zero.png")
    plt.clf()

    # plot tuple_sizes
    tuples_size_counts = (
        pd_dataset["label_tuple"]
        .apply(lambda x: x.count("-") + 1 if x else 0)
        .value_counts()
    )
    tuples_size_counts.plot.pie()
    tuples_size_counts.to_csv("output/tuple_size_counts.csv")
    plt.savefig("output/tuple_size_counts.png")


def print_examples_for_classnames(cfg: DictConfig):
    columnnames = [cfg.task.symptom]

    dataset = create_dataset(cfg.dataset.path, test_size=0.2)
    # Create dataframes for train and test set
    train = dataset["train"]
    test = dataset["test"]
    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)

    # create boolean mask
    train_mask = pd.Series([True] * len(train_df))
    test_mask = pd.Series([True] * len(test_df))
    for column in columnnames:
        train_mask = train_df[column] == 1 & train_mask
        test_mask = test_df[column] == 1 & test_mask

    text = list(train_df[train_mask]["text"])
    text.extend(list(test_df[test_mask]["text"]))

    for element in text:
        print(element)
        print()


def add_generated_samples(dataset):
    dataset = dataset.to_pandas()
    dataset_generated = pd.read_csv(
        f"{get_original_cwd()}/data/generated_tweets_pain_de.csv"
    )
    rows_generated = len(dataset_generated)
    for column in dataset.columns:
        if column == dataset.columns[12]:
            dataset_generated[column] = np.ones(rows_generated, dtype=int)
        elif column == "train_id":
            dataset_generated.insert(
                0, "train_id", np.zeros(rows_generated, dtype=float)
            )
        elif column != "text":
            dataset_generated[column] = np.zeros(rows_generated, dtype=int)

    updated_dataset = pd.concat([dataset, dataset_generated], ignore_index=True)
    shuffle(updated_dataset, random_state=42)
    return Dataset.from_pandas(updated_dataset)
