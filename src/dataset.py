import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle


def create_dataset(dataset_path: str, val_size: float = 0.15, test_size: float = 0.15):
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

    pd_dataset_stratify = pd_dataset[pd_dataset["tuple_count"] >= 3]

    pd_dataset_no_stratify = pd_dataset[pd_dataset["tuple_count"] < 3]

    # split them
    stratified_train, stratified_tmp = train_test_split(
        pd_dataset_stratify,
        test_size=val_size + test_size,
        random_state=42,
        stratify=pd_dataset_stratify[["label_tuple"]],
    )
    unstratified_train, unstratified_tmp = train_test_split(
        pd_dataset_no_stratify, random_state=42, test_size=val_size + test_size
    )

    # create val and test
    # first count them to know which can be stratified
    counts = stratified_tmp["label_tuple"].value_counts().to_dict()
    stratified_tmp["tuple_count"] = stratified_tmp["label_tuple"].apply(
        lambda x: counts[x]
    )

    unstratified_tmp = pd.concat(
        [unstratified_tmp, stratified_tmp[stratified_tmp["tuple_count"] == 1]],
        ignore_index=True,
    )

    stratified_tmp = stratified_tmp[stratified_tmp["tuple_count"] != 1]

    # split tmp into val test

    stratified_test, stratified_val = train_test_split(
        stratified_tmp,
        test_size=test_size / (val_size + test_size),
        random_state=42,
        stratify=stratified_tmp[["label_tuple"]],
    )
    unstratified_test, unstratified_val = train_test_split(
        unstratified_tmp, random_state=42, test_size=test_size / (val_size + test_size)
    )

    # combine them
    train = pd.concat([stratified_train, unstratified_train])
    val = pd.concat([stratified_val, unstratified_val])
    test = pd.concat([stratified_test, unstratified_test])

    # clean them
    train = train.drop(["label_tuple", "tuple_count"], axis=1)
    val = val.drop(["label_tuple", "tuple_count"], axis=1)
    test = test.drop(["label_tuple", "tuple_count"], axis=1)

    # change back to dataset class
    dataset = DatasetDict(
        {
            "train": dataset.from_pandas(train),
            "val": dataset.from_pandas(val),
            "test": dataset.from_pandas(test),
        }
    )

    dataset = dataset.remove_columns(["__index_level_0__"])
    return dataset


def save_splits_to_file(path):
    dataset = create_dataset(path)
    path_stemmed = path.split(".")[0]
    pd.DataFrame(dataset["train"]).to_csv(f"{path_stemmed}_train.csv")
    pd.DataFrame(dataset["val"]).to_csv(f"{path_stemmed}_val.csv")
    pd.DataFrame(dataset["test"]).to_csv(f"{path_stemmed}_test.csv")


def load_dataset_from_file(path):
    path_stemmed = path.split(".")[0]
    dataset_files = {
        "train": f"{path_stemmed}_train.csv",
        "val": f"{path_stemmed}_val.csv",
        "test": f"{path_stemmed}_test.csv",
    }
    dataset = load_dataset("csv", data_files=dataset_files)
    dataset = dataset.remove_columns(["Unnamed: 0"])
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


def sum_df(df: pd.DataFrame):
    # helper function for count_class_occurences
    sums = df.drop(["train_id", "text"], axis=1)
    sums.rename(columns=lambda x: x.split(":")[1] if x != "other" else x, inplace=True)
    sums = sums.sum()

    return sums


def count_class_occurences(
    train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame
):
    sums_train = sum_df(train_set)
    sums_val = sum_df(val_set)
    sums_test = sum_df(test_set)

    sums = pd.DataFrame({"train": sums_train, "val": sums_val, "test": sums_test})
    return sums


def class_distribution(dataset_sums: pd.DataFrame):
    print(dataset_sums)
    percentages = pd.DataFrame()
    percentages["train %"] = round(
        (dataset_sums["train"] / dataset_sums["train"].sum()) * 100, 3
    )
    percentages["val %"] = round(
        (dataset_sums["val"] / dataset_sums["val"].sum()) * 100, 3
    )
    percentages["test %"] = round(
        (dataset_sums["test"] / dataset_sums["test"].sum()) * 100, 3
    )
    return percentages


def pie_chart_distibution(dataset_sums: pd.DataFrame):
    dataset_sums.plot.pie(subplots=True, legend=False, ylabel="", figsize=(17, 14))
    plt.savefig(f"{os.getcwd()}/pie_chart.png", format="png", dpi=1200)


def get_pd_datasets(cfg: DictConfig):
    dataset = load_dataset_from_file(cfg.dataset.path)
    # Create dataframes for train and test set
    train = dataset["train"]
    val = dataset["val"]
    test = dataset["test"]
    train_df = pd.DataFrame(train)
    val_df = pd.DataFrame(val)
    test_df = pd.DataFrame(test)

    return train_df, val_df, test_df


def examine_dataset(cfg: DictConfig):
    train_df, val_df, test_df = get_pd_datasets(cfg)
    print(f"training data size  = {len(train_df)}")
    print(f"val data size  = {len(val_df)}")
    print(f"test data size  = {len(test_df)}")
    dataset_sums = count_class_occurences(train_df, val_df, test_df)

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
    # todo: refactor (why the mask ) just load df from csv and print column

    dataset = load_dataset_from_file(cfg.dataset.path)
    # Create dataframes for train and test set
    train = dataset["train"]
    val = dataset["val"]
    test = dataset["test"]
    train_df = pd.DataFrame(train)
    val_df = pd.DataFrame(val)
    test_df = pd.DataFrame(test)

    # create boolean mask
    train_mask = pd.Series([True] * len(train_df))
    val_mask = pd.Series([True] * len(val_df))
    test_mask = pd.Series([True] * len(test_df))
    for column in columnnames:
        train_mask = train_df[column] == 1 & train_mask
        val_mask = val_df[column] == 1 & val_mask
        test_mask = test_df[column] == 1 & test_mask

    text = list(train_df[train_mask]["text"])
    text.extend(list(val_df[val_mask]["text"]))
    text.extend(list(test_df[test_mask]["text"]))

    for element in text:
        print(element)
        print()
