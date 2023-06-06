import numpy as np


def tokenize(dataset, labels, tokenizer):
    encoded_dataset = dataset.map(
        lambda x: preprocess_data(x, labels, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    encoded_dataset.set_format("torch")
    return encoded_dataset


def preprocess_data(examples, labels, tokenizer):
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding
