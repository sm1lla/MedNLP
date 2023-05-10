from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np

def tokenize():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded_dataset = dataset.map(
        preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    


def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text, padding="max_length",
                       truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()

  return encoding

def train():
    dataset = load_dataset()

    labels = [label for label in dataset['train'].features.keys() if label not in ['train_id', 'text']]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-german-dbmdz-uncased",
                                                            problem_type="multi_label_classification",
                                                            num_labels=len(
                                                                labels),
                                                            id2label=id2label,
                                                            label2id=label2id)
