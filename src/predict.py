import numpy as np
import torch
from omegaconf import DictConfig

from .dataset import load_dataset_from_file
from .train import initialize_trainer


def predict(cfg: DictConfig):
    trainer = initialize_trainer(cfg)
    dataset = load_dataset_from_file(cfg.dataset.path)
    labels = trainer.eval_dataset["labels"]
    texts = dataset["val"]["text"]
    output = trainer.predict(trainer.eval_dataset)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(output.predictions))
    indices = [x for x, y in enumerate(output.label_ids) if len(np.nonzero(y)[0])]
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= cfg.threshold)] = 1

    if cfg.task.print_probs:
        for index in indices:
            print(probs[index, :])
            print(output.label_ids[index])
    if cfg.task.print_wrong_predictions:
        for index in range(labels.shape[0]):
            if not np.array_equal(y_pred[index], np.array(labels[index])):
                print(f"tweet: {texts[index]}")
                print(f"predicted: {y_pred[index]}")
                print(f"labels   : {np.array(labels[index])}")
