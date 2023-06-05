import numpy as np
import torch
from omegaconf import DictConfig

from .train import initialize_trainer


def predict(cfg: DictConfig):
    trainer = initialize_trainer(cfg)
    output = trainer.predict(trainer.eval_dataset)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(output.predictions))
    indices = [x for x, y in enumerate(output.label_ids) if len(np.nonzero(y)[0])]

    for index in indices:
        print(probs[index, :])
        print(output.label_ids[index])
