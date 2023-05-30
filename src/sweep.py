# Import the W&B Python Library and log into W&B

from omegaconf import DictConfig

import wandb

from .train import train


def main(cfg: DictConfig):
    wandb.init(project="my-first-sweep")
    train(cfg)


# Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "eval/f1-score"},
    "parameters": {
        "weight_decay": {"max": 0.1, "min": 0.01},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3,
        },
    },
}


# Start the sweep
def sweep(config: DictConfig):
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
    wandb.init(project="my-first-sweep")
    wandb.agent("m3d0a3sx", function=main, count=10)
