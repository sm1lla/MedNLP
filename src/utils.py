import wandb
from omegaconf import DictConfig


def configure_wandb(cfg: DictConfig):
    if cfg.run_name != "None":
        wandb.init(project=cfg.project_name, name=cfg.run_name)
    else:
        wandb.init(project=cfg.project_name)
