import os
import shutil
from pathlib import Path

from omegaconf import DictConfig
from transformers.trainer_callback import TrainerState

import wandb


def configure_wandb(cfg: DictConfig):
    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name if cfg.project_name != "None" else None,
            name=cfg.run_name if cfg.run_name != "None" else None,
            group=cfg.group_name if cfg.group_name != "None" else None,
            reinit=True,
        )


def configure_wandb_without_cfg(project_name: str, run_name: str, group_name: str):
    wandb.init(
        project=project_name if project_name != "None" else None,
        name=run_name if run_name != "None" else None,
        group=group_name if group_name != "None" else None,
        reinit=True,
    )


def add_section_to_metric_log(
    section: str, metric_dict: str, key_substring_to_delete: str = ""
):
    metric_dict = {
        section + "/" + k.replace(key_substring_to_delete, ""): v
        for k, v in metric_dict.items()
    }
    return metric_dict


def finish_wandb(cfg):
    if cfg.use_wandb:
        wandb.run.finish()


def finish_wandb():
    wandb.run.finish()


def get_last_checkpoint_name(folder):
    ckpt_dirs = os.listdir(folder)
    ckpt_dirs = [dir for dir in ckpt_dirs if "checkpoint" in dir]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpt_dirs[-1]

    return last_ckpt


def get_best_checkpoint_path(folder):
    folder = Path(folder)
    last_ckpt = get_last_checkpoint_name(folder)
    state = TrainerState.load_from_json(folder / last_ckpt / "trainer_state.json")

    return state.best_model_checkpoint


def delete_checkpoints(cfg, train_folder: str = None):
    folder = os.getcwd() if not train_folder else train_folder

    checkpoint_folders = list(filter(lambda x: "checkpoint" in x, os.listdir(folder)))

    folder = Path(folder)
    if not cfg.delete:
        checkpoint_folders.remove(Path(get_best_checkpoint_path(folder)).name)

    for checkpoint_folder in checkpoint_folders:
        shutil.rmtree(folder / checkpoint_folder)
