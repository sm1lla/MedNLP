from omegaconf import DictConfig

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
