from pathlib import Path

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import wandb
from src.evaluate import evaluate_model
from src.utils import add_section_to_metric_log, get_best_checkpoint_path


def evaluate_wandb_run(cfg: DictConfig):
    # get run by wandb api

    # get model path by wandb api

    api = wandb.Api()

    wand_project_path = "mednlp/" + cfg.project_name

    runs = api.runs(wand_project_path)

    # find run
    results = []
    for run in runs:
        if run.name == cfg.run_name:
            results.append(run)
    assert len(results) == 1

    run = results[0]

    # get output_dir of run

    output_dir = run.config["output_dir"]

    # get best checkpoint path

    model_path = get_best_checkpoint_path(output_dir)

    # rewrite cfg and evaluate

    task_config_directory = (
        Path(get_original_cwd()) / "src" / "config" / "task" / "evaluate.yaml"
    )

    cfg.task = OmegaConf.load(task_config_directory)

    cfg.task.model_path = model_path

    old_use_wandb = cfg.use_wandb

    cfg.use_wandb = False
    results = evaluate_model(cfg, use_test=True)
    cfg.use_wandb = old_use_wandb

    if cfg.use_wandb:
        wandb.log(add_section_to_metric_log("newtest", results, "eval_"))

    return results
