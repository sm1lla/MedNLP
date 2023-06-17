from omegaconf import DictConfig
import torch
from src.train import train, initialize_trainer
from src.evaluate import evaluate_model
from pathlib import Path
import numpy as np 
import os
from transformers.trainer_callback import TrainerState
from src.utils import configure_wandb_without_cfg, finish_wandb, add_section_to_metric_log
import wandb


def get_best_checkpoint(folder):
    ckpt_dirs = os.listdir(folder)
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
    last_ckpt = ckpt_dirs[-1]

    state = TrainerState.load_from_json(folder/last_ckpt/"trainer_state.json")
    
    return state.best_model_checkpoint


class estimator():
    def __init__(self, name, cfg: DictConfig):
        self.name = name
        self.cfg = DictConfig(cfg)
        self.cfg.run_name = self.name
        self.folder = Path(cfg.task.ensemble_path)  / self.name
        self.cfg.task.model_path = self.folder / get_best_checkpoint(self.folder)

    def load_model(self):
        raise NotImplementedError() 
    
    def clear_session(self):
        torch.cuda.empty_cache()
        
    def predict(self,texts:list):
        raise NotImplementedError()        
    

    def train(self):
        train(self.cfg,self.folder)  

    def train_on_selected_data(self,dataset):     
        train(self.cfg, dataset, self.folder)
        
    def get_predictions_and_labels_on_datast(self,on_test_data:bool=False):    
        trainer = initialize_trainer(self.cfg,on_test_data)
        predictions,labels, metrics = trainer.predict(trainer.eval_dataset)
        
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= self.cfg.threshold)] = 1

        return predictions, probs, labels

    def validate(self,use_test:bool = False):
        if not use_test:
            return evaluate_model(self.cfg,use_test)         
        else:
            #log to wandb 
            configure_wandb_without_cfg(self.cfg.project_name,self.name,self.cfg.group_name)
            wandb.log(add_section_to_metric_log("test",evaluate_model(self.cfg,use_test),"eval_"))
            finish_wandb()
    def get_prediction_scores(self,  image_paths: list[str]):
        raise NotImplementedError()         
                  
    def predict_from_prediction_scores(self, prediction_scores): 
        raise NotImplementedError()    