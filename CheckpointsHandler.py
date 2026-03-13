import torch
import os
from typing import Literal

class CheckpointsHandler:
    def __init__(self, save_every: int, increasing_metric: bool, output_path: str):
        self.increasing_metric = increasing_metric
        self.save_every = save_every
        self.output_path = output_path
        self.previous_best_value = -float('inf') if increasing_metric else float('inf')

    def check_save_every(self, current_epoch: int) -> bool:
        return current_epoch%self.save_every == 0
    
    def metric_has_improved(self, metric_val: float):
        if self.increasing_metric:
            if metric_val > self.previous_best_value:
                self.previous_best_value = metric_val
                return True
        else:
            if metric_val < self.previous_best_value:
                self.previous_best_value = metric_val
                return True
        return False
    
    # def save_model(self, model: torch.nn.Module , optim, sched , psnr,epoch: int, preds: list[torch.Tensor], loss: float, save_type: str):
    def save_model(self, training_env_state_dict: dict, save_type: Literal["last", "best"]):
        
        output_path = os.path.join(self.output_path, f"{save_type}.pth")
        torch.save(training_env_state_dict, output_path)


    def load_env(self, env_path):
        pass