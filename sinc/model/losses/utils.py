
import torch
from torch.nn import Module

class LossTracker(Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = losses
        self.training = False
        self.count = 0
        for loss in self.losses:
            self.register_buffer(loss, torch.tensor(0.0, device='cpu'), persistent=False)

    def reset(self):
        self.count = 0
        for loss in self.losses:
            getattr(self, loss).__imul__(0)

    def update(self, losses_dict):
        self.count += 1
        for loss_name, loss_val in losses_dict.items():
            getattr(self, loss_name).__iadd__(loss_val)

    def compute(self):
        if self.count == 0:
            raise ValueError("compute should be called after update")
        # compute the mean
        return {loss: getattr(self, loss)/self.count for loss in self.losses}

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            
            if '_multi' in loss:
                if 'bodypart' in loss:
                    loss_type, name, multi, _ = loss.split("_")                
                    name = f'{name}_multiple_bp'
                else:
                    loss_type, name, multi = loss.split("_")                
                    name = f'{name}_multiple'
            else:
                loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name
