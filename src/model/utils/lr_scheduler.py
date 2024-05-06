from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from functools import partial
import math
import torch
import math
import functools

def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations, final_learning_rate, initial_learning_rate):
    """
    Adjusted function to handle linear warmup and cosine decay that correctly transitions to the final learning rate.
    """
    if iteration <= warmup_iterations:
        # Linear warmup: multiplier increases from 0 to 1
        return iteration / warmup_iterations
    else:
        # Cosine decay phase: decay from 1 to final_learning_rate/initial_learning_rate
        decayed = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        decayed = 0.5 * (1 + math.cos(math.pi * decayed))
        # Normalize the decay such that it ends at final_learning_rate/initial_learning_rate
        decay_multiplier = decayed * (1 - final_learning_rate / initial_learning_rate) + final_learning_rate / initial_learning_rate
        return decay_multiplier

def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup, lr_final, lr_initial):
    _decay_func = functools.partial(
        _cosine_decay_warmup, 
        warmup_iterations=T_warmup, total_iterations=T_max,
        final_learning_rate=lr_final,
        initial_learning_rate=lr_initial
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler
