from copy import copy
import numpy as np
import torch
from torch import Tensor
import logging
import os
import random
from einops import rearrange
def freeze(model) -> None:
    r"""
    Freeze all params for inference.
    """
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

# A logger for this file
log = logging.getLogger(__name__)
def to_tensor(array):
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array)

def DotDict(in_dict):
    if isinstance(in_dict, dotdict):
        return in_dict 
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
        if isinstance(v,dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def cast_dict_to_tensors(d, device="cpu"):
    if isinstance(d, dict):
        return {k: cast_dict_to_tensors(v, device) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float().to(device)
    elif isinstance(d, torch.Tensor):
        return d.to(device)
    else:
        return d