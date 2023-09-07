from copy import copy
import numpy as np
import torch
from torch import Tensor
import logging
import os
import random
from einops import rearrange

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
    
def cast_dict_to_tensors(d):
    if isinstance(d, dict):
        return {k: cast_dict_to_tensors(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float()
    else:
        return d