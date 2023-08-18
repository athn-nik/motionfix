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

def parse_npz(filename, allow_pickle=True, framerate_ratio=None,
              chunk_start=None, chunk_duration=None, load_joints=True,
              undo_interaction=False, trim_nointeractions=False):
    npz = np.load(filename, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    npz['chunk_start'] = 0

    if load_joints:
        # get precomputed joint features if they exist
        l = filename.split('/')
        l.insert(-2, 'joints')
        l = os.path.join('/',*l)
        npz_joints = l[:-4] + '_joints' + l[-4:]
        npz_joints = np.load(npz_joints, allow_pickle=allow_pickle)
        npz_joints = {k: npz_joints[k] for k in npz_joints.files}
        npz = {**npz, **npz_joints}

    if framerate_ratio is not None:
        # reduce framerate
        assert isinstance(framerate_ratio, int)
        old_framerate = npz['framerate']
        new_framerate = old_framerate / framerate_ratio
        npz = subsample(npz, framerate_ratio)
        npz['framerate'] = new_framerate
        npz['n_frames'] = npz['body']['params']['transl'].shape[0]

    # undo interaction
    if undo_interaction:
        # TODO: also zero out contact on human
        # no contact to the object
        npz['contact']['object'] = np.zeros_like(npz['contact']['object'])
        # no motion to the object
        temp = np.zeros_like(npz['object']['params']['transl'])
        temp += npz['object']['params']['transl'][:1]
        npz['object']['params']['transl'] = temp
        # no rotation to the object
        npz['object']['params']['global_orient'] = np.zeros_like(npz['object']['params']['global_orient'])

    if trim_nointeractions:
        # trim start and end of clip where no interactions take place
        contact = npz['contact']['object']
        contact_frames = np.nonzero(contact)[0]
        contact_start = contact_frames.min()
        contact_length = contact_frames.max() - contact_start
        npz = cut_chunk(npz, contact_start, contact_length)

    # cut to smaller continuous chunks
    if chunk_duration is not None:
        chunk_length = min(int(chunk_duration * npz['framerate']), npz['n_frames'])
        if chunk_start is None:
            chunk_start = random.randint(0, npz['n_frames'] - chunk_length)
        npz = cut_chunk(npz, chunk_start, chunk_length)
        
    return DotDict(npz)

def cut_chunk(npz, chunk_start, chunk_length):
    """
    cut a chunk of a sequence of length chunk_length | let's get functional here :P
    """
    npz = _cut_chunk(npz, chunk_start=chunk_start, chunk_length=chunk_length)
    # readjust metadata
    if 'trans' in npz.keys():
        npz['n_frames'] = npz['trans'].shape[0]
    else:
        npz['n_frames'] = npz['body']['params']['transl'].shape[0]
    npz['chunk_start'] = chunk_start
    return npz

def _cut_chunk(npz, chunk_start, chunk_length):
    if isinstance(npz, np.ndarray) or isinstance(npz, Tensor):
        return npz[chunk_start:chunk_start + chunk_length]
    elif isinstance(npz, dict):
        return {k: _cut_chunk(v, chunk_start, chunk_length)
                for k, v in npz.items()}
    else:
        return npz

def subsample(npz, ratio):
    """
    Subsample 0-dim (frames) | let's get functional here :P
    """
    if isinstance(npz, np.ndarray) or isinstance(npz, Tensor):
        return npz[::ratio]
    elif isinstance(npz, dict):
        return {k: subsample(v, ratio) for k, v in npz.items()}
    else:
        return npz

# def prepare_params(params, frame_mask, dtype = np.float32):
#     return {k: v[frame_mask].astype(dtype) for k, v in params.items()}

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


def sequential(dims, layernorm=False, end_with=None, nonlinearity=torch.nn.ReLU):
    """
    Instantiate a Sequential with ReLUs
    dims: list with integers specifying the dimentionality of the mlp
    e.g. if you want three layers dims should look like [d1, d2, d3, d4]. d1 is
    the input and d4 the output dimention.
    if dims == [] or dims == [k] then you get the identity module
    """
    if len(dims) <= 1:
        return torch.nn.Identity()

    def linear(i):
        if i == len(dims) - 2:
            if layernorm:
                return [torch.nn.LayerNorm(dims[i]), torch.nn.Linear(dims[i], dims[i + 1])]
            return [torch.nn.Linear(dims[i], dims[i + 1])]
        else:
            if layernorm:
                return [torch.nn.LayerNorm(dims[i]), torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.ReLU()]
            return [torch.nn.Linear(dims[i], dims[i + 1]), nonlinearity()]

    modules = [linear(i) for i in range(len(dims) - 1)]
    if end_with is not None:
        modules.append([end_with()])
    modules = sum(modules, [])
    return torch.nn.Sequential(*modules)

def cast_dict_to_tensors(d):
    if isinstance(d, dict):
        return {k: cast_dict_to_tensors(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float()
    else:
        return d
    

class RunningMaxMin():
    def __init__(self):
        super().__init__()
        self.max = None
        self.min = None

    def forward(self, x):
        x = rearrange(x, '... d -> b d')
        if self.max is None:
            self.max = torch.max(x, dim=0)
            self.min = torch.min(x, dim=0)
        else:
            curr_max = torch.max(x, dim=0)
            self.max = torch.maximum(self.max, curr_max)
            curr_min = torch.min(x, dim=0)
            self.min = torch.minimum(self.min, curr_min)

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, layers_n, **kwargs):
        super().__init__()
        next_pow_two = lambda x: 2**(x - 1).bit_length()
        mid_dims = np.linspace(in_dim, out_dim, layers_n, dtype=int).tolist()[1:-1]
        mid_dims = list(map(next_pow_two, mid_dims))
        self.layers = sequential([in_dim] + mid_dims + [out_dim], **kwargs)

    def forward(self, x):
        return self.layers(x)