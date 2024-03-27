import logging
import random
from glob import glob
from os import listdir
import bisect
from os.path import exists, join
from pathlib import Path
from typing import List, Dict
import joblib
import numpy as np
from omegaconf import DictConfig
import smplx
import torch
from einops import rearrange
from src import data
from src.data.tools.collate import collate_tensor_with_padding
from src.tools.geometry import matrix_to_euler_angles, matrix_to_rotation_6d
from pytorch_lightning import LightningDataModule
from smplx.joint_names import JOINT_NAMES
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from src.data.base import BASEDataModule
from src.utils.genutils import DotDict, cast_dict_to_tensors, to_tensor
from src.tools.transforms3d import (
    change_for, local_to_global_orient, transform_body_pose, remove_z_rot,
    rot_diff, get_z_rot)
from src.tools.transforms3d import canonicalize_rotations
from src.model.utils.smpl_fast import smpl_forward_fast
from src.utils.genutils import freeze
from src.data.bodilex import BodilexDataset
from src.data.sinc_synth import SincSynthDataset
from src.data.humanml3d import HumanML3DDataset
from src.utils.file_io import read_json, write_json
import os
# A logger for this file
log = logging.getLogger(__name__)



class MixerDataModule(BASEDataModule):

    def get_set_item_idx(self, concat_dataset, idx):
        if idx < 0:
            if -idx > len(concat_dataset):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(concat_dataset) + idx
        dataset_idx = bisect.bisect_right(concat_dataset.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - concat_dataset.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __init__(self,
                 load_feats: List[str],
                 dataset_names: List[str],
                 dataname: str,
                 datapaths: Dict[str, str],
                 batch_size: int = 32,
                 num_workers: int = 16,
                 debug: bool = False,
                 debug_datapaths: str = "",
                 preproc: DictConfig = None,
                 smplh_path: str = "",
                 rot_repr: str = "6d",
                 batch_sampler=None,
                 hml3d_ratio=40,
                 sinc_synth_ratio=30,
                 bodilex_ratio=30,
                 **kwargs):
        if batch_sampler is not None:
            dataset_percentages = {'hml3d': hml3d_ratio / 100,
                                   'sinc_synth': sinc_synth_ratio / 100,
                                   'bodilex': bodilex_ratio / 100 }
        else:
            dataset_percentages = None
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         load_feats=load_feats, 
                         batch_sampler=batch_sampler,
                         dataset_percentages=dataset_percentages)
        dataset_names = [dname for dname in dataset_names if dataset_percentages[dname] > 0]
        self.datapaths = [datapaths[name] for name in dataset_names]
        # concatenate the sorted datapaths
        self.dataname = dataname
        self.datapaths = datapaths
        self.debug = debug
        self.smpl_p = smplh_path
        self.rot_repr = rot_repr
        self.preproc = preproc
        self.load_feats = load_feats
        DATA_CLASS_MAP = {
            'bodilex': BodilexDataset,
            'sinc_synth': SincSynthDataset,
            'hml3d': HumanML3DDataset,
        }
        dataset_list = []
        for name in dataset_names:
            assert name in DATA_CLASS_MAP.keys(), f"Dataset {name} not recognized"
            dataset_list.append(DATA_CLASS_MAP[name].load_and_instantiate(datapath=self.datapaths[name],
                                                    debug=self.debug,
                                                    smplh_path=self.smpl_p,
                                                    n_body_joints=self.preproc.n_body_joints,
                                                    stats_file=self.preproc.stats_file,
                                                    norm_type=self.preproc.norm_type,
                                                    rot_repr=self.rot_repr,
                                                    load_feats=self.load_feats,
                                                    do_augmentations=True))
        self.dataset = {split: ConcatDataset([d[split] for d in dataset_list])
                         for split in ['train', 'val', 'test']}

        self.stats = self.calculate_feature_stats(self.dataset['train'])
        
        # NOTE: arbitrarily choose to use the nfeats of the first dataset
        self.nfeats = self.dataset['train'].datasets[0].nfeats

    def _canonica_facefront(self, rotations, translation):
        rots_motion = rotations
        trans_motion = translation
        datum_len = rotations.shape[0]
        rots_motion_rotmat = transform_body_pose(rots_motion.reshape(datum_len,
                                                           -1, 3),
                                                           'aa->rot')
        orient_R_can, trans_can = canonicalize_rotations(rots_motion_rotmat[:,
                                                                             0],
                                                         trans_motion)            
        rots_motion_rotmat_can = rots_motion_rotmat
        rots_motion_rotmat_can[:, 0] = orient_R_can
        translation_can = trans_can - trans_can[0]
        rots_motion_aa_can = transform_body_pose(rots_motion_rotmat_can,
                                                 'rot->aa')
        rots_motion_aa_can = rearrange(rots_motion_aa_can, 'F J d -> F (J d)',
                                       d=3)
        return rots_motion_aa_can, translation_can


    def calculate_feature_stats(self, dataset):
        stat_path = self.preproc.stats_file
        if self.debug:
            stat_path = stat_path.replace('.npy', '_debug.npy')

        if not exists(stat_path):
            log.info(f"No dataset stats found. Calculating and saving to {stat_path}")
            
            feature_names = dataset.datasets[0].get_all_features(0).keys()
            feature_dict = {name.replace('_source', ''): [] for name in feature_names
                            if '_target' not in name}
            for i in tqdm(range(len(dataset))):
                set_idx, sample_idx = self.get_set_item_idx(dataset, i)
                x = dataset.datasets[set_idx].get_all_features(sample_idx)
                feature_names = dataset.datasets[set_idx].get_all_features(0).keys()
                for name in feature_names:
                    x_new = x[name]
                    name = name.replace('_source', '')
                    name = name.replace('_target', '')
                    if torch.is_tensor(x_new):
                        feature_dict[name].append(x_new)
            feature_dict = {name: torch.cat(feature_dict[name],
                                            dim=0).float()
                                             for name in feature_dict.keys()}
            stats = {name: {'max': x.max(0)[0].numpy(),
                            'min': x.min(0)[0].numpy(),
                            'mean': x.mean(0).numpy(),
                            'std': x.std(0).numpy()}
                     for name, x in feature_dict.items()}
            
            # stats_source = {f'{name}_source': v for name, v in stats.items()}
            # stats_target = {f'{name}_target': v for name, v in stats.items()}
            # stats_dup = stats_source | stats_target
            log.info("Calculated statistics for the following features:")
            log.info(feature_names)
            log.info(f"saving to {stat_path}")
            np.save(stat_path, stats)
        log.info(f"Will be loading feature stats from {stat_path}")
        stats = np.load(stat_path, allow_pickle=True)[()]
        return stats
