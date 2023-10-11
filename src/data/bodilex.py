import logging
import random
from glob import glob
from os import listdir
from os.path import exists, join
from pathlib import Path
from typing import List
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
from torch.quantization.observer import \
    MovingAveragePerChannelMinMaxObserver as mmo
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.data.base import BASEDataModule
from src.utils.genutils import DotDict, cast_dict_to_tensors, to_tensor
from src.tools.transforms3d import (
    change_for, local_to_global_orient, transform_body_pose, remove_z_rot,
    rot_diff, get_z_rot)
from src.tools.transforms3d import canonicalize_rotations

# A logger for this file
log = logging.getLogger(__name__)


class BodilexDataset(Dataset):
    def __init__(self, data: list, n_body_joints: int,
                 stats_file: str, norm_type: str,
                 smplh_path: str, rot_repr: str = "6d",
                 load_feats: List[str] = None,
                 do_augmentations=False):
        self.data = data
        self.norm_type = norm_type
        self.rot_repr = rot_repr
        self.load_feats = load_feats
        self.do_augmentations = do_augmentations
        
        # self.seq_parser = SequenceParserAmass(self.cfg)
        bm = smplx.create(model_path=smplh_path, model_type='smplh', ext='npz')
        self.body_chain = bm.parents
        stat_path = join(stats_file)
        self.stats = None
        self.n_body_joints = n_body_joints
        self.joint_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        if exists(stat_path):
            stats = np.load(stat_path, allow_pickle=True)[()]
            self.stats = cast_dict_to_tensors(stats)
        self._feat_get_methods = {
            "body_transl": self._get_body_transl,
            "body_transl_z": self._get_body_transl_z,
            "body_transl_delta": self._get_body_transl_delta,
            "body_transl_delta_pelv": self._get_body_transl_delta_pelv,
            "body_transl_delta_pelv_xy": self._get_body_transl_delta_pelv_xy,
            "body_orient": self._get_body_orient,
            "body_orient_xy": self._get_body_orient_xy,
            "body_orient_delta": self._get_body_orient_delta,
            "body_pose": self._get_body_pose,
            "body_pose_delta": self._get_body_pose_delta,

            "body_joints": self._get_body_joints,
            "body_joints_rel": self._get_body_joints_rel,
            "body_joints_vel": self._get_body_joints_vel,
            "joint_global_oris": self._get_joint_global_orientations,
            "joint_ang_vel": self._get_joint_angular_velocity,
            "wrists_ang_vel": self._get_wrists_angular_velocity,
            "wrists_ang_vel_euler": self._get_wrists_angular_velocity_euler,
        }
        self._meta_data_get_methods = {
            "framerate": self._get_framerate,
        }
        self.nfeats = self.get_features_dimentionality()

    def get_features_dimentionality(self):
        """
        Get the dimentionality of the concatenated load_feats
        """
        item = self.__getitem__(0)

        return [item[feat + '_source'].shape[-1] for feat in self.load_feats
                if feat in self._feat_get_methods.keys()]

    def normalize_feats(self, feats, feats_name):
        if feats_name not in self.stats.keys():
            log.error(f"Tried to normalise {feats_name} but did not found stats \
                      for this feature. Try running calculate_statistics.py again.")
        if self.norm_type == "std":
            mean, std = self.stats[feats_name]['mean'].to(feats.device), self.stats[feats_name]['std'].to(feats.device)
            return (feats - mean) / (std + 1e-5)
        elif self.norm_type == "norm":
            max, min = self.stats[feats_name]['max'].to(feats.device), self.stats[feats_name]['min'].to(feats.device)
            return (feats - min) / (max - min + 1e-5)

    def _get_body_joints(self, data):
        joints = to_tensor(data['joint_positions'][:, :self.n_body_joints, :])
        return rearrange(joints, '... joints dims -> ... (joints dims)')

    def _get_joint_global_orientations(self, data):
        body_pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        body_orient = to_tensor(data['rots'][..., :3])
        joint_glob_oris = local_to_global_orient(body_orient, body_pose,
                                                 self.body_chain,
                                                 input_format='aa',
                                                 output_format="rotmat")
        return rearrange(joint_glob_oris, '... j k d -> ... (j k d)')

    def _get_joint_angular_velocity(self, data):
        pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        # pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        # pose = axis_angle_to_matrix(to_tensor(pose))
        pose = transform_body_pose(pose, "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_rotation_6d(rot_diffs), '... j c -> ... (j c)')

    def _get_wrists_angular_velocity(self, data):
        pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        # pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        # pose = axis_angle_to_matrix(to_tensor(pose[..., 19:21, :]))
        pose = transform_body_pose(pose, "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_rotation_6d(rot_diffs), '... j c -> ... (j c)')

    def _get_wrists_angular_velocity_euler(self, data):
        pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
        pose = rearrange(pose, '... (j c) -> ... j c', c=3)
        pose = transform_body_pose(to_tensor(pose[..., 19:21, :]), "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_euler_angles(rot_diffs, "XYZ"), '... j c -> ... (j c)')

    def _get_body_joints_vel(self, data):
        joints = to_tensor(data['joint_positions'][:, :self.n_body_joints, :])
        joint_vel = joints - joints.roll(1, 0)  # shift one right and subtract
        joint_vel[0] = 0
        return rearrange(joint_vel, '... j c -> ... (j c)')

    def _get_body_joints_rel(self, data):
        """get body joint coordinates relative to the pelvis"""
        joints = to_tensor(data['joint_positions'][:, :self.n_body_joints, :])
        pelvis_transl = to_tensor(joints[:, 0, :])
        joints_glob = to_tensor(joints[:, :self.n_body_joints, :])
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient = transform_body_pose(pelvis_orient, "aa->rot").float()
        # relative_joints = R.T @ (p_global - pelvis_translation)
        rel_joints = torch.einsum('fdi,fjd->fji', pelvis_orient, joints_glob - pelvis_transl[:, None, :])
        return rearrange(rel_joints, '... j c -> ... (j c)')

    @staticmethod
    def _get_framerate(data):
        """get framerate"""
        return torch.tensor([30])

    @staticmethod
    def _get_chunk_start(data):
        """get number of original sequence frames"""
        return torch.tensor([data['chunk_start']])

    @staticmethod
    def _get_num_frames(data):
        """get number of original sequence frames"""
        return torch.tensor([data['rots'].shape[0]])

    def _get_body_transl(self, data):
        """get body pelvis tranlation"""
        return to_tensor(data['trans'])
        # body.translation is NOT the same as the pelvis translation
        # TODO: figure out why
        # return to_tensor(data.body.params.transl)

    def _get_body_transl_z(self, data):
        """get body pelvis tranlation"""
        return to_tensor(data['trans'])[..., 2:] # only z

    def _get_body_transl_delta(self, data):
        """get body pelvis tranlation delta"""
        trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        trans_vel[0] = 0  # zero out velocity of first frame
        return trans_vel

    def _get_body_transl_delta_pelv(self, data):
        """
        get body pelvis tranlation delta relative to pelvis coord.frame
        v_i = t_i - t_{i-1} relative to R_{i-1}
        """
        trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        pelvis_orient = transform_body_pose(to_tensor(data['rots'][..., :3]), "aa->rot")
        trans_vel_pelv = change_for(trans_vel, pelvis_orient.roll(1, 0))
        trans_vel_pelv[0] = 0  # zero out velocity of first frame
        return trans_vel_pelv

    def _get_body_transl_delta_pelv_xy(self, data):
        """
        get body pelvis tranlation delta while removing the global z rotation of the pelvis
        v_i = t_i - t_{i-1} relative to R_{i-1}_xy
        """
        trans = to_tensor(data['trans'])
        trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
        pelvis_orient = to_tensor(data['rots'][..., :3])
        R_z = get_z_rot(pelvis_orient, in_format="aa")
        # rotate -R_z
        trans_vel_pelv = change_for(trans_vel, R_z.roll(1, 0), forward=True)
        trans_vel_pelv[0] = 0  # zero out velocity of first frame
        return trans_vel_pelv

    def _get_body_orient(self, data):
        """get body global orientation"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        if self.rot_repr == "6d":
            # axis-angle to rotation matrix & drop last row
            pelvis_orient = transform_body_pose(pelvis_orient, "aa->6d")
        return pelvis_orient

    def _get_body_orient_xy(self, data):
        """get body global orientation"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        if self.rot_repr == "6d":
            # axis-angle to rotation matrix & drop last row
            pelvis_orient_xy = remove_z_rot(pelvis_orient, in_format="aa")
        return pelvis_orient_xy

    def _get_body_orient_delta(self, data):
        """get global body orientation delta"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient_delta = rot_diff(pelvis_orient, in_format="aa",
                                       out_format=self.rot_repr)
        return pelvis_orient_delta

    def _get_body_pose(self, data):
        """get body pose"""
        # default is axis-angle representation: Frames x (Jx3) (J=21)
        pose = to_tensor(data['rots'][..., 3:3 + 21*3])  # drop pelvis orientation
        pose = transform_body_pose(pose, f"aa->{self.rot_repr}")
        return pose

    def _get_body_pose_delta(self, data):
        """get body pose rotational deltas"""
        # default is axis-angle representation: Frames x (Jx3) (J=21)
        pose = to_tensor(data['rots'][..., 3:3 + 21*3])  # drop pelvis orientation
        pose_diffs = rot_diff(pose, in_format="aa", out_format=self.rot_repr)
        return pose_diffs

    def __len__(self):
        return len(self.data)

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


    def __getitem__(self, idx):
        datum = self.data[idx]
        # perform augmentations except when in test mode
        # if self.do_augmentations:
        #     datum = self.seq_parser.augment_npz(datum)

        # CANONICALIZE MOTION AND BRING IT TO 0
        for mtype in ['motion_source', 'motion_target']:
            
            rots_can, trans_can = self._canonica_facefront(datum[mtype]['rots'],
                                                           datum[mtype]['trans']
                                                           )

            datum[mtype]['rots'] = rots_can
            datum[mtype]['trans'] = trans_can

        data_dict_source = {f'{feat}_source': self._feat_get_methods[feat](datum['motion_source'])
                            for feat in self.load_feats}
        data_dict_target = {f'{feat}_target': self._feat_get_methods[feat](datum['motion_target'])
                            for feat in self.load_feats}
        meta_data_dict = {feat: method(datum)
                          for feat, method in self._meta_data_get_methods.items()}
        data_dict = {**data_dict_source, **data_dict_target, **meta_data_dict}
        data_dict['length_source'] = len(data_dict['body_pose_source'])
        data_dict['length_target'] = len(data_dict['body_pose_target'])
        data_dict['text'] = datum['text']
        data_dict['split'] = datum['split']
        data_dict['id'] = datum['id']
        # data_dict['dims'] = self._feat_dims
        return DotDict(data_dict)

    def npz2feats(self, idx, npz):
        """turn npz data to a proper features dict"""
        data_dict = {feat: self._feat_get_methods[feat](npz)
                     for feat in self.load_feats}
        if self.stats is not None:
            norm_feats = {f"{feat}_norm": self.normalize_feats(data, feat)
                        for feat, data in data_dict.items()
                        if feat in self.stats.keys()}
            data_dict = {**data_dict, **norm_feats}
        meta_data_dict = {feat: method(npz)
                          for feat, method in self._meta_data_get_methods.items()}
        data_dict = {**data_dict, **meta_data_dict}
        data_dict['filename'] = self.file_list[idx]['filename']
        # data_dict['split'] = self.file_list[idx]['split']
        return DotDict(data_dict)

    def get_all_features(self, idx):
        datum = self.data[idx]        
        for mtype in ['motion_source', 'motion_target']:
            
            rots_can, trans_can = self._canonica_facefront(datum[mtype]['rots'],
                                                           datum[mtype]['trans']
                                                           )

            datum[mtype]['rots'] = rots_can
            datum[mtype]['trans'] = trans_can

        data_dict_source = {f'{feat}_source': self._feat_get_methods[feat](datum['motion_source'])
                            for feat in self.load_feats}
        data_dict_target = {f'{feat}_target': self._feat_get_methods[feat](datum['motion_target'])
                            for feat in self.load_feats}
        meta_data_dict = {feat: method(datum)
                          for feat, method in self._meta_data_get_methods.items()}
        data_dict = {**data_dict_source, **data_dict_target, **meta_data_dict}
        return DotDict(data_dict)


class BodilexDataModule(BASEDataModule):

    def __init__(self,
                 load_feats: List[str],
                 batch_size: int = 32,
                 num_workers: int = 16,
                 datapath: str = "",
                 debug: bool = False,
                 debug_datapath: str = "",
                 preproc: DictConfig = None,
                 smplh_path: str = "",
                 dataname: str = "",
                 rot_repr: str = "6d",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         load_feats=load_feats)
        self.dataname = dataname
        self.batch_size = batch_size

        self.datapath = datapath
        self.debug_datapath = debug_datapath

        self.load_feats = load_feats
        self.debug = debug
        self.dataset = {}
        self.preproc = preproc
        self.smpl_p = smplh_path if not debug else kwargs['smplh_path_dbg']
        self.rot_repr = rot_repr
        self.Dataset = BodilexDataset
        # calculate splits
        if self.debug:
            # takes <2sec to load
            ds_db_path = Path(self.debug_datapath)
            
        else:
            # takes ~4min to load
            ds_db_path = Path(self.datapath)
        # define splits
        # For example
        # from itertools import islice
        # def chunks(data, SIZE=10000):
        # it = iter(data)
        # for i in range(0, len(data), SIZE):
        #     yield {k:data[k] for k in islice(it, SIZE)}
        # and then process with the AmassDataset as you like
        # pass this or split for dataloading into sets
        dataset_dict_raw = joblib.load(ds_db_path)
        for k, v in dataset_dict_raw.items():
            
            if len(v['motion_source']['rots'].shape) > 2:
                rots_flat_src = v['motion_source']['rots'].flatten(-2).float()
                dataset_dict_raw[k]['motion_source']['rots'] = rots_flat_src
            if len(v['motion_target']['rots'].shape) > 2:
                rots_flat_tgt = v['motion_target']['rots'].flatten(-2).float()
                dataset_dict_raw[k]['motion_target']['rots'] = rots_flat_tgt
        
        subkeys = list(dataset_dict_raw.keys())[:20]
        dataset_dict_raw = { k: v for k, v in dataset_dict_raw.items()
                             if k in subkeys }
        for k in dataset_dict_raw.keys():
            dataset_dict_raw[k]['motion_source']['rots'] = dataset_dict_raw[k]['motion_source']['rots'][:60]
            dataset_dict_raw[k]['motion_source']['trans'] = dataset_dict_raw[k]['motion_source']['trans'][:60] 
            dataset_dict_raw[k]['motion_source']['joint_positions'] = dataset_dict_raw[k]['motion_source']['joint_positions'][:60] 

            dataset_dict_raw[k]['motion_target']['rots'] = dataset_dict_raw[k]['motion_target']['rots'][:60]
            dataset_dict_raw[k]['motion_target']['trans'] = dataset_dict_raw[k]['motion_target']['trans'][:60] 
            dataset_dict_raw[k]['motion_target']['joint_positions'] = dataset_dict_raw[k]['motion_target']['joint_positions'][:60] 

        data_dict = cast_dict_to_tensors(dataset_dict_raw)

        # add id fiels in order to turn the dict into a list without loosing it
        random.seed(self.preproc.split_seed)
        data_ids = list(data_dict.keys())
        data_ids.sort()
        random.shuffle(data_ids)
        if self.debug:
            # 70-10-20% train-val-test for each sequence
            num_train = int(len(data_ids) * 0.8)
            num_val = int(len(data_ids) * 0.1)
        else:
            # 70-10-20% train-val-test for each sequence
            num_train = int(len(data_ids) * 0.7)
            num_val = int(len(data_ids) * 0.1)
        # give ids to data sets--> 0:train, 1:val, 2:test

        split = np.zeros(len(data_ids))
        split[num_train:num_train + num_val] = 1
        split[num_train + num_val:] = 2
        id_split_dict = {id: split[i] for i, id in enumerate(data_ids)}
        random.random()  # restore randomness in life (maybe randomness is life)
        # calculate feature statistics
        for k, v in data_dict.items():
            v['id'] = k
            v['split'] = id_split_dict[k]
        self.stats = self.calculate_feature_stats(BodilexDataset([v for k, v in data_dict.items()
                                                       if id_split_dict[k] <= 1],
                                                      self.preproc.n_body_joints,
                                                      self.preproc.stats_file,
                                                      self.preproc.norm_type,
                                                      self.smpl_p,
                                                      self.rot_repr,
                                                      self.load_feats,
                                                      do_augmentations=False))
        # import ipdb; ipdb.set_trace()

        # setup collate function meta parameters
        # self.collate_fn = lambda b: collate_batch(b, self.cfg.load_feats)
        # create datasets
        self.dataset['train'], self.dataset['val'], self.dataset['test'] = (
           BodilexDataset([v for k, v in data_dict.items() if id_split_dict[k] == 0],
                        self.preproc.n_body_joints,
                        self.preproc.stats_file,
                        self.preproc.norm_type,
                        self.smpl_p,
                        self.rot_repr,
                        self.load_feats,
                        do_augmentations=True), 
           BodilexDataset([v for k, v in data_dict.items() if id_split_dict[k] == 1],
                        self.preproc.n_body_joints,
                        self.preproc.stats_file,
                        self.preproc.norm_type,
                        self.smpl_p,
                        self.rot_repr,
                        self.load_feats,
                        do_augmentations=True), 
           BodilexDataset([v for k, v in data_dict.items() if id_split_dict[k] == 2],
                        self.preproc.n_body_joints,
                        self.preproc.stats_file,
                        self.preproc.norm_type,
                        self.smpl_p,
                        self.rot_repr,
                        self.load_feats,
                        do_augmentations=False) 
        )
        for splt in ['train', 'val', 'test']:
            log.info("Set up {} set with {} items."\
                     .format(splt, len(self.dataset[splt])))
        self.nfeats = self.dataset['train'].nfeats

    # def setup(self, stage):
    #     pass

    def calculate_feature_stats(self, dataset: BodilexDataset):
        stat_path = self.preproc.stats_file
        if self.debug:
            stat_path = stat_path.replace('.npy', '_debug.npy')

        if not exists(stat_path):
            log.info(f"No dataset stats found. Calculating and saving to {stat_path}")
            
            feature_names = dataset.get_all_features(0).keys()
            feature_dict = {name.replace('_source', ''): [] for name in feature_names
                            if '_target' not in name}
            for i in tqdm(range(len(dataset))):
                x = dataset.get_all_features(i)
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


def _pad_n(n):
    """get padding function for padding x at the first dimension n times"""
    return lambda x: pad(x[None], (0, 0) * (len(x.shape) - 1) + (0, n), "replicate")[0]

def _apply_on_feats(t, name: str, f, feats):
    """apply function f only on features"""
    return f(t) if name in feats or name.endswith('_norm') else t
