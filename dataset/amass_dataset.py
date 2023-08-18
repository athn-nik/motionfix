import logging
import random
from glob import glob
from os import listdir
from os.path import exists, join
from pathlib import Path

import joblib
import numpy as np
import smplx
import torch
from einops import rearrange
from pytorch3d.transforms import matrix_to_euler_angles, matrix_to_rotation_6d
from pytorch_lightning import LightningDataModule
from smplx.joint_names import JOINT_NAMES
from torch.nn.functional import pad
from torch.quantization.observer import \
    MovingAveragePerChannelMinMaxObserver as mmo
from torch.utils.data import DataLoader, Dataset
from utils.masking import LengthMask
from utils.misc import DotDict, cast_dict_to_tensors, to_tensor
from utils.transformations import (
    change_for, local_to_global_orient, transform_body_pose, remove_z_rot,
    rot_diff, get_z_rot)

from dataset.sequence_parser_amass import SequenceParserAmass

# A logger for this file
log = logging.getLogger(__name__)


class AmassDataset(Dataset):
    def __init__(self, data: list, cfg, do_augmentations=False):
        self.data = data
        self.cfg = cfg
        self.do_augmentations = do_augmentations
        self.seq_parser = SequenceParserAmass(self.cfg)
        bm = smplx.create(model_path=cfg.smplx_models_path, model_type='smplx')
        self.body_chain = bm.parents
        # stat_path = join(self.cfg.project_dir, f"statistics_{self.cfg.dataset}.npy")
        suffix = "_debug" if cfg.debug else ""
        stat_path = join(cfg.project_dir, f"statistics_{cfg.dataset}{suffix}.npy")
        self.stats = None
        self.joint_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        if exists(stat_path):
            stats = np.load(stat_path, allow_pickle=True)[()]
            self.stats = cast_dict_to_tensors(stats)

        # declare functions that implement features here
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

        # declare functions that return metadata here 
        self._meta_data_get_methods = {
            "n_frames_orig": self._get_num_frames,
            "chunk_start": self._get_chunk_start,
            "framerate": self._get_framerate,
        }
        
    def normalize_feats(self, feats, feats_name):
        if feats_name not in self.stats.keys():
            log.error(f"Tried to normalise {feats_name} but did not found stats \
                      for this feature. Try running calculate_statistics.py again.")
        if self.cfg.norm_type == "std":
            mean, std = (self.stats[feats_name]['mean'].to(feats.device),
                         self.stats[feats_name]['std'].to(feats.device))
            return (feats - mean) / (std + 1e-5)
        elif self.cfg.norm_type == "norm":
            max, min = (self.stats[feats_name]['max'].to(feats.device),
                        self.stats[feats_name]['min'].to(feats.device))
            return (feats - min) / (max - min + 1e-5)

    def _get_body_joints(self, data):
        joints = to_tensor(data['joint_positions'][:, :self.cfg.n_body_joints, :])
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
        pose = transform_body_pose(pose, "aa->rot")
        rot_diffs = torch.einsum('...ik,...jk->...ij', pose, pose.roll(1, 0))
        rot_diffs[0] = torch.eye(3).to(rot_diffs.device)  # suppose zero angular vel at first frame
        return rearrange(matrix_to_rotation_6d(rot_diffs), '... j c -> ... (j c)')

    def _get_wrists_angular_velocity(self, data):
        pose = to_tensor(data['rots'][..., 3:3 + 3*21])  # drop pelvis orientation
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
        joints = to_tensor(data['joint_positions'][:, :self.cfg.n_body_joints, :])
        joint_vel = joints - joints.roll(1, 0)  # shift one right and subtract
        joint_vel[0] = 0
        return rearrange(joint_vel, '... j c -> ... (j c)')

    def _get_body_joints_rel(self, data):
        """get body joint coordinates relative to the pelvis"""
        joints = to_tensor(data['joint_positions'][:, :self.cfg.n_body_joints, :])
        pelvis_transl = to_tensor(joints[:, 0, :])
        joints_glob = to_tensor(joints[:, :self.cfg.n_body_joints, :])
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient = transform_body_pose(pelvis_orient, "aa->rot").float()
        # relative_joints = R.T @ (p_global - pelvis_translation)
        rel_joints = torch.einsum('fdi,fjd->fji',
                                  pelvis_orient, joints_glob - pelvis_transl[:, None, :])
        return rearrange(rel_joints, '... j c -> ... (j c)')

    @staticmethod
    def _get_framerate(data):
        """get framerate"""
        return torch.tensor([data['fps']])

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
        return to_tensor(data['trans'])[..., 2]
        # body.translation is NOT the same as the pelvis translation
        # TODO: figure out why
        # return to_tensor(data.body.params.transl)

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
        pelvis_orient =transform_body_pose(to_tensor(data['rots'][..., :3]), "aa->rot")
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
        pelvis_orient =to_tensor(data['rots'][..., :3])
        R_z = get_z_rot(pelvis_orient, in_format="aa")
        # rotate -R_z
        trans_vel_pelv = change_for(trans_vel, R_z.roll(1, 0), forward=True)
        trans_vel_pelv[0] = 0  # zero out velocity of first frame
        return trans_vel_pelv

    def _get_body_orient(self, data):
        """get body global orientation"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        if self.cfg.rot_repr == "6d":
            # axis-angle to rotation matrix & drop last row
            pelvis_orient = transform_body_pose(pelvis_orient, "aa->6d")
        return pelvis_orient

    def _get_body_orient_xy(self, data):
        """get body global orientation"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        if self.cfg.rot_repr == "6d":
            # axis-angle to rotation matrix & drop last row
            pelvis_orient_xy = remove_z_rot(pelvis_orient, in_format="aa")
        return pelvis_orient_xy

    def _get_body_orient_delta(self, data):
        """get global body orientation delta"""
        # default is axis-angle representation
        pelvis_orient = to_tensor(data['rots'][..., :3])
        pelvis_orient_delta = rot_diff(pelvis_orient, in_format="aa", out_format=self.cfg.rot_repr)
        return pelvis_orient_delta

    def _get_body_pose(self, data):
        """get body pose"""
        # default is axis-angle representation: Frames x (Jx3) (J=21)
        pose = to_tensor(data['rots'][..., 3:3 + 21*3])  # drop pelvis orientation
        pose = transform_body_pose(pose, f"aa->{self.cfg.rot_repr}")
        return pose

    def _get_body_pose_delta(self, data):
        """get body pose rotational deltas"""
        # default is axis-angle representation: Frames x (Jx3) (J=21)
        pose = to_tensor(data['rots'][..., 3:3 + 21*3])  # drop pelvis orientation
        pose_diffs = rot_diff(pose, in_format="aa", out_format=self.cfg.rot_repr)
        return pose_diffs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # parse data: apply augmentations, if any, and add metadata fields
        datum = self.seq_parser.parse_datum(self.data[idx])
        # perform augmentations except when in test mode
        if self.do_augmentations:
            datum = self.seq_parser.augment_npz(datum)

        # compute all features declared in config
        data_dict = {feat: self._feat_get_methods[feat](datum)
                     for feat in self.cfg.load_feats}

        # compute their normalised versions as "feat_norm"
        if self.stats is not None:
            norm_feats = {f"{feat}_norm": self.normalize_feats(data, feat)
                        for feat, data in data_dict.items()
                        if feat in self.stats.keys()}
            # append normalized features to data_dict
            data_dict = {**data_dict, **norm_feats}
        # add some meta-data
        meta_data_dict = {feat: method(datum)
                          for feat, method in self._meta_data_get_methods.items()}
        data_dict = {**data_dict, **meta_data_dict}
        data_dict['filename'] = datum['fname']
        data_dict['split'] = datum['split']
        data_dict['id'] = datum['id']
        return DotDict(data_dict)

    def npz2feats(self, idx, npz):
        """turn npz data to a proper features dict"""
        data_dict = {feat: self._feat_get_methods[feat](npz)
                     for feat in self.cfg.load_feats}
        if self.stats is not None:
            norm_feats = {f"{feat}_norm": self.normalize_feats(data, feat)
                        for feat, data in data_dict.items()
                        if feat in self.stats.keys()}
            data_dict = {**data_dict, **norm_feats}
        meta_data_dict = {feat: method(npz)
                          for feat, method in self._meta_data_get_methods.items()}
        data_dict = {**data_dict, **meta_data_dict}
        data_dict['filename'] = self.file_list[idx]['filename']
        data_dict['split'] = self.file_list[idx]['split']
        return DotDict(data_dict)

    def get_all_features(self, idx):
        datum = self.data[idx]

        data_dict = {feat: self._feat_get_methods[feat](datum)
                     for feat in self._feat_get_methods.keys()}
        return DotDict(data_dict)


class AmassDataModule(LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.dataset = {}

    def setup(self, stage):
        # load data
        if self.cfg.debug:
            # takes <2sec to load
            ds_db_path = Path(self.cfg.amass_path).parent / 'TCD_handMocap/TCD_handMocap.pth.tar'
        else:
            # takes ~4min to load
            ds_db_path = Path(self.cfg.amass_path)
        data_dict = cast_dict_to_tensors(joblib.load(ds_db_path))

        # calculate splits
        random.seed(self.cfg.preproc.split_seed)
        data_ids = list(data_dict.keys())
        data_ids.sort()
        random.shuffle(data_ids)
        # 70-10-20% train-val-test for each sequence
        num_train = int(len(data_ids) * 0.7)
        num_val = int(len(data_ids) * 0.1)
        # give ids to data sets--> 0:train, 1:val, 2:test
        split = np.zeros(len(data_ids), dtype=np.long)
        split[num_train:num_train + num_val] = 1
        split[num_train + num_val:] = 2
        id_split_dict = {id: split[i] for i, id in enumerate(data_ids)}
        random.random()  # restore randomness in life (maybe randomness is life)

        # setup collate function meta parameters
        self.collate_fn = lambda b: collate_batch(b, self.cfg.load_feats)
        for k, v in data_dict.items():
            v['id'] = k
            v['split'] = id_split_dict[k]

        # create datasets
        self.dataset['train'], self.dataset['val'], self.dataset['test'] = (
           AmassDataset([v for k, v in data_dict.items() if id_split_dict[k] == 0],
                        self.cfg, do_augmentations=True), 
           AmassDataset([v for k, v in data_dict.items() if id_split_dict[k] == 1],
                        self.cfg, do_augmentations=True), 
           AmassDataset([v for k, v in data_dict.items() if id_split_dict[k] == 2],
                        self.cfg, do_augmentations=False) 
        )
        for splt in ['train', 'val', 'test']:
            log.info("Set up {} set with {} items."\
                     .format(splt, len(self.dataset[splt])))

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.cfg.batch_size,
                          shuffle=True, collate_fn=self.collate_fn,
                          num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['val'], batch_size=self.cfg.batch_size,
                          shuffle=False, collate_fn=self.collate_fn,
                          num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.cfg.batch_size,
                          shuffle=False, collate_fn=self.collate_fn,
                          num_workers=self.cfg.num_workers)


def collate_batch(batch, feats):
    batch, orig_lengths = pad_batch(batch, feats)
    batch =  {k: torch.stack([b[k] for b in batch])\
              if k in feats or k.endswith('_norm') else [b[k] for b in batch]
              for k in batch[0].keys()}
    batch['orig_lengths'] = orig_lengths
    batch['max_length'] = max(orig_lengths)
    batch['seq_mask'] = LengthMask(orig_lengths)
    batch['seq_pad_mask_adtv'] = LengthMask(orig_lengths).additive_matrix
    batch['seq_pad_mask_bool'] = ~LengthMask(orig_lengths).bool_matrix
    batch['batch_size'] = len(orig_lengths)
    return batch

def pad_batch(batch, feats):
    """
    pad feature tensors to account for different number of frames
    we do NOT zero pad to avoid wierd values in normalisation later in the model
    input:
        - batch: list of input dictionaries
        - feats: list of features to apply padding on (rest left as is)
    returns:
        - padded batch list
        - original length of sequences (could be < n_frames when subsequensing)
    """
    max_frames = max(b['n_frames_orig'].item() for b in batch)
    pad_length = torch.tensor([max_frames - b['n_frames_orig'] for b in batch])
    return (
        [{k: _apply_on_feats(v, k, _pad_n(pad_length[i]), feats)
            for k, v in b.items()}
         for i, b in enumerate(batch)],
        max_frames - pad_length
        )

def _pad_n(n):
    """get padding function for padding x at the first dimension n times"""
    return lambda x: pad(x[None], (0, 0) * (len(x.shape) - 1) + (0, n), "replicate")[0]

def _apply_on_feats(t, name: str, f, feats):
    """apply function f only on features"""
    return f(t) if name in feats or name.endswith('_norm') else t

