import json
import os
from glob import glob
from re import A
from typing import Dict, List, Optional, Tuple
import logging
import joblib
from copy import deepcopy
from networkx import minimum_cut_value
import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from sinc.data.tools import smpl

from sinc.utils.file_io import read_json, write_json

from sinc.tools.easyconvert import matrix_to, axis_angle_to
from sinc.transforms import Transform
import math
from .base import BASEDataModule
from sinc.data.tools.extract_pairs import extract_frame_labels
from sinc.info.joints import smpl2gpt, gpt_side
import string

from sinc.tools.frank import text_to_bp, text_list_to_bp

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test", "all", "subset"]

def get_split(path: str, split: str, subset: Optional[str] = ''):
    assert split in SPLITS
    filepath = Path(path) / f'{split}{subset}.pth.tar'
    split_data = joblib.load(filepath)
    return split_data


def get_babel_keys(path: str):
    filepath = Path(path) / f'../babel_v2.1/id2fname/amass-path2babel.json'
    amass2babel = read_json(filepath)
    return amass2babel


def load_gpt_labels(path: str):
    if 'gpt/gpt3-labels-list.json' not in str(path):
        path = path.replace('gpt/gpt3-labels.json','gpt/gpt3-labels-list.json')
    gpt_labels = read_json(path)
    return gpt_labels


class BABELDataModule(BASEDataModule):
    def __init__(self,
                 data_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 dtype: str = '',
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         datatype=dtype)
        self.save_hyperparameters(logger=False)
        self.Dataset = BABEL
        sample_overrides = {
            "split": "train",
            "tiny": True,
            "progress_bar": False
        }

        self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        self.transforms = self._sample_set.transforms


class BABEL(Dataset):
    dataname = "BABEL"

    def __init__(self,
                 datapath: str,
                 transforms: Transform,
                 split: str = "train",
                 transforms_xyz: Optional[Transform] = None,
                 transforms_smpl: Optional[Transform] = None,
                 correspondance_path: str = None,
                 amass_path: str = None,
                 smplh_path: str = None,
                 sampler=None,
                 framerate: float = 12.5,
                 progress_bar: bool = True,
                 load_with_rot=True,
                 downsample=True,
                 tiny: bool = False,
                 walk_only: Optional[bool] = False,
                 kit_only: Optional[bool] = False,
                 synthetic: Optional[bool] = False,
                 heuristic: Optional[bool] = False,
                 proportion_synthetic: float = 0.5,
                 random_synthetic: Optional[bool] = False,
                 centered_compositions: Optional[bool] = False,
                 dtype: str = '',
                 mode: str = 'train',
                 simultaneous_max: int = 2,
                 **kwargs):
        self.simultaneous_max = simultaneous_max
        self.split = split
        self.load_with_rot = load_with_rot
        self.downsample = downsample
        self.dtype = dtype  # seg or seq or empty string for segments or sequences
        # or all of the stuff --> 'seg', 'seq', 'pairs', 'pairs_only', ''
        self.synthetic = synthetic
        self.heuristic = heuristic

        self.proportion_synthetic = proportion_synthetic
        self.random_synthetic = random_synthetic
        self.walk_only = walk_only
        self.kit_only = kit_only
        self.centered_compositions = centered_compositions
        if not self.load_with_rot:
            self.transforms_xyz = deepcopy(transforms_xyz)
            self.transforms_smpl = deepcopy(transforms_smpl)
            self.transforms = deepcopy(transforms_xyz)
        else:
            self.transforms = transforms

        if self.split not in SPLITS:
            raise ValueError(f"{self.split} is not a valid split")

        self.sampler = sampler
        super().__init__()
        if tiny:
            data_for_split = get_split(path=datapath,
                                       split=split,
                                       subset='_tiny')
            self.babel_annots = read_json(
                Path(datapath) / f'../babel_v2.1/{split}.json')
        else:
            data_for_split = get_split(path=datapath, split=split)
            self.babel_annots = read_json(
                Path(datapath) / f'../babel_v2.1/{split}.json')
        gpt_path = kwargs['gpt_path']
        # gtp_path =  Path(datapath) / f'../../../deps/gpt/gpt3-labels-list.json'
        gpt_labels_full_sent = load_gpt_labels(gpt_path)

        fname2key = get_babel_keys(path=datapath)
        motion_data = {}
        joints_data = {}
        texts_data = {}
        durations = {}
        dtypes = {}
        gpt_labels = {}
        if progress_bar:
            enumerator = enumerate(
                tqdm(data_for_split, f"Loading BABEL {split}"))
        else:
            enumerator = enumerate(data_for_split)

        if tiny:
            maxdata = 1000
        else:
            maxdata = np.inf

        datapath = Path(datapath)

        num_bad_actions = 0
        num_bad_short = 0
        valid_data_len = 0
        invalid = 0
        all_data_len = 0
        num_bad_bml = 0

        num_not_kit = 0
        # discard = read_json('/home/nathanasiou/Desktop/conditional_action_gen/teach/data/amass/amass_cleanup_seqs/BMLrub.json')
        req_dtypes = self.dtype.split('+')
        logger.info(f'The required datatypes are : {req_dtypes}')
        if self.heuristic:
            from sinc.info.joints import bp2ids
            jts_pos = joblib.load(Path(datapath) / f'../jts_{split}.pkl')
            key_pos = joblib.load(Path(datapath) / f'../keys_{split}.pkl')
            vars_bps = {x:[] for x in bp2ids.keys()}
            from sinc.info.joints import smplh_joints, smpl_bps
            bp2ids = {
                bp_name: [smplh_joints.index(j) for j in jts_names]
                for bp_name, jts_names in smpl_bps.items()
            }
            loks = []
            for k, mot_jts in jts_pos.items():
                for bp, idxs in bp2ids.items():
                    vars_bps[bp].append(mot_jts[:, idxs])
                loks.append(k)

            
        for i, sample in enumerator:
            # if sample['fname'] in discard:
            #     num_bad_bml += 1
            #     continue
            if self.kit_only and not tiny and 'KIT/KIT' not in sample['fname']:
                num_not_kit += 1
                continue

            if len(motion_data) >= maxdata:
                break
            # from temos.data.sampling import subsample
            all_data_len += 1
            # smpl_data = {x: smpl_data[x] for x in smpl_data.files}
            nframes_total = len(sample['poses'])
            last_framerate = sample['fps']
            babel_id = sample['babel_id']
            frames = np.arange(nframes_total)
            all_actions = extract_frame_labels(self.babel_annots[babel_id],
                                               fps=last_framerate,
                                               seqlen=nframes_total,
                                               max_simultaneous=self.simultaneous_max
                                               )
            # if not valid:
            #     invalid += 1
            #     continue

            possible_actions = {
                k: v
                for k, v in all_actions.items() if k in req_dtypes
            }

            # if self.dtype == 'seg': possible_actions = all_actions['seg']
            # elif self.dtype == 'separate_pairs': possible_actions = all_actions['separate_pairs']
            # elif self.dtype == 'seq': possible_actions = all_actions['seq']
            # elif self.dtype == 'spatial_pairs': possible_actions = all_actions['spatial_pairs']
            # else: raise NotImplementedError(f'The datatype {self.dtype} is not supported.')
            # GPT part
            # if i == 500: break

            for dtype, extracted_actions in possible_actions.items():

                for index, seg in enumerate(extracted_actions):
                    if isinstance(seg[-1], str) and seg[-1] == '':
                        # 1 bad label
                        continue
                    if dtype == 'separate_pairs':
                        frames = np.arange(seg[0][0], seg[-1][1])
                        duration = [(e - s + 1) for s, e in seg]
                        duration[-1] -= 1
                        if len(duration) == 2: duration.insert(1, 0)
                    if dtype == 'spatial_pairs':
                        frames = np.arange(seg[0], seg[1])
                        duration = seg[1] - seg[0]
                    else:
                        frames = np.arange(seg[0], seg[1])
                        duration = len(frames)
                    smpl_data = {
                        "poses":
                        torch.from_numpy(sample['poses'][frames]).float(),
                        "trans":
                        torch.from_numpy(sample['trans'][frames]).float()
                    }
                    # pose: [T, 22, 3, 3]
                    # if split != 'test': # maybe include this (it was there originally): split != "test"
                    if not self.dtype == 'separate_pairs':
                        # Accept or not the sample, based on the duration
                        if not self.sampler.accept(duration):
                            num_bad_short += 1
                            continue
                    else:
                        dur1, dur_tr, dur2 = duration
                        # check acceptance for long sequences ... TODO
                        if not self.sampler.accept(
                                dur1) or not self.sampler.accept(dur2 +
                                                                 dur_tr):
                            # if not self.sampler.accept(dur1+dur2+dur_tr):
                            num_bad_short += 1
                            continue
                    valid_data_len += 1
                    from sinc.data.tools.smpl import smpl_data_to_matrix_and_trans
                    smpl_data = smpl_data_to_matrix_and_trans(smpl_data,
                                                              nohands=True)
                    # Load rotation features (rfeats) data from AMASS
                    if mode == 'train':
                        # if split != 'test' and split != 'val':
                        if load_with_rot:
                            features = self.transforms.rots2rfeats(smpl_data)
                        # Load xyz features (jfeats) data from AMASS
                        else:
                            joints = self.transforms_smpl.rots2joints(
                                smpl_data)
                            features = self.transforms_xyz.joints2jfeats(
                                joints)
                    else:
                        joints = smpl_data  #self.transforms.rots2joints(smpl_data)

                    keyid = f'{dtype}-{babel_id}-{index}'
                    if self.dtype == 'separate_pairs':
                        texts_data[keyid] = seg[-1]
                        durations[keyid] = duration
                        # assert features.shape[0] == sum(duration), f' \nMismatch: {babel_id}, \n {seg_ids} \n {seg} \n {frames} \n {fpair} \n {duration}--> {features.shape[0]}  {sum(duration)}'
                    else:
                        if isinstance(seg[-1], tuple):
                            texts_data[keyid] = seg[-1]
                        else:
                            texts_data[keyid] = (
                                seg[-1], )
                        durations[keyid] = duration
                    if mode == 'train':
                        motion_data[keyid] = features
                    else:
                        motion_data[keyid] = joints


                    dtypes[keyid] = dtype
                    gpt_labels[keyid] = []
                    if self.heuristic:
                        if len(texts_data[keyid]) == 1:
                            idx_of_mot = key_pos.index(keyid)
                            gpt_labels[keyid].append(self.compute_heuristic(idx_of_mot,
                                                                            vars_bps))
                    else:
                        
                        for a_t in texts_data[keyid]:
                            bp_list = text_list_to_bp(a_t, gpt_labels_full_sent)
                            # bp_list = text_to_bp(a_t, gpt_labels_full_sent)
                            gpt_labels[keyid].append(bp_list)
        if synthetic:
            # from a keyid, prepare what keyids is possible to be chosen
            from sinc.info.joints import get_compat_matrix
            self.compat_matrix = get_compat_matrix(gpt_labels_full_sent)

            from collections import defaultdict
            self.keyids_from_text = defaultdict(list)

            self.compat_seqs = {}
            for key, val in texts_data.items():
                if "seq" not in key and "seg" not in key:
                    continue
                self.keyids_from_text[val[0]].append(key)

            for key, val in texts_data.items():
                self.compat_seqs[key] = [
                    y for x in self.compat_matrix[val[0]]
                    for y in self.keyids_from_text[x]
                ]
        if split != "test" and not tiny:
            if 'spatial_pairs' in self.dtype:
                total = valid_data_len
                from collections import Counter
                multiple_action = dict(
                    Counter(map(len, [v for _, v in texts_data.items()])))
                # logger.info(f"Processed {all_data_len} sequences and found {invalid} invalid cases based on the datatype of sequences.")
                logger.info(f"{total} sequences -- datatype:{self.dtype}.")
                percentage = 100 * (num_bad_short) / (total)
                logger.info(
                    f"{percentage:.4}% of the sequence which are rejected by the sampler, because they are too short(<{self.sampler.min_len/30} secs) or too long(>{self.sampler.max_len/30} secs)."
                )
                for length, _ in multiple_action.items():
                    multiple_action[length] /= total
                    multiple_action[length] = round(multiple_action[length], 2)
                logger.info(
                    f"Counts of multiple actions for different lengths: \n{multiple_action}"
                )
            else:
                total = valid_data_len
                logger.info(
                    f"Processed {all_data_len} sequences and found {invalid} invalid cases based on the datatype."
                )
                logger.info(f"{total} sequences -- datatype:{self.dtype}.")
                percentage = 100 * (num_bad_actions + num_bad_short) / (
                    total + num_bad_short + num_bad_actions)
                logger.info(
                    f"{percentage:.4}% of the sequences which are rejected by the sampler in total."
                )
                percentage = 100 * num_bad_actions / (total + num_bad_short +
                                                      num_bad_actions)
                logger.info(
                    f"{percentage:.4}% of the sequence which are rejected by the sampler, because of the excluded actions."
                )
                percentage = 100 * num_bad_short / (total + num_bad_short +
                                                    num_bad_actions)
                logger.info(
                    f"{percentage:.4}% of the sequence which are rejected by the sampler, because they are too short(<{self.sampler.min_len/30} secs) or too long(>{self.sampler.max_len/30} secs)."
                )
                logger.info(f"Discard from BML: {num_bad_bml}")
                logger.info(f"Discard not KIT: {num_not_kit}")
        
        
                
                # plt.figure()
                # plt.hist(avg_vels, bins='fd', density=True)
                # plt.savefig(f'/home/nathanasiou/Desktop/scratch/velhist_{bp}.png')
        
        # annot_path_fx = Path(datapath)/f'../deps/gpt/gpt3-labels_gt_100.json'
        # from sinc.utils.text_constants import unique_texts_babel_train_val
        # xx = {v[0]:gpt_labels[k][0] for k,v in texts_data.items() if v[0] in unique_texts_babel_train_val[:100]}
        # # write_json(xx, '/home/nathanasiou/Desktop/conditional_action_gen/heur_gpt.json')
        # import ipdb; ipdb.set_trace()
        self.motion_data = motion_data
        self.texts_data = texts_data
        self._split_index = list(motion_data.keys())
        self._num_frames_in_sequence = durations
        self.keyids = list(self.motion_data.keys())
        self.single_keyids = [x for x in self.keyids if "spatial_pairs" not in x]
        self.pairs_keyids = [x for x in self.keyids if "spatial_pairs" in x]
        self.sample_dtype = dtypes
        self.gpt_labels = gpt_labels
        # from hydra.utils import get_original_cwd
        # ddict = {}
        # for k, v in texts_data.items():
        #     ddict[k] = [v[0], v[1], durations[k]]
        # write_json(ddict,
        #            f'{get_original_cwd()}/deps/inference/labels_{split}_spatial.json')
        if self.heuristic:
            del jts_pos
            del key_pos

        if split == 'test' or split == 'val' or mode == 'inference':
            # does not matter should be removed, just for code to not break
            # TODO fix this may be it is ftuile, ia m tired
            self.nfeats = 135  # len(self[0]["datastruct"].features[0])
        elif self.dtype == 'separate_pairs':
            self.nfeats = 135  # len(self[0]["features_0"][0])
        else:
            self.nfeats = 135 #self[0]["datastruct"].features.shape[-1]

    def compute_heuristic(self, idx, joints_pos):

        # mot_stats = {x:[] for x in bp2ids.keys()}
        # # from matplotlib import pyplot as plt
        # for bp, motions in vars_bps.items():
        #     avg_vels = []
        #     for mo in motions:
        #         velmo = mo[1:] - mo[:-1]
        #         avg_vel = velmo.mean(1) # aggregrate parts
        #         avg_vels += [torch.linalg.norm(avg_vel, dim=1).mean()] # power of vels per frame averaged
        #     import ipdb; ipdb.set_trace()
        #     max_vel = max(avg_vels)
        #     min_vel = min(avg_vels)
        #     mean_vel = sum(avg_vels) / len(avg_vels)
        #     mot_stats[bp] = [max_vel, min_vel, mean_vel, avg_vels]
        # for k, v in joints_pos.items():
        # aggr_mo = {x:[] for x in joints_pos.keys()}
        from sinc.info.joints import smpl_bps_ids 

        avg_vels = {}
        for bp, v in joints_pos.items():
            mo = v[idx]
            velmo = mo[1:] - mo[:-1]
            avg_vel = velmo.mean(1) # aggregrate parts
            avg_vels[bp]= torch.linalg.norm(avg_vel, dim=1).mean() # power of vels per frame averaged

        velmaxi = max(avg_vels.values())
        velmini = min(avg_vels.values())
        bp_list = [0] * len(smpl_bps_ids)
        
        avg_vels_norm = {k: (v-velmini)/(velmaxi - velmini) for k, v in avg_vels.items() }
        for bp, v in avg_vels_norm.items():
            if v > 0.65:
                bp_list[smpl_bps_ids[bp]] += 1

        return bp_list

    
    def _load_datastruct(self, keyid, frame_ix=None):
        features = self.motion_data[keyid][frame_ix]
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        return sequences

    def _load_actions(self, keyid):
        actions_all = self.action_datas[keyid]
        return actions_all

    def load_keyid(self, keyid, mode='train', proportion=None):
        proportion = proportion if proportion is not None else self.proportion_synthetic

        # force composition in loading
        force_comp = False
        if "synth" in keyid:
            keyid, keyid_p = keyid.split("synth_")[1].split("_")
            force_comp = True
        isnan =  True
        text = self._load_text(keyid)

        if mode == 'train':
            if self.sample_dtype[keyid] in ['seg', 'seq', '', 'spatial_pairs']:
                check_compat = self.synthetic and self.sample_dtype[keyid] in ['seg', 'seq'] and (self.compat_seqs[keyid] or self.random_synthetic)
                if np.random.uniform() < proportion and check_compat:
                    # GPT CASE
                    while isnan:
                        isnan = False
                        # check compatibility with synthetic
                        # randomly do synthetic data or not
                        # if there is some compatibility
                        # take a random pair
                        if not force_comp:
                            if self.random_synthetic:
                                keyid_p = np.random.choice(self.single_keyids)
                            else:
                                keyid_p = np.random.choice(self.compat_seqs[keyid])

                        text_p = self._load_text(keyid_p)
                        feats = self.motion_data[keyid]
                        feats_p = self.motion_data[keyid_p]
                        # make sure trasnform is on the correct device 
                        # self.transforms.rots2rfeats = self.transforms.rots2rfeats.to(feats.device)
                        
                        # smpl_data = self.transforms.rots2rfeats.inverse(feats)
                        # smpl_data_p = self.transforms.rots2rfeats.inverse(feats_p)
                        # import ipdb; ipdb.set_trace()
                        
                        # if smpl_data.rots.isnan().any() or smpl_data.rots.max() > 1000000 or smpl_data.rots.min() < -1000000:
                        #     # torch.save(smpl_data.rots, f'{keyid}--{keyid_p}--smpl_data_poses.pt')
                        #     # torch.save(smpl_data.trans, f'{keyid}--{keyid_p}--smpl_data_trans.pt')
                        #     isnan = True
                        # if smpl_data_p.rots.isnan().any() or smpl_data_p.rots.max() > 1000000 or smpl_data_p.rots.min() < -1000000:
                        #     # torch.save(feats_p, f'{keyid}--{keyid_p}--feats_p.pt')
                        #     # torch.save(feats, f'{keyid}--{keyid_p}--feats.pt')

                        #     # torch.save(smpl_data_p.rots, f'{keyid}--{keyid_p}--smpl_data_p_poses.pt')
                        #     # torch.save(smpl_data_p.trans, f'{keyid}--{keyid_p}--smpl_data_p_trans.pt')
                        #     isnan = True

                        if self.random_synthetic:
                            bp = list(np.random.randint(0, 2, 6))
                            bp_p = list(np.random.randint(0, 2, 6))
                        else:
                            bp = self.gpt_labels[keyid][0]
                            bp_p = self.gpt_labels[keyid_p][0]

                        # from sinc.tools.frank import combine_motions
                        # smpl_comb = combine_motions(smpl_data, smpl_data_p, 
                        #                             bp, bp_p, 
                        #                             center=self.centered_compositions)
                        # if smpl_comb.rots.isnan().any() or smpl_comb.rots.max() > 1000000 or smpl_comb.rots.min() < -1000000:
                        #     isnan = True
                            # torch.save(smpl_comb.rots, f'{keyid}--{keyid_p}--smpl_comb.pt')

                        # feats_comb = self.transforms.rots2rfeats(smpl_comb)

                        # if feats_comb.isnan().any() or feats_comb.max() > 1000000 or feats_comb.min() < -1000000:
                        #     isnan = True
                            # torch.save(feats_comb, f'{keyid}--{keyid_p}--feats_comb.pt')

                        # if isnan:
                        #     continue
                        # num_frames = len(feats_comb)
                        # frame_ix = self.sampler(num_frames)
                        # features = feats_comb[frame_ix]
                        # datastruct = self.transforms.Datastruct(features=features)
                        # bp = [0,1,1,0,0,0]
                        # bp_p = [0,1,1,0,0,0]
                        num_frames = self._num_frames_in_sequence[keyid]
                        frame_ix = self.sampler(num_frames)
                        datastruct = self._load_datastruct(keyid, frame_ix)

                        element = {'datastruct': self.motion_data[keyid][frame_ix],
                                   'text': (text[0], text_p[0]),
                                   'length': len(datastruct),
                                   'keyid': f"synth_{keyid}_{keyid_p}",
                                   'bp-gpt': [bp, bp_p],
                                   'datastruct_a': feats,
                                   'datastruct_b': feats_p,
                                   'frames_ix': frame_ix,
                                   }
                else:
                    num_frames = self._num_frames_in_sequence[keyid]
                    frame_ix = self.sampler(num_frames)
                    datastruct = self._load_datastruct(keyid, frame_ix)                    
                    element = {
                        'datastruct': self.motion_data[keyid][frame_ix],
                        'text': text,
                        'length': len(datastruct),
                        'keyid': keyid,
                        'bp-gpt': self.gpt_labels[keyid],
                        'datastruct_a': self.motion_data[keyid][frame_ix],
                        'datastruct_b': self.motion_data[keyid][frame_ix],
                        'frames_ix': frame_ix,
                    }
            else:
                pass

        else:  # split or val

            if self.sample_dtype[keyid] in ['seg', 'seq', '', 'spatial_pairs']:
                num_frames = self._num_frames_in_sequence[keyid]
                frame_ix = self.sampler(num_frames)
                # datastruct = self._load_datastruct(keyid, frame_ix)
                element = {
                    'datastruct': self.motion_data[keyid],
                    'text': text,
                    'length': self._num_frames_in_sequence[keyid],
                    'keyid': keyid,
                    'bp-gpt': self.gpt_labels[keyid]
                }
            else:
                pass
        return element

    def load_seqid(self, seqid):

        segs_keyids = [
            keyid for keyid in self._split_index
            if keyid.split('-')[0] == seqid
        ]
        segs_keyids = sorted([(e.split('-')[0], int(e.split('-')[1]))
                              for e in segs_keyids],
                             key=lambda x: x[1])
        segs_keyids = ['-'.join([seq, str(id)]) for seq, id in segs_keyids]
        keyids_to_return = []
        current = segs_keyids[0]
        texts = []
        lens = []
        ov = False
        if len(segs_keyids) == 1:
            t0, t1 = self._load_text(current)
            l0, lt, l1 = self._num_frames_in_sequence[current]
            lens = [l0, l1 + lt]
            texts = [t0, t1]
        else:
            while True:
                t0, t1 = self._load_text(current)
                l0, lt, l1 = self._num_frames_in_sequence[current]
                if not ov:
                    texts.append(t0)
                    texts.append(t1)
                    l1t = lt + l1
                    lens.append(l0)
                    lens.append(l1t)
                else:
                    texts.append(t1)
                    l1t = lt + l1
                    lens.append(l1t)
                if current == segs_keyids[-1]:
                    break
                candidate_next = [
                    i for i in segs_keyids[(segs_keyids.index(current) + 1):]
                    if self._load_text(i)[0] == t1
                ]

                if candidate_next:
                    ov = True
                    max_id = np.argmax(
                        np.array([
                            self._num_frames_in_sequence[cn][1]
                            for cn in candidate_next
                        ]))
                    next_seg = candidate_next[max_id]
                    current = next_seg
                else:
                    ov = False
                    if current != segs_keyids[-1]:
                        current = segs_keyids[segs_keyids.index(current) + 1]
                    else:
                        continue
        # breakpoint()
        # to_del = [idx for idx, item in enumerate(texts) if item in texts[:idx]]
        # texts = [e for i, e in enumerate(texts) if i not in to_del]
        # texts = [e for i, e in enumerate(texts) if i not in to_del]
        element = {'length': lens, 'text': texts}
        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid, mode='train')

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"
