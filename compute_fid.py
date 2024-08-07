import numpy as np
from tmr_evaluator.fid import calculate_fid
import os
from omegaconf import DictConfig
import logging
import hydra
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from typing import List, Dict
from torch import Tensor

from src.utils.file_io import write_json, read_json

def calculate_feat_stats(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

logger = logging.getLogger(__name__)

mat2name = {
            'sim_matrix_s_t': 'source_target',
            'sim_matrix_t_t': 'target_generated'
            }

import os
import json
from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig) -> str:
    path = os.path.join(cfg.run_dir, "config.json")
    config = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w") as f:
        string = json.dumps(config, indent=4)
        f.write(string)
    return path


def read_config(run_dir: str, return_json=False) -> DictConfig:
    path = os.path.join(run_dir, "config.json")
    with open(path, "r") as f:
        config = json.load(f)
    if return_json:
        return config
    cfg = OmegaConf.create(config)
    cfg.run_dir = run_dir
    return cfg


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

def line2dict(line):
    names_of_metrics = ["R@1_s2t", "R@2_s2t", "R@3_s2t", "R@5_s2t", "R@10_s2t", "MedR_s2t", "AvgR_s2t",
                        "R@1", "R@2", "R@3", "R@5", "R@10", "MedR", "AvgR"]
    metrics_nos = line.replace('\\', '').split('&')
    metrics_nos = [x.strip() for x in metrics_nos if x]
    return dict(zip(names_of_metrics, metrics_nos))

def lengths_to_mask_njoints(lengths: List[int], njoints: int, device: torch.device) -> Tensor:
    # joints*lenghts
    joints_lengths = [njoints*l for l in lengths]
    joints_mask = lengths_to_mask(joints_lengths, device)
    return joints_mask


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len,
                        device=device).expand(len(lengths),
                                              max_len) < lengths.unsqueeze(1)
    return mask

def collect_gen_samples(motion_gen_path, normalizer, device):
    cur_samples = {}
    cur_samples_raw = {}
    # it becomes from 
    # translation | root_orient | rots --> trans | rots | root_orient 
    logger.info("Collecting Generated Samples")
    from src.data.features import _get_body_transl_delta_pelv_infer
    import glob

    sample_files = glob.glob(f'{motion_gen_path}/*.npy')
    for fname in tqdm(sample_files):
        keyid = str(Path(fname).name).replace('.npy', '')
        gen_motion_b = np.load(fname,
                               allow_pickle=True).item()['pose']
        gen_motion_b = torch.from_numpy(gen_motion_b)
        trans = gen_motion_b[..., :3]
        global_orient_6d = gen_motion_b[..., 3:9]
        body_pose_6d = gen_motion_b[..., 9:]

        trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d,
                                                  trans)
        gen_motion_b_fixed = torch.cat([trans_delta, body_pose_6d,
                                        global_orient_6d], dim=-1)
        gen_motion_b_fixed = normalizer(gen_motion_b_fixed)
        cur_samples[keyid] = gen_motion_b_fixed.to(device)
        cur_samples_raw[keyid] = torch.cat([trans, global_orient_6d, 
                                            body_pose_6d], dim=-1).to(device)
    return cur_samples, cur_samples_raw

def compute_fid_score(model, dataset, keyids, gen_samples,
                       batch_size=256, progress=True):
    import torch
    import numpy as np
    from src.data.tools.collate import collate_text_motion
    from src.tmr.tmr import get_sim_matrix
    import numpy as np
    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}
    keyids_ordered = {}
    with torch.no_grad():

        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)
        # by batch (can be too costly on cuda device otherwise)
        cur_samples = []
        latent_motions_A = []
        latent_motions_B = []
        keys_ordered_for_run = []

        if progress:
            data_iter = tqdm(all_data_splitted, leave=False)
        else:
            data_iter = all_data_splitted
        for data in data_iter:
            # batch = collate_text_motion(data, device=device)
            from src.data.tools.collate import collate_tensor_with_padding
            cur_batch_keys = [x['keyid'] for x in data]
            for x in cur_batch_keys:
                if x not in gen_samples:
                    print(x)
                    cur_batch_keys.remove(x)
            keys_ordered_for_run.extend(cur_batch_keys)
            # TODO load the motions for the generations
            # Text is already encoded
            motion_a = collate_tensor_with_padding(
                [x['motion_target'] for x in data]).to(model.device)
            lengths_a = [len(x['motion_target']) for x in data]
            lengths_tgt = [len(x['motion_target']) for x in data]

            if gen_samples:

                cur_samples = [gen_samples[key_in_batch][:lengths_tgt[ix]] for ix, key_in_batch in enumerate(cur_batch_keys)]
                lengths_b = [len(x) for x in cur_samples]
                motion_b = collate_tensor_with_padding(cur_samples
                                                        ).to(model.device)
            else:
                # motion_b = collate_tensor_with_padding([
                #     x['motion_target'] for x in data]).to(
                #         model.device)
                # lengths_b = [len(x['motion_target']) for x in data]
                motion_b = collate_tensor_with_padding([
                    x['motion_source'] for x in data]).to(
                        model.device)
                lengths_b = [len(x['motion_source']) for x in data]


            masks_a = length_to_mask(lengths_a, device=motion_a.device)
            masks_b = length_to_mask(lengths_b, device=motion_b.device)
            motion_a_dict = {'length': lengths_a, 'mask': masks_a,
                            'x': motion_a}
            motion_b_dict = {'length': lengths_b, 'mask': masks_b, 
                            'x': motion_b}

            # Encode both motion and text
            latent_motion_A = model.encode(motion_a_dict, 
                                        sample_mean=True)
            latent_motion_B = model.encode(motion_b_dict,
                                        sample_mean=True)
            latent_motions_A.append(latent_motion_A)
            latent_motions_B.append(latent_motion_B)
        latent_motions_A = torch.cat(latent_motions_A)
        latent_motions_B = torch.cat(latent_motions_B)

        gt_stats = calculate_feat_stats(latent_motions_A)
        gen_stats = calculate_feat_stats(latent_motions_B)
        fid_score = calculate_fid(gt_stats, gen_stats)

    return fid_score

def get_motion_distances(model, dataset, keyids, gen_samples,
                         batch_size=256, body_model=None):

    import torch
    import numpy as np
    import numpy as np
    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}
    if body_model is None:
        import smplx
        body_model = smplx.SMPLHLayer(f'datasets/body_models/smplh',
                                        model_type='smplh',
                                        gender='neutral',
                                        ext='npz').to('cuda').eval();

    with torch.no_grad():

        all_data = [dataset.load_keyid_raw(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)
        for sett in ['t_t']:
            cur_samples = []
            motions_a = []
            motions_b = []
            tot_lens_a = []
            tot_lens_b = []
            for data in tqdm(all_data_splitted, leave=False):
                # batch = collate_text_motion(data, device=device)
                from src.data.tools.collate import collate_tensor_with_padding
                # TODO load the motions for the generations
                keyids_of_cursplit = [x['keyid'] for x in data]
                # Text is already encoded
                if sett == 's_t':

                    motion_a = collate_tensor_with_padding(
                        [x['motion_source'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_source']) for x in data]
                    if gen_samples:
                        cur_samples = [gen_samples[kd] for kd in keyids_of_cursplit]
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(
                            cur_samples).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding(
                           [x['motion_target'] for x in data]).to(model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                elif sett == 't_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_b = [len(x['motion_target']) for x in data]
                    if gen_samples:
                        cur_samples = [gen_samples[kd] for kd in keyids_of_cursplit]
                        lengths_b = [len(x) for x in cur_samples]
                        if motion_a.shape[1] < cur_samples[0].shape[0]:
                            cur_samples = [cs[:motion_a.shape[1]] for cs in cur_samples]
                            motion_b = collate_tensor_with_padding(cur_samples
                                                                ).to(model.device)
                        else:
                            motion_b = collate_tensor_with_padding(cur_samples
                                                                ).to(model.device)
                            

                    else:
                        motion_b = collate_tensor_with_padding([
                            x['motion_target'] for x in data]).to(
                                model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                def split_into_chunks(N, k): 
                    chunked = [k*i for i in range(1, N//k+1)] + ([N] if N%k else [])
                    return [0] + chunked

                ids_for_smpl = split_into_chunks(motion_a.shape[0], 16)
                def sliding_window(lst):
                    return [(lst[i], lst[i+1]) for i in range(len(lst) - 1)]

                for s, e in sliding_window(ids_for_smpl):
                    motions_a.append(run_smpl_fwd(motion_a[s:e, :, :3],
                                                motion_a[s:e, :, 3:9],
                                                motion_a[s:e, :, 9:],
                                                body_model).detach().cpu())
                    motions_b.append(run_smpl_fwd(motion_b[s:e, :, :3],
                                                motion_b[s:e, :, 3:9],
                                                motion_b[s:e, :, 9:],
                                                body_model).detach().cpu())
                tot_lens_a.extend(lengths_a)
                tot_lens_b.extend(lengths_b)

            mask_a = lengths_to_mask(tot_lens_a, device).detach().cpu()
            mask_b = lengths_to_mask(tot_lens_b, device).detach().cpu()

            from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
            max_a = -5
            for x in motions_a:
                if len(x[0]) > max_a:
                    max_a = len(x[0])
            max_b = -5
            for x in motions_b:
                if len(x[0]) > max_b:
                    max_b = len(x[0])

            motions_a_proc = []
            for x in motions_a:
                if len(x[0]) != max_a:
                    zeros_to_add = torch.zeros(x.size(0),
                                               max_a - len(x[0]), 
                                               22, 3)
                    motions_a_proc.append(torch.cat((x, 
                                                     zeros_to_add), dim=1))
                else:
                    motions_a_proc.append(x)

            motions_b_proc = []
            for x in motions_b:
                if len(x[0]) != max_b:
                    zeros_to_add = torch.zeros(x.size(0),
                                               max_b - len(x[0]), 
                                               22, 3)
                    motions_b_proc.append(torch.cat((x, 
                                                     zeros_to_add), dim=1))
                else:
                    motions_b_proc.append(x)

            from einops import rearrange
            motions_a = torch.cat(motions_a_proc).detach().cpu()
            motions_b = torch.cat(motions_b_proc).detach().cpu()

        def total_average_l2_distance(tensor_a, tensor_b, mask):
            """
            Compute the total average L2 distance between tensor_a and tensor_b, excluding elements discarded by the mask.
            
            :param tensor_a: First tensor with shape [717, 150, 22, 3]
            :param tensor_b: Second tensor with shape [717, 150, 22, 3]
            :param mask: Mask tensor with shape [717, 150]
            :return: Total average L2 distance
            """
            # Compute the L2 distance for each corresponding element
            l2_distance = torch.sqrt(torch.sum((tensor_a - tensor_b) ** 2, dim=-1))
            
            # Expand the mask to match the dimensions of l2_distance
            mask_expanded = mask.unsqueeze(-1)  # Shape: [717, 150, 1, 1]
            
            # Apply the mask
            masked_l2_distance = l2_distance * mask_expanded
            
            # Sum the distances where mask is 1
            total_distance = masked_l2_distance.sum()
            
            # Count the number of valid (non-zero) elements in the mask
            valid_elements = mask.sum() * l2_distance.shape[-1]
            
            # Compute the mean distance
            mean_distance = total_distance / valid_elements
            
            return mean_distance.item()

        global_edit_accuracy = total_average_l2_distance(motions_a, motions_b, 
                                                mask_a)

        tot_gl_edacc = global_edit_accuracy

        # global_edit_accuracy = global_edit_accuracy.mean()
    return tot_gl_edacc

def run_smpl_fwd(body_transl, body_orient, body_pose, body_model,
                 verts=True):
    from src.tools.transforms3d import transform_body_pose
    
    if len(body_transl.shape) > 2:
        bs, seqlen = body_transl.shape[:2]
        body_transl = body_transl.flatten(0, 1)
        body_orient = body_orient.flatten(0, 1)
        body_pose = body_pose.flatten(0, 1)

    batch_size = body_transl.shape[0]
    body_model.batch_size = batch_size
    jts = body_model(transl=body_transl, body_pose=transform_body_pose(body_pose,
                                                            '6d->rot'),
                      global_orient=transform_body_pose(body_orient,
                                                        '6d->rot')).joints[:, 
                                                                           :22]
    return jts.reshape(bs, seqlen, -1, 3)


def retrieval(path_for_samples, body_model, eval_model) -> None:
    device = 'cuda'
    # run_dir = 'eval-deps/tmr_humanml3d_amass_feats'
    batch_size = 256
    motion_gen_path = path_for_samples
    dataset = 'bodilex' # motionfix
    sets = 'test' # val all
    # save_dir = os.path.join(run_dir, "motionfix/contrastive_metrics")
    # os.makedirs(save_dir, exist_ok=True)

    datasets = {}
    results = {}
    keyids_ord = {}
    bs_m2m = 32 # for the batch size metric
    model = eval_model
    if motion_gen_path is not None:
        # curdir = Path(hydra.utils.get_original_cwd())
        curdir = Path('/is/cluster/fast/nathanasiou/logs/tmr/tmr_humanml3d_amass_feats/')
        # calculate splits
        from src.tmr.data.motionfix_loader import Normalizer
        normalizer = Normalizer(curdir/'stats/humanml3d/amass_feats')
        gen_samples, gen_samples_raw = collect_gen_samples(motion_gen_path,
                                                           normalizer, 
                                                           model.device)
    else: 
        gen_samples = None
        gen_samples_raw = None

    if sets == 'all':
        sets_to_load = ['val', 'test']
        extra_str = '_val_test'
    elif sets == 'val':
        sets_to_load = ['val']
        extra_str = '_val'
    else:
        sets_to_load = ['test']
        extra_str = ''

    # Load the dataset if not already
    from src.tmr.data.motionfix_loader import MotionFixLoader
    # from src.data.sincsynth_loader import SincSynthLoader
    # if newcfg.dataset == 'sinc_synth':
    #     dataset = SincSynthLoader()
    # else:
    dataset = MotionFixLoader(sets=sets_to_load)
    # rms = ['002274', '002273', '002223', '002226', '002265', '002264']
    # for k in rms:
    #     dataset.motions.pop(k)
    #     dataset.keyids.remove(k)

    # TODO Load the motion editing test set
    datasets.update(
        {key: dataset for key in ["normal"]}
    )
    if gen_samples is not None:
        gen_samples = {k:v for k, v in gen_samples.items() if k in dataset.motions.keys()}
    else:
        if motion_gen_path is not None:
            exit('Expected samples to be provided in the path.')
    dataset = datasets["normal"]

    # Compute sim_matrix for each protocol

    fid_score = compute_fid_score(model, dataset, dataset.keyids, 
                                                 gen_samples=gen_samples,
                                                 batch_size=batch_size)
    edit_acc = get_motion_distances(model, dataset, dataset.keyids, 
                                                 gen_samples=gen_samples,
                                                 batch_size=batch_size,
                                                 body_model=body_model)

    print("FID:", fid_score)
    print("Edit Accuracy:", edit_acc)

def load_eval_model():
    run_dir = '/is/cluster/fast/nathanasiou/logs/tmr/tmr_humanml3d_amass_feats'
    ckpt_name = 'last'
    device = 'cuda'
    # Load last config

    cfg = read_config(run_dir)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.tmr.load_model import load_model_from_cfg
    from src.tmr.metrics import all_contrastive_metrics_mot2mot, print_latex_metrics_m2m

    pl.seed_everything(cfg.seed)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    return model

if __name__ == "__main__":
    bsl_mots_path = '/is/cluster/fast/nathanasiou/logs/kinedit/hml_3d/lr1-4_300ts_bs128/'
    tmed_mots_path = '/is/cluster/fast/nathanasiou/logs/tog-mfix/bodilex/lr1e-4_bs128/'

    path_to_tmed = 'steps_300_bodilex_noise_last'
    guid_comb_gt  = '/ld_txt-2.0_ld_mot-1.0'
    guid_comb_tmed  = '/ld_txt-2.0_ld_mot-3.0'

    paths_to_bsl = {'mdm_s': '3way_steps_300_bodilex_source_last',
                     'mdm': '3way_steps_300_bodilex_noise_last',
                     'mdm_bp_s': '3way_steps_300_bodilex_source_last_inpaint_bsl',
                     'mdm_bp': '3way_steps_300_bodilex_noise_last_inpaint_bsl',
                    #  'tmed': 'steps_300_bodilex_noise_last'
                    }
    # mots_to_compute = bsl_mots_path + paths_to_bsl['mdm'] + guid_comb_gt
    mots_to_compute_tmed = tmed_mots_path + path_to_tmed + guid_comb_tmed
    # mots_to_compute = None

    # path_to_mots = 'tog-mfix/bodilex/lr1e-4_bs128/steps_300_bodilex_noise_last'
    # guid_comb  = '/ld_txt-2.0_ld_mot-3.0'
    evaluator_model = load_eval_model()
    import smplx
    body_model = smplx.SMPLHLayer(f'data/body_models/smplh',
                                    model_type='smplh',
                                    gender='neutral',
                                    ext='npz').to('cuda').eval();

    # mots_to_compute = mots_path + '/' + path_to_mots + '/' + guid_comb
    # tog-mfix/bodilex/lr1e-4_bs128
    # kinedit/hml_3d/lr1-4_300ts_bs128
    for bsl_name, bsl_p in paths_to_bsl.items():
        print(f'-----{bsl_name}-----')

        if 'mdm_bp_s' != bsl_name:
            # if 'mdm_bp' == bsl_name:
            #     guids_p = 'ld_txt-3.0_ld_mot-2.0'
            # else:
                guids_p = guid_comb_gt
        else:
            guids_p = '/ld_txt-3.0_ld_mot-1.0' 
            continue
        path_to_mots_bsl = bsl_mots_path + bsl_p + guids_p
        
        # retrieval(path_to_mots_bsl, body_model, evaluator_model)

    print('-----TMED-----')
    retrieval(mots_to_compute_tmed, body_model, evaluator_model)

