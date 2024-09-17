from omegaconf import DictConfig
import logging
import hydra
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np


def collect_gen_samples(motion_gen_path, normalizer, device):
    cur_samples = {}
    cur_samples_raw = {}
    # it becomes from
    # translation | root_orient | rots --> trans | rots | root_orient
    print("Collecting Generated Samples")
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


@hydra.main(config_path="configs", version_base="1.2", config_name="compute_metrics")
def _compute_metrics(cfg: DictConfig):
    return compute_metrics(cfg)

def compute_metrics(newcfg: DictConfig) -> None:
    from tmr_evaluator.motion2motion_retr import retrieval
    from pathlib import Path
    samples_folder = newcfg.folder
    metrs_batches, metrs_full = retrieval(samples_folder)
    print("\n===== Metrics for Retrieval on Batches of 32 =====")
    print(metrs_batches)
    print("\n===== Metrics for Retrieval on the full test set =====")
    print( metrs_full)

if __name__ == '__main__':
    _compute_metrics()

