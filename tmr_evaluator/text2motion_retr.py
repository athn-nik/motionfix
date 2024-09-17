import os
from omegaconf import DictConfig
import logging
import hydra
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger(__name__)

mat2name = {
            'sim_matrix_s_t': 'source_target',
            'sim_matrix_t_t': 'target_generated'
            }
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
from typing import List, Dict
import torch
from torch import Tensor

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
    # it becomes from 
    # translation | root_orient | rots --> trans | rots | root_orient 
    logger.info("Collecting Generated Samples")
    from prepare.compute_amass import _get_body_transl_delta_pelv
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
        trans_delta = _get_body_transl_delta_pelv(global_orient_6d,
                                                  trans)
        gen_motion_b_fixed = torch.cat([trans_delta, body_pose_6d,
                                        global_orient_6d], dim=-1)
        gen_motion_b_fixed = normalizer(gen_motion_b_fixed)
        cur_samples[keyid] = gen_motion_b_fixed.to(device) 
    return cur_samples

def compute_sim_matrix(model, dataset, keyids, gen_samples,
                       batch_size=256):
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
    with torch.inference_mode():
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        all_data_splitted = np.array_split(all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)
        latent_texts = []
        latent_motions = []
        sent_embs = []
        for data in tqdm(all_data_splitted, leave=False):
            from src.data.tools.collate import collate_tensor_with_padding
            cur_batch_keys = [x['keyid'] for x in data]
            batch = collate_text_motion(data, device=device)
            # Text is already encoded
            text_x_dict = batch["text_x_dict"]
            sent_emb = batch["sent_emb"]

            if gen_samples is not None:
                cur_samples = [gen_samples[key_in_batch] for key_in_batch in cur_batch_keys]
                lengths = [len(x) for x in cur_samples]
                motions = collate_tensor_with_padding(
                    cur_samples).to(model.device)
                masks = length_to_mask(lengths, device=motions.device)
                motion_x_dict = {'length': lengths, 'mask': masks, 'x': motions}
            else:
                motion_x_dict = batch["motion_x_dict"]

            # Encode both motion and text
            latent_text = model.encode(text_x_dict, sample_mean=True)
            latent_motion = model.encode(motion_x_dict, sample_mean=True)

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)

        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)
        sent_embs = torch.cat(sent_embs)
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
    returned = {
        "sim_matrix": sim_matrix.cpu().numpy(),
        "sent_emb": sent_embs.cpu().numpy(),
    }
    return returned

def shorten_metric_line(line_to_shorten):
    # Split the string into a list of numbers
    numbers = line_to_shorten.split('&')

    # Remove the elements at the 4th, 5th, 6th, 11th, 12th, and 13th indices
    indices_to_remove = [4, 5, 6, 11, 12, 13]
    for index in sorted(indices_to_remove, reverse=True):
        del numbers[index]

    # Join the list back into a string
    return '&'.join(numbers)

def retrieval(path_for_samples) -> None:
    protocol = ['normal', 'batches']
    device = 'cuda'
    run_dir = 'eval-deps/tmr_humanml3d_amass_feats'
    ckpt_name = 'last'
    batch_size = 256
 
    motion_gen_path = path_for_samples
    protocols = protocol
    dataset = 'motionfix' # motionfix
    sets = 'test' # val all
    motion_gen_path = newcfg.samples_path

    protocols = protocol
    # Load last config
    from src.tmr.load_model import read_config
    cfg = read_config(run_dir)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.tmr.load_model import load_model_from_cfg
    from src.tmr.metrics import all_contrastive_metrics_mot2mot, print_latex_metrics_m2m

    pl.seed_everything(cfg.seed)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    datasets = {}
    results = {}
    bs_m2m = 32 # for the batch size metric
    if motion_gen_path is not None:
        curdir = Path(hydra.utils.get_original_cwd())
        # calculate splits
        from src.tmr.data.motionfix_loader import Normalizer
        normalizer = Normalizer(curdir/'stats/humanml3d/amass_feats')
        gen_samples = collect_gen_samples(motion_gen_path,
                                          normalizer, 
                                          model.device)
    else:
        gen_samples = None

    for protocol in protocols:
        logger.info(f"|------Protocol {protocol.upper()}-----|")
        # Load the dataset if not already
        if protocol not in datasets:
            dataset = instantiate(cfg.data, split="test")
            # TODO Load the motion editing test set
            datasets.update(
                {key: dataset for key in ["normal", "guo"]}
            )
        if gen_samples is not None:
            gen_samples = {k:v for k, v in gen_samples.items() if k in dataset.keyids}
        dataset = datasets[protocol]
        
        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol=="normal":
                res = compute_sim_matrix(
                    model, dataset, dataset.keyids, 
                    gen_samples=gen_samples,
                    batch_size=batch_size,
                )
                results.update({key: res for key in ["normal"]})
                # dists = get_motion_distances(
                #     model, dataset, dataset.keyids, 
                #     motion_gen_path=motion_gen_path,
                #     batch_size=batch_size,
                # )

            elif protocol == "guo":
                keyids = sorted(dataset.keyids)
                N = len(keyids)

                # make batches of 32
                idx = np.arange(N)
                np.random.seed(0)
                np.random.shuffle(idx)
                idx_batches = [
                    idx[bs_m2m * i : bs_m2m * (i + 1)] for i in range(len(keyids) // bs_m2m)
                ]

                # split into batches of 32
                # batched_keyids = [ [32], [32], [...]]
                results["guo"] = [
                    compute_sim_matrix(
                        model,
                        dataset,
                        np.array(keyids)[idx_batch],
                        gen_samples=gen_samples,
                        batch_size=batch_size,
                    )
                    for idx_batch in idx_batches
                ]
                # dists = get_motion_distances(
                #     model, dataset, dataset.keyids, 
                #     motion_gen_path=motion_gen_path,
                #     batch_size=batch_size,
                # )
                
        result = results[protocol]
        from src.tmr.metrics import all_contrastive_metrics_text2mot, print_latex_metrics_t2m

        # Compute the metrics
        if protocol == "guo":
            all_metrics = []
            for x in result:
                sim_matrix = x["sim_matrix"]
                metrics = all_contrastive_metrics_text2mot(sim_matrix, rounding=None)
                all_metrics.append(metrics)

            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = round(
                    float(np.mean([metrics[key] for metrics in all_metrics])), 2
                )

            metrics = avg_metrics
            protocol_name = protocol

        else:
            sim_matrix = result["sim_matrix"]

            protocol_name = protocol
            emb, threshold = None, None
            metrics = all_contrastive_metrics_text2mot(sim_matrix, emb, threshold=threshold)
        #import ipdb;ipdb.set_trace()
        if protocol == 'normal':
            line_for_all = print_latex_metrics_t2m(metrics)
        else:
            line_for_guo = print_latex_metrics_t2m(metrics)
        print_latex_metrics_t2m(metrics)
        print_latex_metrics_t2m(metrics, short=True)
        # TODO do this at some point!
            # run = wandb.init()
            # my_table = wandb.Table(columns=["a", "b"],
            #                        data=[["1a", "1b"], ["2a", "2b"]])
            # run.log({"table_key": my_table})
        if newcfg.samples_path is not None:
            short_expname = newcfg.samples_path.replace('/is/cluster/fast/nathanasiou/logs/motionfix-sigg/', '')
        else:
            short_expname = 'GroundTruth Results'

        logger.info(f"Testing done")
        logger.info(f"-----------")
        
    print(f'----Experiment Folder----\n\n{short_expname}')
    
    print('---------Full Metric-------')
    print(f'----Batches of {bs_m2m}----\n\n{line_for_guo}')
    print(f'----Full Set----\n\n{line_for_all}')

    print('---------Short Metric-------')
    print(f'----Batches of {bs_m2m}----\n\n{shorten_metric_line(line_for_guo)}')
    print(f'----Full Set----\n\n{shorten_metric_line(line_for_all)}')

if __name__ == "__main__":
    retrieval()
