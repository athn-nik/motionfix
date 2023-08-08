import logging
from statistics import mode
import yaml
import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import sinc.launch.prepare  # noqa
from tqdm import tqdm
import torch

from sinc.utils.eval_utils import sanitize, regroup_metrics
from sinc.utils.file_io import get_samples_folder, save_metric, get_metric_paths

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="eval")
def _eval(cfg: DictConfig):
    return eval(cfg)

def eval(cfg: DictConfig) -> None:

    logger.info(f"Evaluation script.")
    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(cfg.folder))
    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # Overload it
    cfg = OmegaConf.merge(prevcfg, cfg)
    from sinc.utils.inference import cfg_mean_nsamples_resolution, get_path
    bak_save_path = Path(output_dir) / 'metrics'
    bak_save_path.mkdir(exist_ok=True, parents=True)
    
    onesample = cfg_mean_nsamples_resolution(cfg)
    model_samples, jointstype = get_samples_folder(cfg.folder,
                                                   cfg.ckpt_name,
                                                   jointstype=cfg.jointstype)
    split = cfg.split

    path = get_path(model_samples, cfg.split, onesample, cfg.mean, cfg.fact)
    if cfg.naive in ['gpt', 'concat']:
        path = Path(f'{str(path)}_naive_{cfg.naive}_pairs')
    else:
        path = Path(f'{str(path)}_pairs')

    save_paths = get_metric_paths(model_samples, cfg.set,
                                  cfg.split, onesample, cfg.mean, cfg.fact) 
     
    if onesample:
        save_path = save_paths
        if cfg.naive is not None:
            assert cfg.naive in ['gpt', 'concat']
            save_path = save_path.parent / (save_path.name + f"_{cfg.naive}")
        logger.info(f"The outputs will be stored in: {save_path}")
    
    else:
        # TODO: update this branch
        avg_path, best_path = save_paths
        logger.info(f"The outputs will be stored in: {avg_path} and {best_path}")
    if cfg.set =='small':
        bak_save_path = bak_save_path / ('JointsBased_' + save_path.name + '_' + str(cfg.ckpt_name) +'_small')
        save_path = Path(f'{save_path}_small')
    else:
        bak_save_path = bak_save_path / ('JointsBased_' + save_path.name + '_' + str(cfg.ckpt_name))

    
    logger.info("Loading the libraries")
    import numpy as np
    import torch
    import json
    from hydra.utils import instantiate
    from space.model.metrics import ComputeMetricsBest, ComputeMetricsSpace
    logger.info("Libraries loaded")

    rots2joints = instantiate(cfg.rots2joints, jointstype=jointstype)

    # If mmmns, it is smpl scale, so it is already in meters
    force_in_meter = cfg.jointstype != "mmmns"
    if onesample:
        CMetrics = ComputeMetricsSpace(force_in_meter=force_in_meter)
    else:
        CMetrics_best = ComputeMetricsBest(force_in_meter=force_in_meter)
        CMetrics_avg = [ComputeMetricsSpace(force_in_meter=force_in_meter) for index in range(cfg.number_of_samples)]

    logger.info(f"Computing the {split} metrics")
    # keep infos for computing

    logger.info("Loading data module")

    cfg.data.dtype = 'spatial_pairs+seg+seq'
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")
    dataset = getattr(data_module, f"{cfg.split}_dataset")
    eval_pairs = cfg.set == 'pairs'
    if cfg.set == 'submission':
        from sinc.utils.inference import sinc_eval_set
        keyids = sinc_eval_set
    elif cfg.set == 'small':
        from sinc.utils.inference import validation_nostand_notrain
        keyids = validation_nostand_notrain
    elif cfg.set == 'supmat':
        from sinc.utils.inference import sinc_supmat
        keyids = sinc_supmat
    else:
        if cfg.set == 'pairs':
            keyids = [k for k in dataset.keyids if k.split('-')[0] == 'spatial_pairs']
        elif cfg.set == 'single':
            keyids = [k for k in dataset.keyids if k.split('-')[0] in ['seq', 'seg']]
        else:
            keyids = dataset.keyids
    with torch.no_grad():
 
        for keyid in tqdm(keyids):

           # if (keyid.split('-')[0] == 'spatial_pairs' and eval_pairs) or not eval_pairs:

                datapoint = dataset.load_keyid(keyid, mode='inference')
                if len(datapoint['text']) > 2:
                    continue
                    
                ref_datastruct = datapoint['datastruct']
                ref_joints = rots2joints(ref_datastruct)

                if not onesample:
                    model_joints_all = []
                    ref_joints_all = []
                    length_all = []
                for index in range(cfg.number_of_samples):
                    # Load model joints
                    seq_id = "" if onesample else f"_{index}"
                    try:
                        model_joints = np.load(path / f"{keyid}{seq_id}.npy",
                                            allow_pickle=True).item()['motion']
                    except:
                        print( f"{keyid}{seq_id}.npy not found")                
                        continue
                    model_joints = torch.from_numpy(model_joints).float()
                    # Take the common lengths to facilitate the computation
                    length = min(len(model_joints), len(ref_joints))
                    if onesample:
                        # Compute part of the metrics
                        CMetrics.update(model_joints[None], ref_joints[None], [length])
                    else:
                        CMetrics_avg[index].update(model_joints[None], ref_joints[None], [length])
                        # keep them all to compute the best one
                        model_joints_all.append(model_joints[None])
                        ref_joints_all.append(ref_joints[None])
                        length_all.append([length])

                if not onesample:
                    CMetrics_best.update(model_joints_all, ref_joints_all, length_all)
    if onesample:
        metrics = sanitize(regroup_metrics(CMetrics.compute(mode='test')))
        logger.info(f"All done, saving at {save_path}")
        save_metric(save_path, metrics)
        metrics['samples-path'] = str(path)
        save_metric(bak_save_path, metrics)
        logger.info(f"Saved metrics in {str(bak_save_path)}")
        logger.info(f"Samples loaded from path: {str(path)}")
        #for key in ["APE_root", "AVE_root"]:
        #    logger.info(f"{key}: {metrics[key]}")
    else:
        # TODO: update
        # best metrics
        best_metrics = sanitize(regroup_metrics(CMetrics_best.compute(mode='test')))

        avgs = []
        for index in range(cfg.number_of_samples):
            avgs.append(regroup_metrics(CMetrics_avg[index].compute(mode='test')))

        # avg metrics
        avg_metrics = sanitize({key: np.mean([avg[key] for avg in avgs]) for key in avgs[0].keys()})

        logger.info(f"All done, saving at {best_path} and {avg_path}")
        save_metric(avg_path, avg_metrics)
        save_metric(best_path, best_metrics)
        logger.info("Done.")

        for name, metrics in [("avg", avg_metrics), ("best", best_metrics)]:
            logger.info(f"{name}")
            for key in ["APE_root", "AVE_root"]:
                logger.info(f"  {key}: {metrics[key]}")

if __name__ == '__main__':
    _eval()
