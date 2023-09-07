import hydra
import logging

import torch
from omegaconf import DictConfig
import logging
import random
from os.path import exists, join
from pathlib import Path

import joblib
import numpy as np
import torch
from utils.misc import cast_dict_to_tensors
from pdb import set_trace
from tqdm import tqdm

from dataset.amass_dataset import AmassDataset


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="stats.yaml")
def main(cfg : DictConfig) -> None:
    if cfg.debug:
        # takes <2sec to load
        log.info("loading small dataset")
        ds_db_path = Path(cfg.amass_path).parent / 'TCD_handMocap/TCD_handMocap.pth.tar'
    else:
        # takes ~4min to load
        log.info("loading full dataset")
        ds_db_path = Path(cfg.amass_path)
    data_dict = cast_dict_to_tensors(joblib.load(ds_db_path))
    dataset = AmassDataset([v for k, v in data_dict.items()], cfg,
                           do_augmentations=False)
    suffix = "_debug" if cfg.debug else ""
    stat_path = join(cfg.project_dir, f"statistics_{cfg.dataset}{suffix}.npy")
    # if not exists(stat_path) or cfg.dl.force_recalculate_stats:
    if not exists(stat_path):
        log.info(f"No dataset stats found. Calculating and saving to {stat_path}")
    elif cfg.dl.force_recalculate_stats:
        log.info(f"Dataset stats will be re-calculated and saved to {stat_path}")
    feature_names = dataset._feat_get_methods.keys()
    feature_dict = {name: [] for name in feature_names}
    for i in tqdm(range(len(dataset))):
        x = dataset.get_all_features(i)
        for name in feature_names:
            feature_dict[name].append(x[name])
    feature_dict = {name: torch.cat(feature_dict[name], dim=0) for name in feature_names}
    stats = {name: {'max': x.max(0)[0].numpy(),
                    'min': x.min(0)[0].numpy(),
                    'mean': x.mean(0).numpy(),
                    'std': x.std(0).numpy()}
                for name, x in feature_dict.items()}
    log.info("Calculated statistics for the following features:")
    log.info(feature_names)
    log.info(f"saving to {stat_path}")
    np.save(stat_path, stats)
    log.info("Done.")

if __name__=='__main__':
    main()
