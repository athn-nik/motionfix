import logging

import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from sinc.data.tools.collate import collate_length_and_text
import sinc.launch.prepare
from sinc.render.mesh_viz import visualize_meshes
from sinc.render.video import save_video_samples, stack_vids
import torch
from sinc.transforms.base import Datastruct
from sinc.utils.inference import cfg_mean_nsamples_resolution, get_path
from sinc.utils.file_io import read_json
import pytorch_lightning as pl
import numpy as np
from hydra.utils import instantiate
from tqdm import tqdm
from sinc.data.tools import collate_text_and_length
from sinc.tools.frank import combine_motions
from sinc.utils.inference import sinc_supmat

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="sample")
def _sample(cfg: DictConfig):
    return sample(cfg)


def fix_config_if_needed(cfg):
    if 'gpt_path' not in cfg.data:
        cfg.data['gpt_path'] = '${path.deps}/gpt/gpt3-labels.json'

def prepare_path(jtype, ckpt_name, outdir, set):
        
    if ckpt_name != 'last':
        logger.info(f"Evaluating on checkpoint: {ckpt_name}")
        ckpt_fd = f'checkpoint-{ckpt_name}'
    else:
        ckpt_fd = 'checkpoint-last'
    if set == 'supmat' or 'ood' in set:
        assert jtype in ['vertices', 'rots']
        storage = outdir / f'visual_samples_{set}/{ckpt_fd}'
    else:
        if jtype == 'vertices':
            storage = outdir / f'samples_vertices/{ckpt_fd}'
        elif jtype == 'rots':
            storage = outdir / f'samples_rots/{ckpt_fd}'
        else: # joints
            storage = outdir / f'samples/{ckpt_fd}'
    storage.mkdir(exist_ok=True,  parents=True)
    return storage

def path_to_save(cur_path, split, baseline, set_to_sample, onesample, mean, fact):

    # check baseline sampling
    if baseline == "gpt":
        split_name = split + "_naive_gpt"
    elif baseline == "concat":
        split_name = split + "_naive_concat"
    else:
        split_name = split 
    # check set to sample from     
    
    split_name = split_name + f"_{set_to_sample}"
    path = get_path(cur_path, split_name,
                    onesample, mean, fact)

    return path

def get_keyids(set_to_sample, data_module):
    if set_to_sample == 'submission':
        from sinc.utils.inference import sinc_eval_set
        keyids = sinc_eval_set
    elif set_to_sample == 'supmat':
        from sinc.utils.inference import sinc_supmat
        keyids = sinc_supmat
    elif set_to_sample == 'ood':
        from sinc.utils.inference import sinc_ood
        keyids = sinc_ood
    elif set_to_sample == 'ood2':
        from sinc.utils.inference import sinc_ood_2
        keyids = sinc_ood_2
    elif set_to_sample == 'ood3':
        from sinc.utils.inference import sinc_ood_three
        keyids = sinc_ood_three
    elif set_to_sample == 'oodgpt':
        from sinc.utils.inference import sinc_ood_gptfail
        keyids = sinc_ood_gptfail

    else:
        if set_to_sample == 'pairs':
            keyids = [k for k in data_module.keyids if k.split('-')[0] == 'spatial_pairs']
        elif set_to_sample == 'single':
            keyids = [k for k in data_module.keyids if k.split('-')[0] in ['seq', 'seg']]
        else:
            keyids = data_module.keyids

    return keyids


def sample(newcfg: DictConfig) -> None:
    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path

    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")

    fix_config_if_needed(prevcfg)

    if newcfg.naive:
        # The naive version should not have been trained on spatial pairs
        assert "spatial_pairs" not in prevcfg.data.dtype

    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)

    # prepare and create path for samples
    onesample = cfg_mean_nsamples_resolution(cfg)
    storage = prepare_path(cfg.jointstype, cfg.ckpt_name, output_dir, cfg.set)
    path = path_to_save(storage, cfg.split,
                        cfg.naive, cfg.set,
                        onesample, cfg.mean,
                        cfg.fact)    
    path.mkdir(exist_ok=True, parents=True)

    logger.info(f'Sample script. The outputs will be stored in:{path}')

    seed_logger = logging.getLogger("pytorch_lightning.utilities.seed")
    seed_logger.setLevel(logging.WARNING)
    pl.seed_everything(cfg.seed)

    # Inittialize Dataloader
    if 'ood' not in cfg.set:
        logger.info("Loading data module")
        cfg.data.dtype = 'spatial_pairs+seg+seq'
        data_module = instantiate(cfg.data)
        logger.info(f"Data module '{cfg.data.dataname}' loaded")
        dataset = getattr(data_module, f"{cfg.split}_dataset")
        keyids = get_keyids(cfg.set, dataset)

    else:
        if cfg.set == 'ood':
            from sinc.utils.inference import sinc_ood
            keyids = sinc_ood
        elif cfg.set == 'ood2':
            from sinc.utils.inference import sinc_ood_2
            keyids = sinc_ood_2
        elif cfg.set == 'ood3':
            from sinc.utils.inference import sinc_ood_three
            keyids = sinc_ood_three
        elif cfg.set == 'oodgpt':
            from sinc.utils.inference import sinc_ood_gptfail
            from sinc.utils.file_io import read_json
            from sinc.tools.frank import text_list_to_bp
            keyids = sinc_ood_gptfail
            gppt_path = './deps/gpt/gpt3-labels-list.json'
            gpt_labels = read_json(gppt_path)


    logger.info("Loading model")

    # Load the model from checkpoint
    model = instantiate(cfg.model,
                        nfeats=135,
                        logger_name="none",
                        nvids_to_save=None,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    model = model.load_from_checkpoint(last_ckpt_path)
    model.eval()
    logger.info("Model weights restored")
    model.sample_mean = cfg.mean
    model.fact = cfg.fact
    logger.info("Trainer initialized")

    motion_type = cfg.jointstype
    # from sinc.utils.inference import extras_mots_puta
    with torch.no_grad():
        with tqdm(total=len(keyids), position=0, leave=True) as pbar:
            for keyid in (pbar := tqdm(keyids,  position=0, leave=True)):
                pbar.set_description(f"Processing {keyid}")
                # if keyid not in extras_mots_puta:
                #     continue
                # one_data = dataset.load_keyid(keyid, mode='inference')
                # mot_ds = one_data['datastruct']
                
                # sample_dict = {'rots': mot_ds.rots.numpy(),
                #                 'trans': mot_ds.trans.numpy(),
                #                 'text': one_data['length'],
                #                 'lengths':one_data['length'],
                # }
                
                # np.save(f'/home/nathanasiou/Desktop/{keyid}.npy', sample_dict)
                # continue
                # continue
                # if (path / f"{keyid}.npy").is_file():
                #     continue
                
                if  'ood' not in cfg.set:
                    one_data = dataset.load_keyid(keyid, mode='inference')

                    batch = collate_text_and_length([one_data])
                
                    cur_lens = batch['length']
                    cur_texts = [list(batch['text'][0])]
                    dtype_sample = keyid.split('-')[0]
                    is_spatial_pair = dtype_sample == 'spatial_pairs'

                else:
                    cur_lens = [keyid[1]]
                    cur_texts = [list(keyid[0])]
                    is_spatial_pair = True
                    acts = []
                    for a in cur_texts[0]:
                        acts.append(a.replace(' ', '-')) 
                    fname = '__'.join(acts)
                    fname = fname + '___fr' + str(keyid[1])                                        
                    
                for index in range(cfg.number_of_samples):
                    # fix the seed
                    pl.seed_everything(index)
                    try:
                        if cfg.set=='oodgpt':
                            gpt_1 = text_list_to_bp(cur_texts[0][0], gpt_labels)
                            gpt_2 = text_list_to_bp(cur_texts[0][1], gpt_labels)
                            gpt_parts = [gpt_1, gpt_2]

                        elif model.hparams.gpt_proxy:
                            gpt_parts = batch['bp-gpt']
                        else:
                            gpt_parts = None

                    except AttributeError:
                        gpt_parts = None


                    if is_spatial_pair and cfg.naive == 'gpt':
                        if 'ood' not in cfg.set:
                            gpt_parts = batch['bp-gpt'][0]

                        motion1 = model.text_to_motion_forward([[cur_texts[0][0]]],
                                                               cur_lens,
                                                               gpt_parts=None,
                                                               return_motion="rotsd",
                                                               )

                        motion2 = model.text_to_motion_forward([[cur_texts[0][1]]],
                                                               cur_lens,
                                                               gpt_parts=None,
                                                               return_motion="rotsd")


                        # rots and transl
                        frank_motion = combine_motions(motion1, motion2, gpt_parts[0], gpt_parts[1], squeeze=True)
                        frank_datastruct = model.Datastruct(rots_=frank_motion)

                        motion = model.motion_from_datastruct(frank_datastruct, return_type=motion_type)

                        # keep individual motions also
                        motion1 = model.motion_from_datastruct(model.Datastruct(rots_=motion1), return_type=motion_type)
                        motion2 = model.motion_from_datastruct(model.Datastruct(rots_=motion2), return_type=motion_type)

                    elif is_spatial_pair and cfg.naive == 'concat':
                        concat_text = [[" while ".join(cur_texts[0])]]
                        motion = model.text_to_motion_forward(concat_text,
                                                              cur_lens,
                                                              gpt_parts=gpt_parts,
                                                              return_motion=motion_type)

                    else:
                        motion = model.text_to_motion_forward(cur_texts,
                                                              cur_lens,
                                                              gpt_parts=gpt_parts,
                                                              return_motion=motion_type,
                                                              conjuct_word=cfg.conj_word)

                    # Save separate or all motions
                    if cfg.jointstype == "rots":

                        # one sample needs to be squeezed
                        sample_dict = {'rots': torch.squeeze(motion[0]).numpy(),
                                       'trans': torch.squeeze(motion[1]).numpy(),
                                       'text': cur_texts[0],
                                       'lengths': cur_lens[0]
                        }
                        if is_spatial_pair and cfg.naive == "gpt":
                            sample_dict['motion1_rots'] = torch.squeeze(motion1[0]).numpy()
                            sample_dict['motion1_trans'] = torch.squeeze(motion1[1]).numpy()
                            sample_dict['motion2_rots'] = torch.squeeze(motion2[0]).numpy()
                            sample_dict['motion2_trans'] = torch.squeeze(motion2[1]).numpy()
                    else:
                        sample_dict = {'motion': torch.squeeze(motion).numpy(),
                                       'text': cur_texts[0],
                                       'lengths': cur_lens[0]
                        }
                        if is_spatial_pair and cfg.naive == "gpt":
                            sample_dict['motion1'] = torch.squeeze(motion1).numpy()
                            sample_dict['motion2'] = torch.squeeze(motion2).numpy()
                    if 'ood' in cfg.set:
                        if cfg.number_of_samples > 1:
                            npypath = path / f"{fname}_{index}.npy"
                            # kd = f'{keyid}_{index}'
                        else:
                            # kd = f'{keyid}'
                            npypath = path / f"{fname}.npy"
                    else:
                        if cfg.number_of_samples > 1:
                            npypath = path / f"{keyid}_{index}.npy"
                            # kd = f'{keyid}_{index}'
                        else:
                            # kd = f'{keyid}'
                            npypath = path / f"{keyid}.npy"

                    np.save(npypath, sample_dict)

    logger.info(f"The samples are ready, you can find them here:\n{path}")


if __name__ == '__main__':
    _sample()
