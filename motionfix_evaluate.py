import os
import logging
import hydra
import joblib
from omegaconf import DictConfig
from omegaconf import OmegaConf
from sympy import O
from src import data
from src.render.mesh_viz import render_motion
from torch import Tensor

# from src.render.mesh_viz import visualize_meshes
from src.render.video import save_video_samples
import src.launch.prepare  # noqa
from tqdm import tqdm
import torch
import itertools
from src.model.utils.tools import pack_to_render
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="motionfix_eval")
def _render_vids(cfg: DictConfig) -> None:
    return render_vids(cfg)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def prepare_test_batch(model, batch):
    batch = { k: v.to(model.device) if torch.is_tensor(v) else v
                for k, v in batch.items() }

    input_batch = model.norm_and_cat(batch, model.input_feats)
    for k, v in input_batch.items():
        if model.input_deltas:
            batch[f'{k}_motion'] = v[1:]
        else:
            batch[f'{k}_motion'] = v
            batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]

    return batch

def cleanup_files(lo_fls):
    for fl in lo_fls:
        os.remove(fl)

def get_folder_name(config):
    sched_name = config.model.infer_scheduler._target_.split('.')[-1]
    sched_name = sched_name.replace('Scheduler', '').lower()
    mot_guid = config.model.diff_params.guidance_scale_motion
    text_guid = config.model.diff_params.guidance_scale_text
    infer_steps = config.model.diff_params.num_inference_timesteps
    if config.init_from == 'source':
        init_from = '_src_init_'
    else:
        init_from = ''
    if config.ckpt_name == 'last':
        ckpt_n = ''
    else:
        ckpt_n = f'ckpt-{config.ckpt_name}_'

    return f'{ckpt_n}{init_from}{sched_name}_steps{infer_steps}'


def render_vids(newcfg: DictConfig) -> None:
    from pathlib import Path
    exp_folder = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")
    
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
    # change scheduler for inference
    cfg.model.infer_scheduler = newcfg.model.infer_scheduler
    cfg.model.diff_params.num_inference_timesteps = newcfg.steps
    cfg.model.diff_params.guidance_scale_motion = newcfg.guidance_scale_motion
    cfg.model.diff_params.guidance_scale_text = newcfg.guidance_scale_text
    init_diff_from = cfg.init_from

    fd_name = get_folder_name(cfg)
    log_name = '__'.join(str(exp_folder).split('/')[-2:])
    log_name = f'{log_name}_{init_diff_from}_{cfg.ckpt_name}'

    output_path = exp_folder / f'{fd_name}_{cfg.data.dataname}'
    output_path.mkdir(exist_ok=True, parents=True)

    log_name = '__'.join(str(exp_folder).split('/')[-2:])

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.render.video import put_text
    from src.render.video import stack_vids
    from tqdm import tqdm

    seed_logger = logging.getLogger("pytorch_lightning.utilities.seed")
    seed_logger.setLevel(logging.WARNING)

    pl.seed_everything(cfg.seed)
    # import wandb
    # wandb.init(project="pose-edit-eval", job_type="evaluate",
    #            name=log_name, dir=output_path)
    aitrenderer = None
    logger.info("Loading model")
    model = instantiate(cfg.model,
                        renderer=None,
                        _recursive_=False)

    logger.info(f"Model '{cfg.model.modelname}' loaded")
    
    # Load the last checkpoint
    model = model.load_from_checkpoint(last_ckpt_path,
                                       renderer=aitrenderer,
                                       infer_scheduler=cfg.model.infer_scheduler,
                                       diff_params=cfg.model.diff_params,
                                       strict=False)
    model.freeze()
    logger.info("Model weights restored")
    logger.info("Trainer initialized")
    logger.info('------Generating using Scheduler------\n\n'\
                f'{model.infer_scheduler}')
    logger.info('------Diffusion Parameters------\n\n'\
                f'{model.diff_params}')

    import numpy as np
    data_module = instantiate(cfg.data, amt_only=True, load_splits=['test'])

    transl_feats = [x for x in model.input_feats if 'transl' in x]
    if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                   "body_transl_delta_pelv_xy"]):
        model.using_deltas_transl = True
    # load the test set and collate it properly
    features_to_load = data_module.dataset['test'].load_feats
    test_dataset = data_module.dataset['test'] + data_module.dataset['val']
    
    from src.data.tools.collate import collate_batch_last_padding
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)

    subset = []
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             shuffle=False,
                                             num_workers=0,
                                             batch_size=128,
                                             collate_fn=collate_fn)
    ds_iterator = testloader 

    from src.utils.art_utils import color_map
    
    mode_cond = cfg.condition_mode
    if cfg.model.motion_condition is None:
        mode_cond = 'text_cond'
    else:
        mode_cond = cfg.condition_mode

    tot_pkls = []
    gd_text = [cfg.guidance_scale_text] #[1.0, 2.5, 5.0]
    gd_motion = [cfg.guidance_scale_motion] #[1.0, 2.5, 5.0]
    guidances_mix = [(x, y) for x in gd_text for y in gd_motion]
    if cfg.model.motion_condition is None:
        mode_cond = 'text_cond'
    else:
        mode_cond = 'full_cond'
    logger.info(f'Evaluation Set length:{len(test_dataset)}')
    with torch.no_grad():
        output_path = output_path / 'samples'
        logger.info(f"Sample MotionFix test set\n in:{output_path}")
        output_path.mkdir(exist_ok=True, parents=True)
        for guid_text, guid_motion in guidances_mix:
            cur_guid_comb = f'ld_txt-{guid_text}_ld_mot-{guid_motion}'
            cur_outpath = output_path / cur_guid_comb
            cur_outpath.mkdir(exist_ok=True, parents=True)
            for batch in tqdm(ds_iterator):

                text_diff = batch['text']
                target_lens = batch['length_target']
                keyids = batch['id']
                no_of_motions = len(keyids)

                input_batch = prepare_test_batch(model, batch)
                # for k, v in input_batch.items():
                #     batch[f'{k}_motion'] = torch.nn.functional.pad(v,
                #                                                    (0, 0, 0, 0,
                #                                                     0, 300 - v.size(0)),
                #                                                    value=0)
                source_mot_pad = torch.nn.functional.pad(input_batch['source_motion'],
                                                        (0, 0, 0, 0, 0,
                                            300 - input_batch['source_motion'].size(0)),
                                                        value=0)
                if model.motion_condition == 'source' or init_diff_from == 'source':
                    source_lens = batch['length_source']
                    mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                    target_lens)
                else:
                    from src.data.tools.tensors import lengths_to_mask
                    mask_target = lengths_to_mask(target_lens,
                                                model.device)
                    batch['source_motion'] = None
                    mask_source = None

                if init_diff_from == 'source':
                    source_init = source_mot_pad
                else:
                    source_init = None
                diffout = model.generate_motion(text_diff,
                                                source_mot_pad,
                                                mask_source,
                                                mask_target,
                                                init_vec=source_init,
                                                init_vec_method=init_diff_from,
                                                condition_mode=mode_cond,
                                                gd_motion=guid_motion,
                                                gd_text=guid_text,
                                                num_diff_steps=newcfg.steps)
                gen_mo = model.diffout2motion(diffout)
                from src.tools.transforms3d import transform_body_pose
                for i in range(gen_mo.shape[0]):
                    dict_to_save = {'pose': gen_mo[i].cpu().numpy() }
                    np.save(cur_outpath / f"{str(batch['id'][i]).zfill(6)}.npy",
                            dict_to_save)
                    # np.load(output_path / f"{str(batch['id'][i]).zfill(6)}.npy")
                # output_path = Path('/home/nathanasiou/Desktop/conditional_action_gen/modilex')
                   
        logger.info(f"Sample script. The outputs are stored in:{cur_outpath}")
    
if __name__ == '__main__':

    _render_vids()
