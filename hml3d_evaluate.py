import os
import logging
import hydra
import joblib
from omegaconf import DictConfig
from omegaconf import OmegaConf
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

@hydra.main(config_path="configs", config_name="hml3d_eval")
def _hml3d_sample(cfg: DictConfig) -> None:
    return hml3d_sample(cfg)

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


def hml3d_sample(newcfg: DictConfig) -> None:
    from pathlib import Path
    exp_folder = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")
    from src.data.humanml3d import HumanML3DDataModule
    hml3d_dataset = HumanML3DDataModule(debug=False,
                                    datapath='data/humanml3d_processed', # amass_bodilex.pth.tar
                                    debug_datapath='data/amass_small.pth.tar',
                                    annot_path='data/annotations/humanml3d/annotations.json',
                                    # Amass
                                    smplh_path='data/body_models',
                                    smplh_path_dbg='minidata/body_models',
                                    load_splits=['test'],
                                    # Machine
                                    batch_size=32,
                                    num_workers=12,
                                    rot_repr='6d',
                                    preproc=OmegaConf.create(
                                      {'stats_file':'deps/stats/statistics_hml3d.npy',  # full path for statistics
                                      'split_seed':0,
                                      'calculate_minmax':True,
                                      'generate_joint_files':True,
                                      'use_cuda':True,
                                      'n_body_joints':22,
                                      'norm_type':'std'}),
                                    framerate=30
                                    ,load_feats=["body_transl"
                                    ,"body_transl_delta"
                                    ,"body_transl_delta_pelv"
                                    ,"body_transl_delta_pelv_xy"
                                    ,"body_transl_z"
                                    ,"body_orient"
                                    ,"body_pose"
                                    ,"body_orient_delta"
                                    ,"body_pose_delta"
                                    ,"body_orient_xy",
                                    'z_orient_delta', 
                                    'body_joints_local_wo_z_rot'
                                    ,"body_joints"],
                                    progress_bar=True
                        )

    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
    # change scheduler for inference
    from src.diffusion import create_diffusion

    from src.diffusion.gaussian_diffusion import ModelMeanType, ModelVarType
    from src.diffusion.gaussian_diffusion import LossType
    if cfg.num_sampling_steps is not None:
        if cfg.num_sampling_steps <= cfg.model.diff_params.num_train_timesteps:
            num_infer_steps = cfg.num_sampling_steps
        else:
            num_infer_steps = cfg.model.diff_params.num_train_timesteps
            logger.info('More sampling steps than the training ones! Sampling with maximum')
            logger.info(f'Number of steps: {num_infer_steps}')
    else:
        num_infer_steps = cfg.model.diff_params.num_train_timesteps

    diffusion_process = create_diffusion(timestep_respacing=None,
                                    learn_sigma=False,
                                    sigma_small=True,
                                    diffusion_steps=num_infer_steps,
                                    noise_schedule=cfg.model.diff_params.noise_schedule,
                                    predict_xstart=False if cfg.model.diff_params.predict_type == 'noise' else True) # noise vs sample

    # cfg.model.infer_scheduler = newcfg.model.infer_scheduler
    # cfg.model.diff_params.num_inference_timesteps = newcfg.steps
    # cfg.model.diff_params.guidance_scale_motion = newcfg.guidance_scale_motion
    # cfg.model.diff_params.guidance_scale_text = newcfg.guidance_scale_text
    init_diff_from = cfg.init_from
    # init_diff_from = 'noise'
    # TODO pUT THIS BACK    
    # fd_name = get_folder_name(cfg)
    fd_name = f'steps_{cfg.num_sampling_steps}'
    log_name = '__'.join(str(exp_folder).split('/')[-2:])
    log_name = f'samples_{log_name}_steps-{cfg.num_sampling_steps}_{cfg.init_from}_{cfg.ckpt_name}'

    output_path = exp_folder / f't2m_{fd_name}_hml3d_{cfg.init_from}_{cfg.ckpt_name}'
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"-------Output path:{output_path}------")
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
    from src.model.base_diffusion import MD    
    # Load the last checkpoint
    model = MD.load_from_checkpoint(last_ckpt_path,
                                       renderer=aitrenderer,
                                    #    infer_scheduler=cfg.model.infer_scheduler,
                                    #    diff_params=cfg.model.diff_params,
                                       strict=False)
    model.freeze()
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    # logger.info('------Generating using Scheduler------\n\n'\
    #             f'{model.infer_scheduler}')
    logger.info('------Diffusion Parameters------\n\n'\
                f'{model.diff_params}')

    import numpy as np
    test_dataset_hml3d = hml3d_dataset.dataset['test']
    from src.data.tools.collate import collate_batch_last_padding
    features_to_load = hml3d_dataset.dataset['test'].load_feats
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)
    from src.utils.art_utils import color_map
    testloader = torch.utils.data.DataLoader(test_dataset_hml3d,
                                             shuffle=False,
                                             num_workers=8,
                                             batch_size=256,
                                             collate_fn=collate_fn)
    ds_iterator = testloader 

    from src.utils.art_utils import color_map
    
    mode_cond = cfg.condition_mode
    if cfg.model.motion_condition is None:
        mode_cond = 'text_cond'
    else:
        mode_cond = cfg.condition_mode

    tot_pkls = []
    if cfg.guidance_scale_text is None:
        gd_text = [2.5, 3.0]
    else:
        gd_text = [cfg.guidance_scale_text] # [1.0, 2.5, 5.0]
    if cfg.model.motion_condition is None:
        mode_cond = 'text_cond'
    else:
        mode_cond = 'full_cond'
    logger.info(f'Evaluation Set length:{len(test_dataset_hml3d)}')
    with torch.no_grad():
        for guid_text in gd_text:
            cur_guid_comb = f'ld_txt-{guid_text}'
            cur_outpath = output_path / cur_guid_comb
            cur_outpath.mkdir(exist_ok=True, parents=True)
            logger.info(f"Sample MotionFix test set\n in:{cur_outpath}")

            for batch in tqdm(ds_iterator):

                text_diff = batch['text']
                target_lens = batch['length_target']
                keyids = batch['id']
                no_of_motions = len(keyids)

                input_batch = prepare_test_batch(model, batch)
                if model.motion_condition == 'source' or init_diff_from == 'source':
                    source_lens = batch['length_source']
                    if model.pad_inputs:
                        mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                        target_lens,
                                                                        max_len=300)
                        source_mot_pad = torch.nn.functional.pad(input_batch['source_motion'],
                                                                (0, 0, 0, 0, 0,
                                                    300 - input_batch['source_motion'].size(0)),
                                                                value=0)
                    else:
                        mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                        target_lens,
                                                                        max_len=None)
                        source_mot_pad = input_batch['source_motion'].clone()

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
                                                source_init,
                                                mask_source,
                                                mask_target,
                                                diffusion_process,
                                                init_vec=source_init,
                                                init_vec_method=init_diff_from,
                                                condition_mode=mode_cond,
                                                gd_motion=None,
                                                gd_text=guid_text,
                                                num_diff_steps=num_infer_steps)
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

    _hml3d_sample()
