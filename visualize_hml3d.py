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
import torch.nn.functional as F
# from src.render.mesh_viz import visualize_meshes
from src.render.video import save_video_samples
import src.launch.prepare  # noqa
from tqdm import tqdm
import torch
import itertools
from src.model.utils.tools import pack_to_render
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="hml3d_viz")
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
            # batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]
    return batch

def cleanup_files(lo_fls):
    for fl in lo_fls:
        os.remove(fl)

def output2renderable(model, lst_of_tensors: list[Tensor]):
    l_of_renders = []
    for el in lst_of_tensors:
        if isinstance(el, list):
            render_readies = []
            for sub_el in el:
                modict_nest = pack_to_render(rots=sub_el[...,
                                                            3:].detach().cpu(),
                                             trans=sub_el[...,
                                                            :3].detach().cpu())
                render_readies.append(modict_nest)
        else:
            render_readies = pack_to_render(rots=el[...,
                                                            3:].detach().cpu(),
                                            trans=el[...,
                                                         :3].detach().cpu())
        l_of_renders.append(render_readies)

    return l_of_renders

def get_folder_name(config):
    sched_name = config.model.infer_scheduler._target_.split('.')[-1]
    sched_name = sched_name.replace('Scheduler', '').lower()
    mot_guid = config.model.diff_params.guidance_scale_motion
    text_guid = config.model.diff_params.guidance_scale_text
    infer_steps = config.model.diff_params.num_inference_timesteps
    if config.init_from == 'source':
        init_from = 'src'
    else:
        init_from = 'noise'
    if config.ckpt_name == 'last':
        ckpt_n = '_ckpt_last_'
    else:
        ckpt_n = f'_ckpt-{config.ckpt_name}_'

    if config.model.motion_condition is not None:
        return f'{init_from}{ckpt_n}{sched_name}_steps{infer_steps}'
    else:
        return f'{init_from}{ckpt_n}{sched_name}_steps{infer_steps}'

def render_vids(newcfg: DictConfig) -> None:
    from pathlib import Path
    from src.data.humanml3d import HumanML3DDataModule
    data_conf = OmegaConf.load(Path(newcfg.path.code_dir) /'configs/data/hml3d.yaml')
    # hml3d_dataset instantiate(cfg.data, )
    # hml3d_dataset = HumanML3DDataModule(debug=False,
    #                                 datapath='data/humanml3d_processed', # amass_bodilex.pth.tar
    #                                 debug_datapath='data/amass_small.pth.tar',
    #                                 annot_path='data/annotations/humanml3d/annotations.json',
    #                                 # Amass
    #                                 smplh_path='data/body_models',
    #                                 smplh_path_dbg='minidata/body_models',
    #                                 load_splits=['test'],
    #                                 # Machine
    #                                 batch_size=32,
    #                                 num_workers=12,
    #                                 rot_repr='6d',
    #                                 preproc=OmegaConf.create(
    #                                   {'stats_file':'deps/stats/statistics_hml3d.npy',  # full path for statistics
    #                                   'split_seed':0,
    #                                   'calculate_minmax':True,
    #                                   'generate_joint_files':True,
    #                                   'use_cuda':True,
    #                                   'n_body_joints':22,
    #                                   'norm_type':'std'}),
    #                                 framerate=30
    #                                 ,load_feats=["body_transl"
    #                                 ,"body_transl_delta"
    #                                 ,"body_transl_delta_pelv"
    #                                 ,"body_transl_delta_pelv_xy"
    #                                 ,"body_transl_z"
    #                                 ,"body_orient"
    #                                 ,"body_pose"
    #                                 ,"body_orient_delta"
    #                                 ,"body_pose_delta"
    #                                 ,"body_orient_xy",
    #                                 'z_orient_delta', 
    #                                 'body_joints_local_wo_z_rot'
    #                                 ,"body_joints"],
    #                                 progress_bar=True
    #                     )

    OmegaConf.merge(data_conf, newcfg.data)
    
    exp_folder = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")

    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
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

    # change scheduler for inference
    init_diff_from = cfg.init_from

    fd_name = get_folder_name(cfg)

    log_name = '__'.join(str(exp_folder).split('/')[-2:])
    if cfg.inpaint:
        log_name = f'{log_name}_{init_diff_from}_{cfg.ckpt_name}_inpaint_bsl'
    else:
        log_name = f'{log_name}_{init_diff_from}_{cfg.ckpt_name}'

    output_path = exp_folder / 'renders' / fd_name
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Sample script. The outputs will be stored in:{output_path}")

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.render.video import put_text
    from src.render.video import stack_vids
    from tqdm import tqdm

    seed_logger = logging.getLogger("pytorch_lightning.utilities.seed")
    seed_logger.setLevel(logging.WARNING)
    pl.seed_everything(cfg.seed)
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.configuration import CONFIG as AITVIEWER_CONFIG
    AITVIEWER_CONFIG.update_conf({"playback_fps": 30,
                                "auto_set_floor": True,
                                "smplx_models": 'data/body_models',
                                'z_up': True})
    aitrenderer = HeadlessRenderer()
    import wandb
    wandb.init(project="hml3d_visuals", job_type="evaluate",
               name=log_name, dir=output_path,
               settings=wandb.Settings(start_method='fork'))

    logger.info("Loading model")
    from src.model.base_diffusion import MD    
    # Load the last checkpoint
    model = MD.load_from_checkpoint(last_ckpt_path,
                                       renderer=aitrenderer,
                                    #    infer_scheduler=cfg.model.infer_scheduler,
                                    #    diff_params=cfg.model.diff_params,
                                       strict=False)
    model.eval()
    hml3d_dataset = instantiate(cfg.data, load_splits=['test'])
    
    logger.info("Model weights restored")
    logger.info("Trainer initialized")
    # logger.info('------Generating using Scheduler------\n\n'\
    #             f'{model.infer_scheduler}')
    logger.info('------Diffusion Parameters------\n\n'\
                f'{model.diff_params}')

    import numpy as np

    transl_feats = [x for x in model.input_feats if 'transl' in x]
    if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                   "body_transl_delta_pelv_xy"]):
        model.using_deltas_transl = True

    # load the test set and collate it properly
    test_dataset_hml3d = hml3d_dataset.dataset['test']
    from src.data.tools.collate import collate_batch_last_padding
    features_to_load = test_dataset_hml3d.load_feats
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)
    from src.utils.art_utils import color_map
    # subset_hml = []
    
    # cnt_sit = 0
    # cnt_walk = 0
    # tot_cnt = 0
    # for elem in test_dataset_hml3d.data:
    #     if 'sit' in elem['text'][0]:
    #         cnt_sit += 1
    #         subset_hml.append(elem)

    #     elif 'walk' in elem['text'][0]:
    #         cnt_walk += 1
    #         subset_hml.append(elem)
    #     elif tot_cnt < 30:
    #         subset_hml.append(elem)
    #         tot_cnt += 1

    #     if cnt_sit > 10 and cnt_walk > 10 and tot_cnt >= 30:
    #         break
    # from src.utils.motionfix_utils import test_subset_hml3d

    # batch_size_test = 16
    # test_subset_hml3d = [elem for elem in test_dataset_hml3d
    #                      if elem['id'] in test_subset_hml3d]

    # idss = [0,1,2,3,8]
    # filtered_list = [subset_hml[index] for index in idss]
    # test_dataset_hml3d.data = filtered_list[:1]
    batch_size_test = 32
    testloader_hml3d = torch.utils.data.DataLoader(test_dataset_hml3d,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   batch_size=batch_size_test,
                                                   collate_fn=collate_fn)

    ds_iterator_hml3d = testloader_hml3d 

    init_diff_from = cfg.init_from
    mode_cond = 'text_cond'

    tot_pkls = []
    gd_text = [2.5, 5.0, 7.5]

    from aitviewer.models.smpl import SMPLLayer
    smpl_layer = SMPLLayer(model_type='smplh', 
                            ext='npz',
                            gender='neutral')
    if cfg.inpaint:
        assert cfg.data.dataname == 'sinc_synth'
        from src.utils.file_io import read_json
        annots_sinc = read_json('data/sinc_synth/for_website_v4.json')
    if cfg.linear_gd:
        use_linear_guid = True
    else:
        use_linear_guid = False

    with torch.no_grad():
        output_path.mkdir(exist_ok=True, parents=True)
        for g_text in gd_text:
            cur_guid_comb = f'ld_txt-{g_text}'
            cur_output_path = output_path / cur_guid_comb
            cur_output_path.mkdir(exist_ok=True, parents=True)
            batch_to_rend = 1
            for batch in tqdm(ds_iterator_hml3d):
                
                if not batch_to_rend:
                    continue

                text_diff = batch['text']
                target_lens = batch['length_target']
                keyids = batch['id']
                no_of_motions = len(keyids)
                in_batch = prepare_test_batch(model, batch)
                if cfg.inpaint:
                    ############### BODY PART BASELINE ###############
                    from src.model.utils.body_parts import get_mask_from_texts, get_mask_from_bps
                    # jts idxs #Texts x [jts ids] list of lists

                    parts_to_keep = [annots_sinc[kd]['source_annot'] 
                                     if kd.endswith(('_0', '_1'))
                                     else annots_sinc[kd]['target_annot']
                                     for kd in keyids]
                    jts_ids = get_mask_from_texts(parts_to_keep)
                    # True for involved body_parts aka joint groups
                    # Tensor #Texts x features [207]
                    mask_features = get_mask_from_bps(jts_ids, device=model.device, 
                                                    feat_dim=sum(model.input_feats_dims)) 
                    ##################################################
                    inpaint_dict = {'mask': mask_features,
                                    'start_motion': in_batch['source_motion'].clone() }
                else:
                    inpaint_dict = None
                if model.motion_condition == 'source' or init_diff_from == 'source':
                    source_lens = batch['length_source']
                    if model.pad_inputs:
                        mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                        target_lens,
                                                                        max_len=300)
                        source_mot_pad = torch.nn.functional.pad(in_batch['source_motion'],
                                                                (0, 0, 0, 0, 0,
                                                    300 - in_batch['source_motion'].size(0)),
                                                                value=0)
                    else:
                        mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                        target_lens,
                                                                        max_len=None)
                        source_mot_pad = in_batch['source_motion'].clone()
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
                diffout = model.generate_motion(text_diff, # text
                                                source_init, # source
                                                mask_source,
                                                mask_target,
                                                diffusion_process,
                                                gd_text=g_text,
                                                init_vec=source_init,
                                                init_vec_method=init_diff_from,
                                                condition_mode=mode_cond,
                                                num_diff_steps=num_infer_steps,
                                                inpaint_dict=inpaint_dict,
                                                use_linear=use_linear_guid
                                                )
                gen_mo = model.diffout2motion(diffout)
                #gen_mot0 = gen_mo.detach().cpu()
                #dtr = pack_to_render(rots=gen_mot0[0, :, 3:], trans=gen_mot0[0, :, :3])
                #ptsavem='/home/nathanasiou/Desktop/conditional_action_gen/modilex'
                #render_motion(aitrenderer, dtr, ptsavem+'/gendiff',pose_repr='aa',smpl_layer=smpl_layer)

                src_mot_cond, tgt_mot = model.batch2motion(in_batch,
                                                pack_to_dict=False)
                tgt_mot = tgt_mot.to(model.device)
                if init_diff_from == 'source':
                    src_mot_cond = src_mot_cond.to(model.device)
                else:
                    src_mot_cond = None
                mots_to_render = [tgt_mot, gen_mo]
                monames = ['target',  'generated']

                lof_mots = output2renderable(model,
                                            mots_to_render)
                lens_to_mask = [target_lens, target_lens]
                logger.info(f"Rendering the motion videos...")
                for elem_id in tqdm(range(no_of_motions)):
                    cur_group_of_vids = []
                    curid = keyids[elem_id]
                    for moid in range(len(monames)):
                        one_motion = lof_mots[moid]
                        cur_mol = []
                        cur_colors = []
                        if lens_to_mask[moid] is not None:
                            crop_len = lens_to_mask[moid]

                        if isinstance(one_motion, list):
                            for xx in one_motion:
                                cur_mol.append({k: v[elem_id] 
                                                for k, v in xx.items()})
                            cur_colors = [color_map['source'],
                                        color_map['target']]
                        else:
                            cur_mol.append({k: v[elem_id][:crop_len[elem_id]]
                                                for k, v in one_motion.items()})
                            cur_colors.append(color_map[monames[moid]])

                        fname = render_motion(aitrenderer, cur_mol,
                                            cur_output_path / f"movie_{elem_id}_{moid}",
                                            pose_repr='aa',
                                            text_for_vid=monames[moid],
                                            color=cur_colors,
                                            smpl_layer=smpl_layer)
                        cur_group_of_vids.append(fname)
                    stacked_vid = stack_vids(cur_group_of_vids,
                                            f'{cur_output_path}/{elem_id}_stacked.mp4',
                                            orient='h')
                    fnal_fl = put_text(text=text_diff[elem_id],
                                       fname=stacked_vid, 
                                       outf=f'{cur_output_path}/{curid}_text.mp4',
                                       position='top_center'
                                       )

                    cleanup_files(cur_group_of_vids+[stacked_vid])
                    video_key = fnal_fl.split('/')[-1].replace('.mp4','')
                    wandb.log({f"{cur_guid_comb}/{video_key}": wandb.Video(fnal_fl,
                                                                fps=30,
                                                                format="mp4")})
                batch_to_rend -= 1

    from src.utils.file_io import write_json   
    write_json(tot_pkls, output_path / 'tot_vids_to_render.json')

if __name__ == '__main__':

    os.system("Xvfb :11 -screen 0 640x480x24 &")
    os.environ['DISPLAY'] = ":11"
    #os.system("Xvfb :11 -screen 1 640x480x24 &")

    _render_vids()
