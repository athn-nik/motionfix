import os
import logging
from cv2 import exp
import hydra
import joblib
from omegaconf import DictConfig
from omegaconf import OmegaConf
from sympy import O
from src import data
from src.render.mesh_viz import render_motion
from torch import Tensor
import sys
# from src.render.mesh_viz import visualize_meshes
from src.render.video import save_video_samples
import src.launch.prepare  # noqa
from tqdm import tqdm
import torch
import itertools
from src.model.utils.tools import pack_to_render
from src.utils.eval_utils import split_txt_into_multi_lines
from src.utils.file_io import write_json, read_json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)
os.environ["WANDB__SERVICE_WAIT"] = "300"


import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
from tqdm import tqdm

def render_video(data):
    elem_id, monames, lof_mots, lens_to_mask, color_map, output_path, text_diff, aitrenderer, smpl_layer, keyid = data
    cur_group_of_vids = []
    curid = keyid
    for moid, one_motion in enumerate(lof_mots):
        cur_mol = []
        cur_colors = []
        crop_len = lens_to_mask[moid]

        if isinstance(one_motion, list):
            for motion in one_motion:
                cur_mol.append({k: v[elem_id][:crop_len[elem_id]] for k, v in motion.items()})
                if monames[moid].endswith('_source'):
                    cur_colors = [color_map['source'], color_map['generated']]
                elif monames[moid].endswith('_target'):
                    cur_colors = [color_map['target'], color_map['generated']]
                elif monames[moid] == 'overlaid_GT':
                    cur_colors = [color_map['source'], color_map['target']]
        else:
            cur_mol.append({k: v[elem_id][:crop_len[elem_id]] for k, v in one_motion.items()})
            cur_colors.append(color_map[monames[moid]])

        fname = render_motion(aitrenderer, cur_mol,
                              output_path / f"movie_{elem_id}_{moid}",
                              pose_repr='aa', text_for_vid=monames[moid],
                              color=cur_colors, smpl_layer=smpl_layer)
        cur_group_of_vids.append(fname)

    stacked_vid = stack_vids(cur_group_of_vids, f'{output_path}/{elem_id}_stacked.mp4', orient='3x3')
    text_wrap = split_txt_into_multi_lines(text_diff[elem_id], 40)
    final_file = put_text(text=text_wrap.replace("'", " "), fname=stacked_vid,
                          outf=f'{output_path}/{curid}_text.mp4', position='top_center')
    cleanup_files(cur_group_of_vids + [stacked_vid])
    return final_file, elem_id, len(text_diff[elem_id].split()) <= 5


@hydra.main(config_path="configs", config_name="mfix_viz")
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
        if os.path.exists(fl):
            os.remove(fl)

def output2renderable(lst_of_tensors: list[Tensor]):
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
        init_from = '_src_init_'
    else:
        init_from = ''
    if config.ckpt_name == 'last':
        ckpt_n = ''
    else:
        ckpt_n = f'_ckpt-{config.ckpt_name}_'
    if config.subset is None:
        sset = ''
    else:
        sset = f'{config.subset}'

    return f'{sset}{ckpt_n}{init_from}{sched_name}_mot{mot_guid}_text{text_guid}_steps{infer_steps}'


def render_vids(newcfg: DictConfig) -> None:
    from pathlib import Path
    
    exp_folder = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")
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
    if cfg.linear_gd:
        use_linear_guid = True
        gd_str = 'lingd_'
    else:
        use_linear_guid = False
        gd_str = ''

    # cfg.model.infer_scheduler = newcfg.model.infer_scheduler
    # cfg.model.diff_params.num_inference_timesteps = newcfg.steps
    # cfg.model.diff_params.guidance_scale_motion = newcfg.guidance_scale_motion
    # cfg.model.diff_params.guidance_scale_text = newcfg.guidance_scale_text
    init_diff_from = cfg.init_from
    if cfg.inpaint:
        assert cfg.data.dataname == 'sinc_synth'
        annots_sinc = read_json('data/sinc_synth/for_website_v4.json')


    fd_name = f'steps_{num_infer_steps}'
    if cfg.inpaint:
        log_name = f'{gd_str}{cfg.data.dataname}_steps-{num_infer_steps}_{cfg.init_from}_{cfg.ckpt_name}_inpaint_bsl'
    else:
        log_name = f'{gd_str}{cfg.data.dataname}_steps-{num_infer_steps}_{cfg.init_from}_{cfg.ckpt_name}'
    last_two_dirs = Path(*exp_folder.parts[-2:])
    exp_str = str(last_two_dirs).replace('hml_3d', 'h3d').replace('sinc_synth', 'syn').replace('bodilex', 'B').replace('/', '__')
    log_name = f'{exp_str.upper()}_{log_name}'
    output_path = exp_folder / log_name
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
    
    wandb.init(project="motionfix-visuals", job_type="evaluate",
               name=log_name, dir=output_path)

    #############################################################
    import numpy as np
    data_module = instantiate(cfg.data)
    from src.model.base_diffusion import MD    
    # Load the last checkpoint
    list_of_all_feats = data_module.nfeats
    cfg.model.input_feats = cfg.data.load_feats
    idx_for_inputs = [cfg.data.load_feats.index(infeat) 
                      for infeat in cfg.model.input_feats]
    total_feats_dim = [list_of_all_feats[i] 
                      for i in idx_for_inputs]
    nfeats = sum(total_feats_dim) 

    cfg.model.nfeats = nfeats
    cfg.model.dim_per_feat = total_feats_dim
    #############################################################
    logger.info("Loading model")

    model = instantiate(cfg.model,
                        renderer=None, _recursive_=False)
    # MD.load_from_checkpoint(last_ckpt_path,
    #                                    renderer=aitrenderer,
    #                                 #    infer_scheduler=cfg.model.infer_scheduler,
    #                                 #    diff_params=cfg.model.diff_params,
    #                                    strict=False)
    model.freeze()
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    # logger.info('------Generating using Scheduler------\n\n'\
    #             f'{model.infer_scheduler}')
    logger.info('------Diffusion Parameters------\n\n'\
                f'{model.diff_params}')

    transl_feats = [x for x in model.input_feats if 'transl' in x]
    if set(transl_feats).issubset(["body_transl_delta", 
                                   "body_transl_delta_pelv",
                                   "body_transl_delta_pelv_xy"]):
        model.using_deltas_transl = True

    # load the test set and collate it properly
    test_dataset = data_module.dataset['test']
    features_to_load = test_dataset.load_feats
    from src.data.tools.collate import collate_batch_last_padding, collate_tensor_with_padding
    collate_fn = lambda b: collate_batch_last_padding(b, features_to_load)
    if cfg.data.dataname == 'sinc_synth':
        # from src.utils.motionfix_utils import test_subset_sinc_synth
        # test_dataset_subset = test_dataset[:128]
        # batch_to_use = 128
        def keep_N(dics, N=64):
           keys = set()
           for elem in dics:
               if not elem['id'].endswith(('_0', '_1', '_2', '_3')):
                   continue
               if len(keys) < N:
                   keys.add(elem['id'])
               else:
                   break
           return list(keys)
        ktokeep = keep_N(test_dataset)
        test_dataset_subset = [elem for elem in test_dataset
                            if elem['id'] in ktokeep]
        extra_keys = [elem['id'] for elem in test_dataset_subset]

        batch_to_use = len(ktokeep)
    elif cfg.data.dataname == 'bodilex':
        counter_short = 24
        from src.utils.motionfix_utils import test_subset_amt
        test_dataset_subset = [elem for elem in test_dataset
                         if elem['id'] in test_subset_amt[:1]]
        test_dataset_subset2 = []
        for elem in test_dataset:
            if len(elem['text'].split()) <= 5 and counter_short > 0:
                test_dataset_subset2.append(elem)
                counter_short -= 1
        test_dataset_subset.extend(test_dataset_subset2)
        extra_keys = [elem['id'] for elem in test_dataset_subset]
        batch_to_use = len(test_dataset_subset)

    
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             shuffle=False,
                                             num_workers=4,
                                             batch_size=128,
                                             collate_fn=collate_fn)

    from src.utils.art_utils import color_map
    from aitviewer.models.smpl import SMPLLayer
    smpl_layer = SMPLLayer(model_type='smplh', ext='npz', gender='neutral')
    # Get a list of all items in the directory
    all_dirnames = os.listdir(cfg.samples)

    # Filter the list to include only directories
    guid_fds = [os.path.join(cfg.samples, item) 
                for item in all_dirnames]
    for x in guid_fds:
        assert os.path.exists(x)

    def render_and_stack_videos(data):
        elem_id, keyid, monames, lof_mots, lens_to_mask, aitrenderer, smpl_layer, output_path, text_diff, color_map = data
        cur_group_of_vids = []
        for moid, moname in enumerate(monames):
            cur_mol = []
            cur_colors = []
            one_motion = lof_mots[moid]
            crop_len = lens_to_mask[moid][elem_id]
            if isinstance(one_motion, list):
                for motion in one_motion:
                    cur_mol.append({k: v[elem_id][:crop_len] for k, v in motion.items()})
                    if moname == 'generated_vs_source':
                        cur_colors = [color_map['source'], color_map['generated']]
                    elif moname == 'generated_vs_target':
                        cur_colors = [color_map['target'], color_map['generated']]
                    elif moname == 'overlaid_GT':
                        cur_colors = [color_map['source'], color_map['target']]
            else:
                cur_mol.append({k: v[elem_id][:crop_len] for k, v in one_motion.items()})
                cur_colors.append(color_map[moname])

            fname = render_motion(aitrenderer, cur_mol, output_path / f"movie_{elem_id}_{moid}", pose_repr='aa', text_for_vid=moname, color=cur_colors, smpl_layer=smpl_layer)
            cur_group_of_vids.append(fname)

        stacked_vid = stack_vids(cur_group_of_vids, f'{output_path}/{elem_id}_stacked.mp4', orient='3x3')
        text_wrap = split_txt_into_multi_lines(text_diff[elem_id], 40)
        final_fl = put_text(text=text_wrap.replace("'", " "), fname=stacked_vid, outf=f'{output_path}/{keyid}_text.mp4', position='top_center')
        cleanup_files(cur_group_of_vids + [stacked_vid])
        return final_fl, keyid, len(text_diff[elem_id].split()) <= 5

    with torch.no_grad():
        output_path = output_path / 'renders'
        output_path.mkdir(exist_ok=True, parents=True)
        tasks = []
        for path_to_fd in guid_fds:
            cur_guid_comb = Path(path_to_fd).name
            all_keys = read_json(f'{path_to_fd}/all_candkeyids.json')
            batch_keys = read_json(f'{path_to_fd}/guo_candkeyids.json')
            keys_to_rend = set(all_keys + batch_keys + extra_keys)
            for batch in tqdm(testloader):
                idx_to_rend = [i for i, x in enumerate(batch['id']) if x in keys_to_rend]
                if not idx_to_rend:
                    continue
                batch = {k: v.index_select(0, torch.tensor(idx_to_rend)) if isinstance(v, torch.Tensor) else [v[i] for i in idx_to_rend] for k, v in batch.items()}
                src_mot_cond, tgt_mot = model.batch2motion(prepare_test_batch(model, batch), pack_to_dict=False)
                gen_mo = [torch.from_numpy(
                                  np.load(f'{path_to_fd}/{key}.npy',
                                  allow_pickle=True).item()['pose'])
                          for key in batch['id']]
                gen_mo = collate_tensor_with_padding(gen_mo)

                monames = ['source', 'target', 'overlaid_GT', 'generated', 'generated_vs_target', 'generated_vs_source']
                lof_mots = output2renderable([src_mot_cond, tgt_mot, [src_mot_cond, tgt_mot], gen_mo, [tgt_mot, gen_mo], [src_mot_cond, gen_mo]])
                lens_of_mots = [batch['length_source'], batch['length_target']] * 3
                crop_lens = [max(a, b) for a, b in zip(batch['length_target'], batch['length_source'])]
                lens_to_mask = [crop_lens] * 6
                for elem_id, keyid in enumerate(batch['id']):
                    tasks.append((elem_id, keyid, monames, lof_mots, lens_to_mask, aitrenderer, smpl_layer, output_path, batch['text'], color_map))

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(render_and_stack_videos, tasks))

        for final_fl, keyid, is_short in results:
            video_key = final_fl.split('/')[-1].replace('.mp4', '')
            short = '_short' if is_short else ''
            wandb.log({f"{cur_guid_comb}{short}/{video_key}": wandb.Video(final_fl, fps=30, format="mp4")})


    # with torch.no_grad():
    #     output_path = output_path / 'renders'
    #     output_path.mkdir(exist_ok=True, parents=True)
    #     for path_to_fd in guid_fds:
    #         cur_guid_comb = Path(path_to_fd).name
    #         all_keys = read_json(f'{path_to_fd}/all_candkeyids.json')
    #         batch_keys = read_json(f'{path_to_fd}/guo_candkeyids.json')
    #         keys_to_rend = list(set(all_keys + batch_keys + extra_keys))
    #         for batch in tqdm(testloader):
    #             idx_to_rend = [i for i, x in enumerate(batch['id']) 
    #                            if x in keys_to_rend]
    #             if not idx_to_rend:
    #                 continue
    #             for k, v in batch.items():
    #                 if isinstance(v, torch.Tensor):
    #                     batch[k] = v.index_select(0, torch.tensor(idx_to_rend))
    #                 else:
    #                     batch[k] = [v[candix] for candix in idx_to_rend]
    #             target_lens = batch['length_target']
    #             source_lens = batch['length_source']
    #             keyids = batch['id']
    #             text_diff = batch['text']
    #             no_of_motions = len(keyids)
    #             input_batch = prepare_test_batch(model, batch)
    #             gen_mo = [torch.from_numpy(
    #                               np.load(f'{path_to_fd}/{key}.npy',
    #                               allow_pickle=True).item()['pose'])
    #                       for key in keyids]
    #             gen_mo = collate_tensor_with_padding(gen_mo)
    #             src_mot_cond, tgt_mot = model.batch2motion(input_batch,
    #                                             pack_to_dict=False)
    #             mots_to_render = [src_mot_cond, tgt_mot, 
    #                               [src_mot_cond, tgt_mot],
    #                               gen_mo,
    #                               [tgt_mot, gen_mo], 
    #                               [src_mot_cond, gen_mo]]
    #             monames = ['source', 'target', 'overlaid_GT', 'generated',
    #                        'generated_vs_target', 'generated_vs_source']
    #             lens_of_mots = [source_lens, target_lens, 
    #                                 [source_lens, target_lens],
    #                                 target_lens,
    #                                 [target_lens, target_lens],
    #                                 [source_lens, target_lens]]
    #             lof_mots = output2renderable(mots_to_render)
    #             crop_lens = [max(a, b) for a, b in zip(target_lens, source_lens)]
    #             lens_to_mask = [crop_lens, crop_lens, crop_lens,
    #                             crop_lens, crop_lens, crop_lens]
    #             for elem_id in range(no_of_motions):
    #                 cur_group_of_vids = []
    #                 curid = keyids[elem_id]
    #                 for moid in range(len(monames)):
    #                     one_motion = lof_mots[moid]
    #                     cur_mol = []
    #                     cur_colors = []
    #                     if lens_to_mask[moid] is not None:
    #                         crop_len = lens_to_mask[moid]

    #                     if isinstance(one_motion, list):
    #                         for xx in one_motion:
    #                             cur_mol.append({k: v[elem_id][:crop_len[elem_id]]
    #                                             for k, v in xx.items()})
    #                             if monames[moid] == 'generated_vs_source':
    #                                 cur_colors = [color_map['source'],
    #                                               color_map['generated']]
    #                             elif monames[moid] == 'generated_vs_target':
    #                                 cur_colors = [color_map['target'],
    #                                               color_map['generated']]
    #                             elif monames[moid] == 'overlaid_GT':
    #                                 cur_colors = [color_map['source'],
    #                                               color_map['target']]
    #                     else:
    #                         cur_mol.append({k: v[elem_id][:crop_len[elem_id]]
    #                                             for k, v in one_motion.items()})
    #                         cur_colors.append(color_map[monames[moid]])

    #                     fname = render_motion(aitrenderer, cur_mol,
    #                                         output_path / f"movie_{elem_id}_{moid}",
    #                                         pose_repr='aa',
    #                                         text_for_vid=monames[moid],
    #                                         color=cur_colors,
    #                                         smpl_layer=smpl_layer)
    #                     cur_group_of_vids.append(fname)
    #                 stacked_vid = stack_vids(cur_group_of_vids,
    #                                         f'{output_path}/{elem_id}_stacked.mp4',
    #                                         orient='3x3')
    #                 text_wrap = split_txt_into_multi_lines(text_diff[elem_id],
    #                                                         40)
    #                 fnal_fl = put_text(text=text_wrap.replace("'", " "),
    #                                    fname=stacked_vid, 
    #                                    outf=f'{output_path}/{curid}_text.mp4',
    #                                    position='top_center')
    #                 cleanup_files(cur_group_of_vids+[stacked_vid])
    #                 video_key = fnal_fl.split('/')[-1].replace('.mp4','')
    #                 if len(text_diff[elem_id].split()) <= 5:
    #                     short = '_short'
    #                 else:
    #                     short = ''
    #                 wandb.log({f"{cur_guid_comb}{short}/{video_key}":
    #                                 wandb.Video(fnal_fl, fps=30, format="mp4")
    #                             })


    
if __name__ == '__main__':

    os.system("Xvfb :11 -screen 0 640x480x24 &")
    os.environ['DISPLAY'] = ":11"
    #os.system("Xvfb :11 -screen 1 640x480x24 &")

    _render_vids()
