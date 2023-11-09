import os
import logging
import hydra
import joblib
from omegaconf import DictConfig
from omegaconf import OmegaConf
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


@hydra.main(config_path="configs", config_name="demo")
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
    cond_type = '_' + config.condition_mode + '_'
    if config.render_vids:
        vds = 'gens_'
    else:
        vds = ''

    if config.model.motion_condition is not None:
        return f'{vds}{cond_type}{sset}{ckpt_n}{init_from}{sched_name}_mot{mot_guid}_text{text_guid}_steps{infer_steps}'
    else:
        return f'{vds}{cond_type}{sset}{ckpt_n}{init_from}{sched_name}_text{text_guid}_steps{infer_steps}'


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

    fd_name = get_folder_name(cfg)
    output_path = exp_folder / cfg.mode / fd_name
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
    if cfg.render_vids:
        from aitviewer.headless import HeadlessRenderer
        from aitviewer.configuration import CONFIG as AITVIEWER_CONFIG
        AITVIEWER_CONFIG.update_conf({"playback_fps": 30,
                                    "auto_set_floor": True,
                                    "smplx_models": 'data/body_models',
                                    'z_up': True})
        aitrenderer = HeadlessRenderer()
    else:
        aitrenderer = None

    logger.info("Loading model")
    model = instantiate(cfg.model,
                        renderer=aitrenderer,
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

    data_module = instantiate(cfg.data)
    transl_feats = [x for x in model.input_feats if 'transl' in x]
    if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                   "body_transl_delta_pelv_xy"]):
        model.using_deltas_transl = True

    if cfg.mode == 'demo':
        lengths = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

        texts = ['slower', 'faster']
        len_text = [list(tup) for tup in list(itertools.product(lengths, texts))]
        idx = 0
        ds_iterator = chunker(len_text, 8)

    else:
        # load the test set and collate it properly
        test_dataset = data_module.dataset['test']
        features_to_load = test_dataset.load_feats
        from src.data.tools.collate import collate_batch_last_padding
        collate_fn = lambda b: collate_batch_last_padding(b,
                                                          features_to_load)

        if cfg.subset == 'cherries':
            from src.utils.eval_utils import test_keyds

            subset = []
            for elem in test_dataset.data:
                if elem['id'] in test_keyds:
                    subset.append(elem)
            batch_size_test = len(subset)
            test_dataset.data = subset
        elif cfg.subset == 'cherries2':
            from src.utils.eval_utils import keyids_for_testing
            subset = []
            for elem in test_dataset.data:
                if elem['id'] in keyids_for_testing:
                    subset.append(elem)
            batch_size_test = len(subset)
            test_dataset.data = subset

        else:
            batch_size_test = 8

        testloader = torch.utils.data.DataLoader(test_dataset,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 batch_size=batch_size_test,
                                                 collate_fn=collate_fn)
        ds_iterator = testloader 
    from src.utils.art_utils import color_map
    
    init_diff_from = cfg.init_from
    mode_cond = cfg.condition_mode
    tot_pkls = []

    if cfg.mode in ['denoise', 'sample']:
        with torch.no_grad():
            for batch in tqdm(ds_iterator):
     
                text_diff = batch['text']
                target_lens = batch['length_target']
                keyids = batch['id']
                no_of_motions = len(keyids)
                batch = prepare_test_batch(model, batch)
                if model.motion_condition == 'source':
                    source_lens = batch['length_source']
                    mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                       target_lens)
                else:
                    from src.data.tools.tensors import lengths_to_mask
                    mask_target = lengths_to_mask(target_lens,
                                                  model.device)
                    batch['source_motion'] = None
                    mask_source = None

                if cfg.mode == 'denoise':
                    diffout = model.denoise_forward(batch, mask_source,
                                                    mask_target)
                    in_mot = model.diffout2motion(diffout['input_motion_feats'])
                    timesteps = diffout['timesteps']
                    no_mo = model.diffout2motion(diffout['noised_motion_feats'])
                    deno_mo = model.diffout2motion(diffout['pred_motion_feats'])
                    mots_to_render = [in_mot, no_mo, deno_mo]
                    monames = ['input', 'noised', 'denoised']
                else:
                    source_init = batch['source_motion']
                    diffout = model.generate_motion(text_diff,
                                                    batch['source_motion'],
                                                    mask_source,
                                                    mask_target,
                                                    init_vec=source_init,
                                                    init_vec_method=init_diff_from,
                                                    condition_mode=mode_cond)
                    gen_mo = model.diffout2motion(diffout)

                    src_mot_cond, tgt_mot = model.batch2motion(batch,
                                                    pack_to_dict=False)
                    tgt_mot = tgt_mot.to(model.device)

                    if model.motion_condition is not None:
                        src_mot_cond = src_mot_cond.to(model.device)
                        mots_to_render = [src_mot_cond, tgt_mot, 
                                            [src_mot_cond, tgt_mot], gen_mo]
                        monames = ['source', 'target', 'overlaid', 
                                    'generated']
                    else:
                        mots_to_render = [tgt_mot, gen_mo]
                        monames = ['target', 'generated']

                lof_mots = output2renderable(model,
                                             mots_to_render)
                # output_path = Path('/home/nathanasiou/Desktop/conditional_action_gen/modilex')
                if cfg.save_pkl:
                    for elem_id in range(no_of_motions):
                        curid = keyids[elem_id]
                        for moid in range(len(monames)):
                            one_motion = lof_mots[moid]
                            if isinstance(one_motion, list):
                                continue
                            cur_mol = {k: v[elem_id] 
                                       for k, v in one_motion.items()}
                            from src.utils.eval_utils import out2blender
                            dic_blend = out2blender(cur_mol)
                            pkl_p = f'{output_path}/{monames[moid]}_{curid}.pth.tar'
                            joblib.dump(dic_blend, pkl_p)
                            pkl_p.replace('.pth.tar', '')
                            tot_pkls.append(pkl_p)

                elif cfg.render_vids:
                    for elem_id in range(no_of_motions):
                        cur_group_of_vids = []
                        curid = keyids[elem_id]
                        for moid in range(len(monames)):
                            one_motion = lof_mots[moid]
                            cur_mol = []
                            cur_colors = []
                            if isinstance(one_motion, list):
                                for xx in one_motion:
                                    cur_mol.append({k: v[elem_id] 
                                                    for k, v in xx.items()})
                                cur_colors = [color_map['source'],
                                            color_map['target']]
                            else:
                                cur_mol.append({k: v[elem_id] 
                                                    for k, v in one_motion.items()})
                                cur_colors.append(color_map[monames[moid]])

                            fname = render_motion(aitrenderer, cur_mol,
                                                output_path / f"movie_{elem_id}_{moid}",
                                                pose_repr='aa',
                                                text_for_vid=monames[moid],
                                                color=cur_colors)
                            cur_group_of_vids.append(fname)
                        stacked_vid = stack_vids(cur_group_of_vids,
                                                f'{output_path}/{elem_id}_stacked.mp4',
                                                orient='h')
                        fnal_fl = put_text(text=text_diff[elem_id],
                                        fname=stacked_vid, 
                                        outf=f'{output_path}/{curid}_text.mp4',
                                        position='top_center')

                        cleanup_files(cur_group_of_vids+[stacked_vid])
    else:
        # --> Free for sampling :D <--
        # sample You can do it :)
        with torch.no_grad():
            for batch_len_text in tqdm(chunker(len_text, 8)):
                lengths, texts = zip(*batch_len_text)
            # task: input or Example
            # prepare batch data  
                # model_out = model([text], [length])[0]
                # model_out = model_out.cpu().squeeze().numpy()
                dif_out = model.test_diffusion_forward(list(lengths),
                                                    list(texts))
                if model.input_deltas:
                    motion_unnorm = model.diffout2motion(dif_out)
                    motion_unnorm = motion_unnorm.permute(1, 0, 2)
                else:
                    motion_unnorm = model.unnorm_delta(dif_out)
                batch_motion = pack_to_render(motion_unnorm[..., 3:],
                                        motion_unnorm[..., :3])
                for jj in range(motion_unnorm.shape[0]):
                    one_motion = {k: v[jj] 
                                for k, v in batch_motion.items()
                                }
                    render_motion(aitrenderer, one_motion,
                                output_path / f"movie_{idx}_{jj}",
                                pose_repr='aa')
                idx += 1
    from src.utils.file_io import write_json
    
    write_json(tot_pkls, output_path / 'tot_vids_to_render.json')
    
if __name__ == '__main__':

    os.environ['DISPLAY'] = ":1"
    os.system("Xvfb :11 -screen 1 640x480x24 &")

    _render_vids()
