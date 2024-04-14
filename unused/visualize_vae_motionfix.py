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

# from src.render.mesh_viz import visualize_meshes
from src.render.video import save_video_samples
import src.launch.prepare  # noqa
from tqdm import tqdm
import torch
import itertools
from src.model.utils.tools import pack_to_render
from src.utils.eval_utils import split_txt_into_multi_lines

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="motionfix_viz_vae")
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

    fd_name = 'samples_vae'
    log_name = '__'.join(str(exp_folder).split('/')[-2:])
    log_name = f'{log_name}_{cfg.ckpt_name}'
    output_path = exp_folder / fd_name
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
    
    # wandb.init(project="motion-edit-eval", job_type="evaluate",
    #            name=log_name, dir=output_path)

    # notes: ''
    # id: ${run_hash}
    # mode: "online"
    # # project: 'CVPR24-Humans'
    # dir: '.'

    logger.info("Loading model")
    model = instantiate(cfg.model,
                        renderer=aitrenderer,
                        _recursive_=False).eval()

    logger.info(f"Model '{cfg.model.modelname}' loaded")
    
    # Load the last checkpoint
    model = model.load_from_checkpoint(last_ckpt_path,
                                       renderer=aitrenderer,
                                       strict=False)
    model.freeze()
    logger.info("Model weights restored")
    logger.info("Trainer initialized")

    import numpy as np

    data_module = instantiate(cfg.data)

    transl_feats = [x for x in model.input_feats if 'transl' in x]
    if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                   "body_transl_delta_pelv_xy"]):
        model.using_deltas_transl = True

    # load the test set and collate it properly
    test_dataset = data_module.dataset['test']
    features_to_load = test_dataset.load_feats
    from src.data.tools.collate import collate_batch_last_padding
    collate_fn = lambda b: collate_batch_last_padding(b,
                                                        features_to_load)
    batch_size_test = 8
    test_dataset.data = test_dataset.data[:batch_size_test*4]

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             shuffle=False,
                                             num_workers=0,
                                             batch_size=batch_size_test,
                                             collate_fn=collate_fn)
    ds_iterator = testloader

    from src.utils.art_utils import color_map
    tot_pkls = []
    from aitviewer.models.smpl import SMPLLayer
    smpl_layer = SMPLLayer(model_type='smplh', ext='npz', gender='neutral')

    with torch.no_grad():
        output_path = output_path / 'renders'
        output_path.mkdir(exist_ok=True, parents=True)
        for batch in tqdm(ds_iterator):
            batch = { k: v.to(model.device) if torch.is_tensor(v) else v
            for k, v in batch.items() }

            input_batch = model.norm_and_cat(batch, model.input_feats)
            for k, v in input_batch.items():
                batch[f'{k}_motion'] = v

            src, tgt = model.fix_input_for_tmr(batch)

            if model.motion_condition:
                mask_source, mask_target = model.prepare_mot_masks(batch['length_source'],
                                                                batch['length_target'],
                                                                max_len=None)
            else:
                mask_target = lengths_to_mask(batch['length_target'],
                                            device=model.device)

                batch['length_source'] = None
                batch['source_motion'] = None
                mask_source = None

            gt_lens_tgt = batch['length_target']
            gt_lens_src = batch['length_source']
            gt_texts = batch['text']
            gt_keyids = batch['id']

            diffout = model.generate_motion(gt_texts,
                                            src,
                                            gt_lens_src,
                                            gt_lens_tgt,
                                            mask_source,
                                            mask_target)
            # gen_mo = model.diffout2motion(diffout)
            
            src_mot_cond, tgt_mot = model.batch2motion(input_batch,
                                            pack_to_dict=False)
            tgt_mot = tgt_mot.to(model.device)

            src_mot_cond = src_mot_cond.to(model.device)
            mots_to_render = [src_mot_cond, tgt_mot, 
                                [src_mot_cond, tgt_mot],
                                [tgt_mot, gen_mo]]
            monames = ['source', 'target', 'overlaid', 
                        'generated']

            lof_mots = output2renderable(model,
                                        mots_to_render)

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
                            if monames[moid] == 'generated':
                                cur_colors = [color_map['target'],
                                            color_map['generated']]
                            else:
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
                                        color=cur_colors,
                                        smpl_layer=smpl_layer)
                    cur_group_of_vids.append(fname)
                stacked_vid = stack_vids(cur_group_of_vids,
                                        f'{output_path}/{elem_id}_stacked.mp4',
                                        orient='h')
                text_wrap = split_txt_into_multi_lines(text_diff[elem_id],
                                                        40)
                fnal_fl = put_text(text=text_wrap.replace("'", " "),
                                    fname=stacked_vid, 
                                    outf=f'{output_path}/{curid}_text.mp4',
                                    position='top_center')

                cleanup_files(cur_group_of_vids+[stacked_vid])
                video_key = fnal_fl.split('/')[-1].replace('.mp4','')
                wandb.log({f"{cur_guid_comb}/{video_key}":
                                wandb.Video(fnal_fl, fps=30, format="mp4")
                            })


    from src.utils.file_io import write_json   
    write_json(tot_pkls, output_path / 'tot_vids_to_render.json')
    
if __name__ == '__main__':

    os.system("Xvfb :11 -screen 0 640x480x24 &")
    os.environ['DISPLAY'] = ":11"
    #os.system("Xvfb :11 -screen 1 640x480x24 &")

    _render_vids()
