from calendar import c
import os
import sys
import logging
import hydra
from pathlib import Path
from omegaconf import DictConfig
from omegaconf import OmegaConf
import numpy as np
import glob

from src.render.mesh_viz import visualize_meshes
from src.render import render_animation
from src.render.anim import render_animation
from src.render.video import save_video_samples
import src.launch.prepare  # noqa
from tqdm import tqdm
from src.utils.file_io import read_json
from src.launch.prepare import get_last_checkpoint
import torch

logger = logging.getLogger(__name__)

plt_logger = logging.getLogger("matplotlib.animation")
plt_logger.setLevel(logging.WARNING)
os.environ['PYOPENGL_PLATFORM'] = 'egl'

@hydra.main(config_path="configs", config_name="demo")
def _render(cfg: DictConfig) -> None:
    return render(cfg)


def fix_config_if_needed(cfg):
    if 'gpt_path' not in cfg.data:
        cfg.data['gpt_path'] = '${path.deps}/gpt/gpt3-labels-list.json'
    else:
        cfg.data['gpt_path'] = '${path.deps}/gpt/gpt3-labels-list.json'

def load_temos(cfg):
    from hydra.utils import instantiate
    temos_path = Path(cfg.temos_path)
    temoscfg = OmegaConf.load(temos_path / ".hydra/config.yaml")

    # Overload it
    logger.info("Loading TEMOS model")
    # Instantiate all modules specified in the configs
    temos_model = instantiate(temoscfg.model,
                              nfeats=135,
                              logger_name="none",
                              nvids_to_save=None,
                              _recursive_=False)

    last_ckpt_path = temos_path / "checkpoints/last.ckpt"
    # Load the last checkpoint
    temos_model = temos_model.load_from_checkpoint(last_ckpt_path)
    temos_model.eval()
    logger.info("TEMOS Model weights restored")
    return temos_model, temoscfg


def compute_scores(cfg, gen_samples, set_to_compute, temos_model,
                   temoscfg, on_the_fly=False):
    from hydra.utils import instantiate
    from src.model.metrics import ComputeMetricsSinc
    from src.utils.eval_utils import sanitize, regroup_metrics
    
    
    rots2joints = instantiate(temoscfg.transforms.rots2joints, jointstype='smplh')
    CMetrics = ComputeMetricsSinc(jointstype='smplh')
 
    metrics = []
    import glob
    gt_samples = Path(cfg.path_to_gt)

    tm_scores = {}
    pos_scores = {}

    for keyid in set_to_compute:
        if on_the_fly:
            ds_gen = gen_samples
        else:
            data_gen = np.load(gen_samples / (keyid + '.npy'), allow_pickle=True).item()
            motion_R = torch.from_numpy(data_gen['rots'])
            motion_t = torch.from_numpy(data_gen['trans'])
            from space.transforms.smpl import RotTransDatastruct
            ds_gen = RotTransDatastruct(rots=motion_R, trans=motion_t)
        
        data_ref = np.load(gt_samples / (keyid + '.npy'), allow_pickle=True).item()
        motion_R = torch.from_numpy(data_ref['rots'])
        motion_t = torch.from_numpy(data_ref['trans'])
        from space.transforms.smpl import RotTransDatastruct
        ds_ref = RotTransDatastruct(rots=motion_R, trans=motion_t)

        distribution_ref = temos_model.motionencoder(torch.squeeze(temos_model.transforms.rots2rfeats(ds_ref))[None])
        distribution_motion = temos_model.motionencoder(torch.squeeze(temos_model.transforms.rots2rfeats(ds_gen))[None])

        mu_ref = distribution_ref.mean[0,0]
        mu_motion = distribution_motion.mean[0,0]

        # dist = torch.linalg.norm(mu_motion-mu_ref)
        metric = 2*(1-torch.nn.CosineSimilarity()(mu_motion[None], mu_ref[None]))
        tm_score = metric.detach().cpu().numpy()[0]
        tm_scores[keyid] = 1 - tm_score/4

        ref_joints = rots2joints(ds_ref)
        gen_joints = rots2joints(ds_gen).squeeze()
        # Take the common lengths to facilitate the computation
        length = min(len(gen_joints), len(ref_joints))
        CMetrics.update(gen_joints[None], ref_joints[None], [length])
        metrics = sanitize(regroup_metrics(CMetrics.compute(mode='test')))
        pos_scores[keyid] = list(metrics.items())[:8]
    pos_scores = {k.replace('spatial_pairs-', ''): round(float(v[0][1]), 4) for k,v in pos_scores.items()}
    tm_scores = {k.replace('spatial_pairs-', ''): round(v, 4) for k,v in tm_scores.items()}
    return pos_scores, tm_scores


def render(newcfg: DictConfig) -> None:
    from pathlib import Path

    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")

    fix_config_if_needed(prevcfg)
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)

    cfg.number_of_samples = 1

    logger.info("Sample script. The outputs will be stored in:")

    path = output_dir / 'metrics'
    path.mkdir(exist_ok=True, parents=True)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    seed_logger = logging.getLogger("pytorch_lightning.utilities.seed")
    seed_logger.setLevel(logging.WARNING)

    pl.seed_everything(cfg.seed)
    logger.info("Loading data module")
    # only pair evaluation to be fair
    # keep same order

    from tqdm import tqdm

    logger.info("Loading model")
    # Instantiate all modules specified in the configs
    model = instantiate(cfg.model,
                        nfeats=135,
                        logger_name="none",
                        nvids_to_save=None,
                        _recursive_=False)

    logger.info(f"Model '{cfg.model.modelname}' loaded")

    # Load the last checkpoint
    model = model.load_from_checkpoint(last_ckpt_path)
    model.eval()
    logger.info("Model weights restored")
    model.sample_mean = cfg.mean
    model.fact = cfg.fact

    if not model.hparams.vae and cfg.number_of_samples > 1:
        raise TypeError("Cannot get more than 1 sample if it is not a VAE.")

    # trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))

    logger.info("Trainer initialized")
    if 'rotsd+vertices':
        model.transforms.rots2joints.jointstype = 'vertices'
    else:
        model.transforms.rots2joints.jointstype = cfg.jointstype

    # ds = model.transforms.Datastruct
    if cfg.set == 'submission':
        from src.utils.inference import sinc_eval_set
        keyids = sinc_eval_set
    elif cfg.set == 'small':
        from src.utils.inference import validation_nostand_notrain
        keyids = validation_nostand_notrain
    elif cfg.set == 'supmat':
        from src.utils.inference import sinc_supmat
        keyids = sinc_supmat
    elif cfg.set == 'ood3':
        from src.utils.inference import sinc_ood_three
        keyids = sinc_ood_three
    elif cfg.set == 'ood2':
        from src.utils.inference import sinc_ood_2
        keyids = sinc_ood_2
    elif cfg.set == 'single':
        from src.utils.inference import sinc_single
        keyids = sinc_single
    elif cfg.set == 'oodgpt':
        from src.utils.inference import sinc_ood_gptfail
        from src.tools.frank import text_list_to_bp
        gppt_path = './deps/gpt/gpt3-labels-list.json'
        gpt_labels = read_json(gppt_path)

        keyids = sinc_ood_gptfail


    motion_type = cfg.jointstype #"vertices"
    import numpy as np
    savedir = Path(cfg.savedir)
    logger.info(f'Saving the on:{str(savedir)}')
    savedir.mkdir(exist_ok=True, parents=True)

    temos_model, temoscfg = load_temos(cfg)
    ape_dict = {}
    temos_dict = {}

    with torch.no_grad():
        with tqdm(total=len(keyids), position=0, leave=True) as pbar:

            for keyid in (pbar := tqdm(keyids,  position=0, leave=True)):
                
                pbar.set_description(f"Processing {keyid}")      

                cur_lens = [keyid[1]]
                cur_texts = [list(keyid[0])]
                is_spatial_pair = True
                acts = []

                for a in cur_texts[0]:
                    acts.append(a.replace(' ', '-')) 
                    
                fname = '__'.join(acts)
                fname = fname + '___fr' + str(keyid[1])
                vid_path = savedir / fname
                is_sp = True
                
                
                logger.info(f"Loging at {vid_path}")
                # batch_size = 1 for reproductability
                # fix the seed
                pl.seed_everything(0)
                try:
                    if cfg.naive == 'gpt':
                        gpt_1 = text_list_to_bp(cur_texts[0][0], gpt_labels)
                        gpt_2 = text_list_to_bp(cur_texts[0][1], gpt_labels)
                        gpt_parts = [gpt_1, gpt_2]
                    else:
                        gpt_parts = None
                except AttributeError:
                    gpt_parts = None

                if is_sp and cfg.naive == "gpt":
                    from space.tools.frank import combine_motions
                    # gpt_parts = batch['bp-gpt'][0]

                    motion1, verts1 = model.text_to_motion_forward([[cur_texts[0][0]]],
                                                        cur_lens,
                                                        gpt_parts=None,
                                                        return_motion=motion_type)

                    motion2, verts2 = model.text_to_motion_forward([[cur_texts[0][1]]],
                                                        cur_lens,
                                                        gpt_parts=None,
                                                    return_motion=motion_type)


                    # rots and transl
                    frank_motion = combine_motions(motion1, motion2, gpt_parts[0], gpt_parts[1], squeeze=True)
                    frank_datastruct = model.Datastruct(rots_=frank_motion)

                    motion_ds, vertices = model.motion_from_datastruct(frank_datastruct, return_type=motion_type)

                    # just in case
                    motion1 = model.motion_from_datastruct(model.Datastruct(rots_=motion1), return_type=motion_type)
                    motion2 = model.motion_from_datastruct(model.Datastruct(rots_=motion2), return_type=motion_type)
                elif is_sp and cfg.naive == "concat":
                    concat_text = [[" while ".join(cur_texts[0])]]
                    motion_ds, vertices = model.text_to_motion_forward(concat_text,
                                                        cur_lens,
                                                        gpt_parts=gpt_parts,
                                                        return_motion=motion_type)

                else:
                    import ipdb; ipdb.set_trace()
                    motion_ds, vertices = model.text_to_motion_forward(cur_texts,
                                                            cur_lens,
                                                            gpt_parts=gpt_parts,
                                                            return_motion=motion_type)
                text = '__'.join(cur_texts[0])

                if 'ood' not in cfg.set:
                    ape_score = 'XX'
                    temos_score = 'XX'
                    if cfg.set != 'single':
                        ape, temos = compute_scores(cfg, motion_ds, [keyid], temos_model,
                                                    temoscfg, on_the_fly=True)
                        
                        # scores for single sample
                        ape_score = ape[keyid.replace('spatial_pairs-', '')]
                        temos_score = temos[keyid.replace('spatial_pairs-', '')]
                        ape_dict[keyid] = ape_score

                        temos_dict[keyid] = temos_score 

                    if not cfg.only_score:
                        vid_ = visualize_meshes(vertices.squeeze().detach().cpu().numpy())
                        vid_p = save_video_samples(vid_,
                                    f'{vid_path.resolve()}___APE_{ape_score}_TM_{temos_score}.mp4',
                                    text,
                                    fps=30)

                else:

                    vid_ = visualize_meshes(vertices.squeeze().detach().cpu().numpy())
                    vid_p = save_video_samples(vid_,
                                    f'{vid_path.resolve()}.mp4',
                                    text,
                                    fps=30)

                if cfg.naive and not cfg.only_score:
                    vid_ = visualize_meshes(verts1.squeeze().detach().cpu().numpy())
                    vid1_p = save_video_samples(vid_,
                                    f'{vid_path.resolve()}-motion1.mp4',
                                    text[0],
                                    fps=30)
                    vid_ = visualize_meshes(verts2.squeeze().detach().cpu().numpy())
                    vid2_p = save_video_samples(vid_,
                                    f'{vid_path.resolve()}-motion2.mp4',
                                    text[1],
                                    fps=30)
                    from space.render.video import stack_vids_moviepy
                    stack_vids_moviepy([vid_p, vid1_p, vid2_p], f'{vid_path.resolve()}-stacked.mp4')
    from src.utils.file_io import write_json
    import shortuuid 
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    hashcd = str(run_gen.random(6))
    ape_dict["PATH"] = '____'.join(cfg.folder.split('/')[-3:-1])
    temos_dict["PATH"] = '____'.join(cfg.folder.split('/')[-3:-1])
    write_json(ape_dict, savedir / f'ape{hashcd}.json')
    write_json(temos_dict, savedir / f'temos{hashcd}.json')

if __name__ == '__main__':
    _render()
