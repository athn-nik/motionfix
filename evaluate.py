import os
from omegaconf import DictConfig, OmegaConf
import trimesh
from tqdm import tqdm
import pandas as pd
import hydra
import numpy as np
import torch
from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
import wandb
import logging
import joblib
from src.model.base_diffusion import MD
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

def dict_to_device(elems, device):
    return {k: v.to(device) for k, v in elems.items()}

def prepare_test_batch(model, batch):
    batch = { k: v.cuda() if torch.is_tensor(v) else v
                for k, v in batch.items() }

    input_batch = model.norm_and_cat(batch, model.input_feats)
    for k, v in input_batch.items():
        if model.input_deltas:
            batch[f'{k}_motion'] = v[1:]
        else:
            batch[f'{k}_motion'] = v
            batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]

    return batch

def get_folder_name(config):
    sched_name = config.model.infer_scheduler._target_.split('.')[-1]
    sched_name = sched_name.replace('Scheduler', '').lower()
    sched_name = '' if sched_name == 'ddpm' else sched_name
    mot_guid = config.model.diff_params.guidance_scale_motion
    text_guid = config.model.diff_params.guidance_scale_text
    infer_steps = config.model.diff_params.num_inference_timesteps
    if config.init_from == 'source':
        init_from = '_src_init_'
    else:
        init_from = '_noise_init_'
    if config.ckpt_name == 'last':
        ckpt_n = ''
    else:
        ckpt_n = f'_ckpt-{config.ckpt_name}_'
    if config.subset is None or config.subset == 'test':
        sset = ''
    else:
        sset = f'{config.subset}'
    if config.condition_mode == 'full_cond':
        cond_type = ''
    else:
        cond_type = config.condition_mode + '_'
    # if config.render_vids:
    #     vds = 'gens_'
    # else:
    #     vds = ''

    if config.model.motion_condition is not None:
        return f'{cond_type}{sset}{ckpt_n}{init_from}{sched_name}_mot{mot_guid}_text{text_guid}_steps{infer_steps}'
    else:
        return f'{cond_type}{sset}{ckpt_n}{init_from}{sched_name}_text{text_guid}_steps{infer_steps}'


@hydra.main(version_base='1.2', config_path="configs", config_name="evaluate")
def evaluate(newcfg: DictConfig) -> None:
    # log job hash
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
    output_path = exp_folder / 'metrics'/ fd_name
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f'Output path: {output_path}')
    # init wandb if it is not None
    if 'name' in cfg.logger:
        cfg.logger.name = f'eval-{exp_folder.name}-{fd_name}'
        cfg.logger.__delattr__('logger_name')
        cfg.logger.__delattr__('save_dir')
        cfg.logger.__delattr__('offline')
        cfg.logger.__delattr__('log_model')

    wandb_logger = wandb.init(**cfg.logger,
                        config=OmegaConf.to_container(cfg, resolve=True))
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
    logger.info('------Logger & Config Loaded---------')

    evaluator = instantiate(cfg.evaluator)

    logger.info('------Evaluator Loaded---------')

    model_path = cfg.last_ckpt_path
    # load the model
    model = MD.load_from_checkpoint(model_path,
                                    renderer=aitrenderer,
                                    infer_scheduler=cfg.model.infer_scheduler,
                                    diff_params=cfg.model.diff_params,
                                    strict=False)
    model.freeze()
    model.cuda()
    logger.info('------Model Loaded---------')


    #########################DATA LOADING FOR TEST##############################
    data_module = instantiate(cfg.data)

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
        batch_size_test = 32

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             shuffle=False,
                                             num_workers=0,
                                             batch_size=batch_size_test,
                                             collate_fn=collate_fn)
    ds_iterator = testloader 
    logger.info('------Data Loaded---------')

    ###########################################################################

    # delete the model file if we loaded it from an artifact
    # create output directory


    init_diff_from = cfg.init_from
    if cfg.model.motion_condition is None:
        # hml3D trained only!
        mode_cond = 'text_cond'
    else:
        mode_cond = cfg.condition_mode

    #######################
    ### DATALOADER ########
    #######################
    iter = 0
    with torch.no_grad():
        for batch in tqdm(ds_iterator):
            text_diff = batch['text']
            target_lens = batch['length_target']
            keyids = batch['id']
            no_of_motions = len(keyids)
            batch = prepare_test_batch(model, batch)
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
                                            pack_to_dict=True)
            # tgt_mot = tgt_mot.cuda()

            # if model.motion_condition is not None:
            #     src_mot_cond = src_mot_cond.cuda()

            # motion_unnorm_metrs = model.unnorm_delta(motion_out_metrs)
            # do something with the full motion
            gen_metrics = pack_to_render(rots=gen_mo[..., 3:].detach().cpu(),
                                         trans=gen_mo[..., :3].detach().cpu())
            # canonicalize the motions everywhere
            from src.tools.transforms3d import rotate_motion_canonical
            for mot in [gen_metrics, src_mot_cond, tgt_mot]:
                rots_raw = torch.cat([mot['body_orient'], mot['body_pose']],
                                      dim=-1)
                for ix in range(rots_raw.shape[0]):
                    rots, trans = rotate_motion_canonical(rots_raw[ix],
                                                          mot['body_transl'][ix]
                                                         )
                    
                    mot['body_orient'][ix] = rots[..., :3]
                    mot['body_pose'][ix] = rots[..., 3:]
                    mot['body_transl'][ix] = trans

            gen_metrics = dict_to_device(gen_metrics, 'cuda')
            src_mot_cond = dict_to_device(src_mot_cond, 'cuda')
            tgt_mot = dict_to_device(tgt_mot, 'cuda')
            evaluator.evaluate_motion_batch(src_mot_cond, tgt_mot, gen_metrics)
            iter += 1
            # if iter == 2:
            #     break            

        results = evaluator.get_metrics()
        results_dict = results['metrics_avg'] | results['metrics']
        # turn results_dict into panda dataframe
        results_df = pd.DataFrame.from_dict(results['metrics_avg'],
                                            orient='index', columns=['values'])

        # save as csv
        results_df.to_csv(output_path / 'results.csv')
        # if there is a wandb logger, then log it as a table
        wandb.log({"metrics_table": wandb.Table(dataframe=results_df)})
        wandb.log({"metrics": results['metrics_avg']})

        # wandb.log({"results_bar" : wandb.plot.bar(wandb.Table(dataframe=results_df), 
        #                                           "label", "value",
        #                                            title="Custom Bar Chart")})

        # print only the average results  as pandas dataframe
        print(results['metrics_avg'])



if __name__ == "__main__":
    import os
    os.environ['HYDRA_FULL_ERROR'] = '1'
    evaluate()