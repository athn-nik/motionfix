import os
import logging
from sched import scheduler
import hydra
from pathlib import Path
from omegaconf import DictConfig
from omegaconf import OmegaConf
from src.render.mesh_viz import render_motion

# from src.render.mesh_viz import visualize_meshes
from src.render.video import save_video_samples
import src.launch.prepare  # noqa
from tqdm import tqdm
from src.utils.file_io import read_json
from src.launch.prepare import get_last_checkpoint
import torch
from aitviewer.headless import HeadlessRenderer
import itertools
from aitviewer.configuration import CONFIG as AITVIEWER_CONFIG
from src.model.utils.tools import pack_to_render
import diffusers
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="demo")
def _render(cfg: DictConfig) -> None:
    return render(cfg)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def render(newcfg: DictConfig) -> None:
    from pathlib import Path

    exp_folder = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")
    # new_sched = newcfg.model.infer_scheduler
    # if new_sched is not None:
    #     logger.info(f"...Sampling with a different scheduler...{new_sched.name}")
    #     logger.info(f"Scheduler: {new_sched.name}")
    #     logger.info(f"Parameters of scheduler: {new_sched.scheduler}")

    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
    
    # if newcfg.scheduler.name == 'ddim':

    #     infer_scheduler = diffusers.DDIMScheduler(
    #         num_train_timesteps=sch_cfg.num_inference_timesteps,
    #         beta_start=sch_cfg.beta_start,
    #         beta_end=sch_cfg.beta_end,
    #         beta_schedule=sch_cfg.beta_schedule,
    #         clip_sample=sch_cfg.clip_sample,
    #         prediction_type=sch_cfg.prediction_type,
    #         set_alpha_to_one= False,
    #         steps_offset= 1
    #     )

    # elif newcfg.scheduler.name == 'ddpm':
    #     infer_scheduler = diffusers.DDPMScheduler(
    #         num_train_timesteps=sch_cfg.num_inference_timesteps,
    #         beta_start=sch_cfg.beta_start,
    #         beta_end=sch_cfg.beta_end,
    #         beta_schedule=sch_cfg.beta_schedule,
    #         clip_sample=sch_cfg.clip_sample,
    #         prediction_type=sch_cfg.prediction_type,
    #     )

    # else:
    #     exit('Scheduler not supported!')

    output_path = exp_folder / 'demo-renders'
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Sample script. The outputs will be stored in:{output_path}")

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    seed_logger = logging.getLogger("pytorch_lightning.utilities.seed")
    seed_logger.setLevel(logging.WARNING)

    pl.seed_everything(cfg.seed)
    # only pair evaluation to be fair
    # keep same order

    from tqdm import tqdm

    logger.info("Loading model")
    # Instantiate all modules specified in the configs
    # FIXME 
    AITVIEWER_CONFIG.update_conf({"playback_fps": 30,
                                "auto_set_floor": True,
                                "smplx_models": 'data/body_models',
                                'z_up': True})

    aitrenderer = HeadlessRenderer()

    model = instantiate(cfg.model,
                        renderer=aitrenderer,
                        _recursive_=False)

    logger.info(f"Model '{cfg.model.modelname}' loaded")

    
    # Load the last checkpoint
    # del checkpoint['state_dict']["text_encoder.text_model.text_model.embeddings.position_ids"]
    # del checkpoint['state_dict']["text_encoder.text_model.vision_model.embeddings.position_ids"]
    model = model.load_from_checkpoint(last_ckpt_path,
                                       renderer=aitrenderer,
                                       strict=False)
                                    #    infer_scheduler=infer_scheduler)    model.eval()
    logger.info("Model weights restored")

    logger.info("Trainer initialized")
    import numpy as np

    data_module = instantiate(cfg.data)
    # state_dict = torch.load(cfg.TEST.CHECKPOINTS,
    #                         map_location="cpu")["state_dict"]
    # # remove mismatched and unused params
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     old, new = "denoiser.decoder.0.", "denoiser.decoder."
    #     # old1, new1 = "text_encoder.text_model.text_model", "text_encoder.text_model.vision_model"
    #     old1 = "text_encoder.text_model.vision_model"
    #     if k[: len(old)] == old:
    #         name = k.replace(old, new)
    #     # elif k[: len(old)] == old:
    #     #     name = k.replace(old, new)
    #     else:
    #         name = k

    #     new_state_dict[name] = v
    #     # if k.split(".")[0] not in ["text_encoder", "denoiser"]:
    #     #     new_state_dict[k] = v
    # model.load_state_dict(new_state_dict, strict=False)

    # model.load_state_dict(state_dict, strict=True)

    # logger.info("model {} loaded".format(cfg.model.model_type))
    # model.sample_mean = cfg.TEST.MEAN
    # model.fact = cfg.TEST.FACT
    # model.to(device)
    # model.eval()

    if cfg.mode == 'demo':
        lengths = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

        texts = ['slower', 'faster']
        len_text = [list(tup) for tup in list(itertools.product(lengths, texts))]
        idx = 0
    else:
        test_dataset = data_module.dataset['test']
        testloader = torch.utils.data.DataLoader(test_dataset,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 batch_size=8)

    if cfg.mode == 'denoise':
        # sample
        with torch.no_grad():
            for batch in tqdm(testloader):
                source_lens = batch['length_source']
                text_diff = batch['text']
                target_lens = batch['length_target']
                keyids = batch['id']
                # model_out = model([text], [length])[0]
                # model_out = model_out.cpu().squeeze().numpy()
                batch = { k: v.to(model.device) if torch.is_tensor(v) else v
                         for k, v in batch.items() }

                input_batch = model.norm_and_cat(batch, model.input_feats)
                for k, v in input_batch.items():
                    if model.input_deltas:
                        batch[f'{k}_motion'] = v[1:]
                    else:
                        batch[f'{k}_motion'] = v
                        batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]


                mask_source, mask_target = model.prepare_mot_masks(source_lens,
                                                                  target_lens)

                diffout = model.denoise_forward(batch, mask_source,
                                                mask_target)
                input_motion_feats = dif_out['input_motion_feats']
                timesteps = dif_out['timesteps']
                noisy_motion = dif_out['noised_motion_feats']
                diffusion_fw_out = dif_out['pred_motion_feats']

                model.visualize_diffusion(diffout, target_lens, 
                                          keyids, text_diff, 
                                          'test', return_fnames=True)
                xy=1
                if model.input_deltas:
                    motion_unnorm = model.diffout2motion(diffout, 
                                                         full_deltas=True)
                    motion_unnorm = motion_unnorm.permute(1, 0, 2)
                elif model.input_deltas_transl:
                    motion_unnorm = model.diffout2motion(diffusion_fw_out)
                else:
                    motion_unnorm = model.unnorm_delta(diffout)
                batch_motion = pack_to_render(motion_unnorm[..., 3:],
                                        motion_unnorm[..., :3])
                for jj in range(motion_unnorm.shape[0]):
                    one_motion = {k: v[jj] 
                                for k, v in batch_motion.items()
                                }
                    render_motion(aitrenderer, one_motion,
                                  output_path / f"movie_{idx}_{jj}",
                                  pose_repr='aa',
                                  text_for_vid=timesteps[jj])
                idx += 1
    
    elif cfg.mode == 'sample':
        # sample
        with torch.no_grad():
            for batch_len_text in tqdm(chunker(len_text, 8)):
                lengths, texts = zip(*batch_len_text)
            # task: input or Example
            # prepare batch data  
                # model_out = model([text], [length])[0]
                # model_out = model_out.cpu().squeeze().numpy()
                mask_source, mask_target = model.prepare_mot_masks(
                                                    batch['length_source'],
                                                    batch['length_target']
                                                    )
                if model.motion_condition is None:
                    source_mot = None
                else:
                    batch['source_motion']
                dif_out = model.generate_motion(list(texts),
                                                batch['source_motion']
                                                mask_source,
                                                mask_target)
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
        
    else:
        # sample
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

if __name__ == '__main__':

    os.environ['DISPLAY'] = ":1"
    os.system("Xvfb :11 -screen 1 640x480x24 &")

    _render()
