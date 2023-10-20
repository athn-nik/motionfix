import os
import logging
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

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="demo")
def _render(cfg: DictConfig) -> None:
    return render(cfg)

def render(newcfg: DictConfig) -> None:
    from pathlib import Path

    exp_folder = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path
    # Load previous config
    prevcfg = OmegaConf.load(exp_folder / ".hydra/config.yaml")

    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)


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

    logger.info("Trainer initialized")
    import numpy as np


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

    AITVIEWER_CONFIG.update_conf({"playback_fps": 30,
                                  "auto_set_floor": False,
                                  "smplx_models": 'data/body_models'})
    aitrenderer = HeadlessRenderer()

    lengths = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    texts = ['slower', 'faster']
    len_text = [list(tup) for tup in list(itertools.product(lengths, texts))]

    # sample
    with torch.no_grad():
        for length, text in tqdm(len_text):
        # task: input or Example
        # prepare batch data  
            model_out = model([text], [length])[0]
            model_out = model_out.cpu().squeeze().numpy()
            import ipdb; ipdb.set_trace()
            dif_out = model.test_diffusion_forward([length], [text])
            if model.input_deltas:
                motion_unnorm = model.diffout2motion(dif_out)
                motion_unnorm = motion_unnorm.permute(1, 0, 2)
            else:
                motion_unnorm = model.unnorm_delta(dif_out)
            motion = pack_to_render(model_out[..., 3:], model_out[..., :3])
            render_motion(aitrenderer, motion, 
                          output_path / f"{text}_{length}.mp4")

if __name__ == '__main__':

    os.environ['DISPLAY'] = ":1"
    os.system("Xvfb :11 -screen 1 640x480x24 &")

    _render()
