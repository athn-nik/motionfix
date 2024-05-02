import logging
from multiprocessing.spawn import prepare
from re import I
import hydra
from omegaconf import DictConfig, OmegaConf
import src.launch.prepare  # noqa
from src.launch.prepare import get_last_checkpoint
from hydra.utils import to_absolute_path
from pathlib import Path
from typing import Optional
from pytorch_lightning.callbacks import LearningRateMonitor
import os
IS_LOCAL_DEBUG = src.launch.prepare.get_local_debug()

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", version_base="1.2", config_name="train")
def _train(cfg: DictConfig):
    ckpt_ft = None
    if cfg.resume is not None:
        # Go back to the code folder
        # in case the resume path is relative
        os.chdir(cfg.path.code_dir)
        # remove right trailing slash
        resume_dir = cfg.resume.rstrip('/')

        # move to the experimentation folder
        os.chdir(resume_dir)

        resume_ckpt_name = cfg.resume_ckpt_name
        # experiment, run_id = resume_dir.split('/')[-3:-1]

        if resume_ckpt_name is None:
            ckpt_ft = get_last_checkpoint(resume_dir)
        else:
            # start from a particular ckpt
            ckpt_ft = get_last_checkpoint(resume_dir,
                                          ckpt_name=resume_ckpt_name)

        cfg = OmegaConf.load('.hydra/config.yaml')

        # import ipdb; ipdb.set_trace()

        cfg.path.working_dir = resume_dir
        # cfg.experiment = experiment
        # cfg.run_id = run_id
        # this only works if you put the experiments in the same place
        # and then you change experiment and run_id also
        # not bad not good solution

    cfg.trainer.enable_progress_bar = True
    return train(cfg, ckpt_ft)

def train(cfg: DictConfig, ckpt_ft: Optional[str] = None) -> None:
    import os
    import torch 
    import socket
    os.environ['HYDRA_FULL_ERROR'] = '1'
    #if socket.gethostname() == 'ps018':
    if cfg.renderer is not None:
        os.system("Xvfb :12 -screen 1 640x480x24 &")

        os.environ['DISPLAY'] = ":12"
        os.environ['WANDB_SILENT'] = "true"
    # multiprocessing.set_start_method('spawn')
        logger.info("Training script. The outputs will be stored in:")
    working_dir = cfg.path.working_dir
    logger.info(f"The working directory is:{to_absolute_path(working_dir)}")
    logger.info("Loading libraries")
    import torch
    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from src.logger import instantiate_logger
    # from pytorch_lightning.accelerators import find_usable_cuda_devices
    logger.info("Libraries loaded")

    logger.info(f"Set the seed to {cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.renderer is not None:
        from aitviewer.configuration import CONFIG as AITVIEWER_CONFIG
        from aitviewer.headless import HeadlessRenderer
        body_models_path = f'{cfg.path.data}/body_models' if not cfg.data.debug else f'{cfg.path.minidata}/body_models'

        AITVIEWER_CONFIG.update_conf({"playback_fps": 30,
                                      "auto_set_floor": True,
                                      "smplx_models": body_models_path,
                                      "z_up": True})
        renderer = HeadlessRenderer()
    else: 
        renderer=None
    ######## DATA LOADING #########
    if IS_LOCAL_DEBUG:
        base_p_lcl = '/home/nathanasiou/Desktop/local-dedug/data/amass_bodilex_' 
        cfg.data.datapath = f'{base_p_lcl}v11.pth.tar'

    logger.info(f'Loading data module: {cfg.data.dataname}')
    data_module = instantiate(cfg.data)
    # here you can access data_module.nfeats
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    list_of_all_feats = data_module.nfeats
    cfg.model.input_feats = cfg.data.load_feats
    idx_for_inputs = [cfg.data.load_feats.index(infeat) 
                      for infeat in cfg.model.input_feats]
    total_feats_dim = [list_of_all_feats[i] 
                      for i in idx_for_inputs]
    nfeats = sum(total_feats_dim) 

    cfg.model.nfeats = nfeats
    cfg.model.dim_per_feat = total_feats_dim
    ######## /DATA LOADING #########

    if cfg.ftune is not None:
        logger.info(f"Loading model from {cfg.ftune_ckpt_path}")
        # model = instantiate(cfg.model,
        #                     renderer=renderer,
        #                     _recursive_=False)
        from src.model.base_diffusion import MD
        model = MD.load_from_checkpoint(cfg.ftune_ckpt_path,
                                        renderer=renderer,
                                        diff_params=cfg.model.diff_params,
                                        motion_condition=cfg.model.motion_condition,
                                        statistics_path=cfg.model.statistics_path,
                                        strict=False)

    else:
        # diffusion related
        model = instantiate(cfg.model,
                            renderer=renderer,
                            _recursive_=False)


    logger.info(f"Model '{cfg.model.modelname}' loaded")

    logger.info("Loading logger")
    train_logger = instantiate_logger(cfg)


    # train_logger.begin(cfg.path.code_dir, cfg.logger.project, cfg.run_id)
    logger.info("Loading callbacks")

    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose"
    }

    callbacks = [
        instantiate(cfg.callback.progress, metric_monitor=metric_monitor),
        instantiate(cfg.callback.latest_ckpt),
        instantiate(cfg.callback.last_ckpt),
        LearningRateMonitor(logging_interval='epoch')
        # instantiate(cfg.callback.render)
    ]

    logger.info("Callbacks initialized")

    logger.info("Loading trainer")
    if int(cfg.devices) > 1:
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
        # cfg.trainer.strategy = "ddp"
        
        logger.info("Force ddp strategy for more than one gpu.")
    else:
        cfg.trainer.strategy = "auto"
    logger.info(f"Training on: {cfg.devices} GPUS using {cfg.trainer.strategy} strategy.")
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True),
                         devices=cfg.devices, logger=train_logger,
                         callbacks=callbacks)
    logger.info("Trainer initialized")

    logger.info("Fitting the model..")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_ft)
    logger.info("Fitting done")

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")

    # train_logger.end(checkpoint_folder)
    logger.info(f"Training done. Reminder, the outputs are stored in:\n{working_dir}")

if __name__ == '__main__':
    _train()
