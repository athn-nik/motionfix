import logging
from multiprocessing.spawn import prepare
from re import I
import hydra
from omegaconf import DictConfig, OmegaConf
import sinc.launch.prepare  # noqa
from sinc.launch.prepare import get_last_checkpoint
from hydra.utils import to_absolute_path
from pathlib import Path
from typing import Optional
import os

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
    os.environ['HYDRA_FULL_ERROR'] = '1'

    # import multiprocessing
    # multiprocessing.set_start_method('spawn')
    logger.info("Training script. The outputs will be stored in:")
    working_dir = cfg.path.working_dir
    logger.info(f"The working directory is:{to_absolute_path(working_dir)}")
    logger.info("Loading libraries")
    import torch
    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from sinc.logger import instantiate_logger
    # from pytorch_lightning.accelerators import find_usable_cuda_devices
    logger.info("Libraries loaded")

    logger.info(f"Set the seed to {cfg.seed}")
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f'Loading data module: {cfg.data.dataname}')
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    # in case you want to use torch.compile()
    # torch._dynamo.config.debug=True
    
    
    
    def load_temos(cfg):
        from pathlib import Path

        from omegaconf import OmegaConf

        from hydra.utils import instantiate
        temos_path = '/is/cluster/fast/nathanasiou/data/motion-language/sinc-checkpoints/temos_score/bs32'

        temos_path = Path(temos_path)
        temoscfg = OmegaConf.load(temos_path / ".hydra/config.yaml")

        # Overload it
        # Instantiate all modules specified in the configs
        temos_model = instantiate(temoscfg.model,
                                  nfeats=135,
                                  logger_name="none",
                                  nvids_to_save=None,
                                  _recursive_=False)


        last_ckpt_path = temos_path / "checkpoints/last.ckpt"
        # Load the last checkpoint
        temos_model.load_state_dict(torch.load(last_ckpt_path)["state_dict"])
        # temos_model = temos_model.load_from_checkpoint(last_ckpt_path)
        temos_model.eval()
        return temos_model, temoscfg
    
    # eval_model, _ = load_temos(cfg)

    # from copy import deepcopy
    # temos_motion_enc = deepcopy(eval_model.motionencoder)
    # #####
    # logger.info(f'Loading model {cfg.model.modelname}')
    temos_motion_enc = None 
    if cfg.model.modelname == 'sinc_mld':
        from mld_specifics import parse_args
        cfg_for_mld = parse_args()  # parse config file

        from sinc.model.mld import MLD
        model = MLD(cfg_for_mld, cfg.transforms, cfg.path)
        state_dict = torch.load('/is/cluster/fast/nathanasiou/logs/sinc/sinc-arxiv/temos-bs64x1-scheduler/babel-amass/checkpoints/latest-epoch=599.ckpt', map_location='cpu')
        # extract encoder/decoder
        from collections import OrderedDict
        decoder_dict = OrderedDict()
        encoder_dict = OrderedDict()
        for k, v in state_dict['state_dict'].items():
            if k.split(".")[0] == "motionencoder":
                name = k.replace("motionencoder.", "")
                encoder_dict[name] = v
            if k.split(".")[0] == "motiondecoder":
                name = k.replace("motiondecoder.", "")
                decoder_dict[name] = v
    
        model.vae_encoder.load_state_dict(encoder_dict, strict=True)
        model.vae_decoder.load_state_dict(decoder_dict, strict=True)


    else:
        model = instantiate(cfg.model, eval_model=temos_motion_enc,
                            nfeats=data_module.nfeats,
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
        # instantiate(cfg.callback.render)
    ]

    logger.info("Callbacks initialized")

    logger.info("Loading trainer")
    if cfg.devices > 1:
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
        # cfg.trainer.strategy = "ddp"
        logger.info("Force ddp strategy for more than one gpu.")
    else:
        cfg.trainer.strategy = "auto"
    logger.info(f"Training on: {cfg.devices} GPUS using {cfg.trainer.strategy} strategy.")
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        devices=cfg.devices,
        logger=train_logger,
        callbacks=callbacks,
    )
    logger.info("Trainer initialized")

    # compiled_model = torch.compile(model)
    # # logger.info("Model Compiled")

    logger.info("Fitting the model..")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_ft)
    logger.info("Fitting done")

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")

    # train_logger.end(checkpoint_folder)
    logger.info(f"Training done. Reminder, the outputs are stored in:\n{working_dir}")


if __name__ == '__main__':
    _train()
