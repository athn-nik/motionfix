import logging
from pathlib import Path
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from src.tools.easyconvert import matrix_to
from src.utils.file_io import read_json
from pytorch_lightning.utilities import rank_zero_only


logger = logging.getLogger(__name__)
plt_logger = logging.getLogger("matplotlib.animation")
plt_logger.setLevel(logging.WARNING)


# Can be run asynchronously 
def render_and_save(args, vid_format="mp4"):
    from src.render import visualize_meshes, render_animation
    jts_or_vts, name, index, split, folder, fps, description, current_epoch = args
    fig_number = str(index).zfill(2)
    filename = f"{name}_{split}_{fig_number}.{vid_format}"
    output = folder / filename
    output = str(output.absolute())
    # Render
    if jts_or_vts.shape[1] > 100:
        output = visualize_meshes(jts_or_vts, save_path=output)
    else:
        render_animation(jts_or_vts, output=output, title=description, fps=30)

    return output, fig_number, name, description


def log_to_none(path: str, log_name: str, fps: float,
                 global_step: int, train_logger,
                 vid_format, **kwargs):
    return


def log_to_wandb(path: str, log_name: str, caption: str, fps: float,
                 vid_format, **kwargs):
    import wandb
    return log_name, wandb.Video(path, fps=int(fps), format=vid_format,
                                 caption=caption)


def log_to_tensorboard(path: str, log_name: str, caption: str, fps: float,
                       global_step: int, train_logger,
                       vid_format, **kwargs):
    if vid_format == "gif":
        # Need to first load the gif by hand
        from PIL import Image, ImageSequence
        import numpy as np
        import torch
        # Load
        gif = Image.open(path)
        seq = np.array([np.array(frame.convert("RGB"))
                        for frame in ImageSequence.Iterator(gif)])
        vid = torch.tensor(seq)[None].permute(0, 1, 4, 2, 3)
    elif vid_format == "mp4":
        a = 1
        import ipdb
        ipdb.set_trace()

    # Logger name
    train_logger.add_video(log_name,
                           vid, fps=fps,
                           global_step=global_step)


class RenderCallback(Callback):
    def __init__(self, bm_path: str = None,
                 path: str = "visuals",
                 logger_type: str = "wandb",
                 save_last: bool = True,
                 vid_format: str = "mp4",
                 every_n_epochs: int = 20,
                 num_workers: int = 0,
                 nvids_to_save: int = 5,
                 fps: float = 30.0,
                 modelname = 'sinc') -> None:

        if logger_type == "wandb":
            self.log_to_logger = log_to_wandb
        elif logger_type == "tensorboard":
            self.log_to_logger = log_to_tensorboard
        elif logger_type == "none":
            self.log_to_logger = log_to_none
        else:
            raise NotImplementedError("This logger is unknown, please use tensorboard or wandb.")

        self.logger_type = logger_type
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

        self.fps = fps
        self.nvids = nvids_to_save
        self.every_n_epochs = every_n_epochs
        self.num_workers = num_workers
        self.vid_format = vid_format
        self.save_last = save_last
        self.model = modelname


        if bm_path is not None:
            self.body_model_path = Path(bm_path) / 'smpl_models'
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule,
                           **kwargs) -> None:
        # if trainer.is_global_zero:
        return self.call_renderer("train", trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer: Trainer,
                            pl_module: LightningModule) -> None:
        # if trainer.is_global_zero:
        return self.call_renderer("val", trainer, pl_module)
    
    def call_renderer(self, split: str, trainer: Trainer,
                      pl_module: LightningModule) -> None:
        # comment out for faster debugging of the module / set nvids to 1-2
        
        if trainer.sanity_checking:
            return
        if self.nvids is None or self.nvids == 0:
            return
        if trainer.global_rank != 0:
            return

        
        # Don't log epoch 0
        if split == 'train':
            if trainer.current_epoch == 0 or trainer.current_epoch % self.every_n_epochs != 0:
                # Log last one (return = don't log, if it is not the last one)
                if trainer.current_epoch != (trainer.max_epochs - 1):
                    return
                # Don't log last one if we don't want it
                elif not self.save_last:
                    return
        
        logger.info(f"Render {split} samples and log to logger: {self.logger_type} from RANK: {trainer.global_rank}")

        # Prepare the folder
        folder = "epoch_" + str(trainer.current_epoch).zfill(3) + split
        folder = self.path / folder
        folder.mkdir(exist_ok=True)

        # Extract the stored data
        store_examples = pl_module.store_examples[split]
        ref_joints_or_verts = store_examples['ref']
        ref_motion_features = store_examples['ref_features']
        keyids_to_render = store_examples['keyids']
        # ref_motion_features = ref_motion_features.features
        # ref_motion_features = ref_motion_features[:self.nvids]
        texts = store_examples['text']
        lengths = store_examples['lengths']
        # Render + log
        # nvids = min(self.nvids, len(ref_joints_or_verts))
        pl_module.eval()
        with torch.no_grad():
            jts_T = pl_module.text_to_motion_forward(texts, lengths, return_motion='vertices')
            jts_T = [mot.detach().cpu().numpy() for mot in jts_T]

            jts_M = pl_module.motion_to_motion_forward(ref_motion_features.to('cuda'), lengths, 
                                                        return_motion='vertices')
            jts_M = [mot.detach().cpu().numpy() for mot in jts_M]
            texts = [', '.join(t) for t in texts] 
            for i, mot_len in enumerate(lengths):
                jts_T[i] = jts_T[i][:mot_len]
                jts_M[i] = jts_M[i][:mot_len]

            texts = [f'{t} | {keyids_to_render[i]}' for i, t in enumerate(texts)] 

            
            # elif split == 'val':
                
            #     jts_T = pl_module.text_to_motion_forward(texts, lengths, return_motion='vertices')
            #     jts_T = [mot.detach().cpu().numpy() for mot in jts_T]

            #     jts_M = pl_module.motion_to_motion_forward(ref_motion_features, lengths,
            #                                                return_motion='vertices')
            #     jts_M = [mot.detach().cpu().numpy() for mot in jts_M]
            #     texts = [', '.join(t) for t in texts]

            #     texts = [f'{t} | {keyids_to_render[i]}' for i, t in enumerate(texts)] 
        pl_module.train()
        train_logger = pl_module.logger.experiment
        list_of_logs = []
        log_dict = {}
        for jts_or_vts, name in zip([ref_joints_or_verts, jts_M, jts_T],
                                ['ref', 'from_motion', 'from_text']):
            for index, description in zip(range(self.nvids), texts):

                output, fig_number, name, text_d = render_and_save((
                                                            jts_or_vts[index],
                                                            name, index,
                                                            split, folder,
                                                            self.fps,
                                                            description,
                                                            trainer.current_epoch))
                log_name = f"visuals_{split}/{name}/{fig_number}"
                list_of_logs.append((output, log_name, text_d))

                # log_name = f"{split}_{fig_number}/{name}"
                
                # train_logger = pl_module.logger.experiment if self.logger_type is not None else None
                logid, vid_entry = self.log_to_logger(path=output,
                                                      log_name=log_name, 
                                                      caption=text_d,
                                                      fps=self.fps,
                                                      vid_format=self.vid_format)
                train_logger.log({logid: vid_entry, 'epoch': trainer.current_epoch})
                # log_dict[logid] = vid_entry

        # import ipdb; ipdb.set_trace()
        # log_dict['epoch'] = trainer.current_epoch
        
        # step_to_log = trainer.global_step

        
        # train_logger.log(log_dict, step=step_to_log)        


        # import operator
        # list_of_logs.sort(key=operator.itemgetter(2))
        # # for a,b,c in list_of_logs:       
        # #     print(a, b)

        # for vid_path, panel_name, text_desc in list_of_logs: 
        #     import ipdb; ipdb.set_trace()
        #     log_name_start, branch_or_gt, _ = panel_name.split('/')
        #     vid_id = vid_path.split('/')[-1].split('_')[-1].split('.')[0]
            
        #     log_name = f'{log_name_start}_{split}/sample-{vid_id}/{branch_or_gt}'
        #     print(vid_path)
        #     print(log_name)
            
        #     self.log_to_logger(path=vid_path, log_name=log_name, caption=text_desc,
        #                         fps=self.fps, global_step=trainer.current_epoch,
        #                         train_logger=train_logger, vid_format=self.vid_format)
        

        # multiprocess does not work with pyrender probably it stucks forever
 
