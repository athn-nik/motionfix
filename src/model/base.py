from pyexpat import features
import numpy as np
import logging
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
from src.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pathlib import Path
from src.render.mesh_viz import render_motion
import wandb
from einops import rearrange, reduce
from torch import Tensor
from typing import List, Union
from src.utils.genutils import freeze
import smplx
from os.path import exists, join
from src.utils.genutils import cast_dict_to_tensors
from src.utils.art_utils import color_map
# A logger for this file
log = logging.getLogger(__name__)

# Monkey patch SMPLH faster
from src.model.utils.smpl_fast import smpl_forward_fast
from src.utils.file_io import hack_path

class BaseModel(LightningModule):
    def __init__(self, statistics_path: str, nfeats: int, norm_type: str,
                 input_feats: List[str], dim_per_feat: List[int],
                 smpl_path: str, num_vids_to_render: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, 
                                  ignore=['eval_model','renderer']) # ignore TEMOS score

        # Save visuals, one validation step per validation epoch
        self.render_data_buffer = {"train": [], "val":[]}
        self.loss_dict = {'train': None,
                          'val': None,}
        self.stats = self.load_norm_statistics(statistics_path, self.device)
        self.nfeats = nfeats
        self.dim_per_feat = dim_per_feat
        self.norm_type = norm_type
        self.first_pose_feats_dims = [3, 6, 21*6]
        self.first_pose_feats = ['body_transl', 'body_orient', 'body_pose']
        self.input_feats_dims = list(dim_per_feat)
        self.input_feats = list(input_feats)
        self.num_vids_to_render = num_vids_to_render
        smpl_path = hack_path(smpl_path, keyword='data')
        self.body_model = smplx.SMPLHLayer(f'{smpl_path}/smplh', model_type='smplh',
                                           gender='neutral',
                                           ext='npz').to(self.device).eval();
        setattr(smplx.SMPLHLayer, 'smpl_forward_fast', smpl_forward_fast)
        freeze(self.body_model)
        log.info(f'Training using these features: {self.input_feats}')

        # Need to define:
        # forward
        # allsplit_step()
        # metrics()
        # losses()

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())
        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    
    def load_norm_statistics(self, path, device):
        # workaround for cluster local/sync
        path = hack_path(path)
        assert exists(path)
        stats = np.load(path, allow_pickle=True)[()]
        return cast_dict_to_tensors(stats, device=device)

    def run_smpl_fwd(self, body_transl, body_orient, body_pose):
        if len(body_transl.shape) > 2:
            body_transl = body_transl.flatten(0, 1)
            body_orient = body_orient.flatten(0, 1)
            body_pose = body_pose.flatten(0, 1)
  
        batch_size = body_transl.shape[0]
        from src.tools.transforms3d import transform_body_pose
        self.body_model.batch_size = batch_size
        return self.body_model.smpl_forward_fast(transl=body_transl,
                               body_pose=transform_body_pose(body_pose,
                                                             'aa->rot'),
                               global_orient=transform_body_pose(body_orient,
                                                                 'aa->rot'))


    def training_step(self, batch, batch_idx):
        train_loss, step_loss_dict = self.allsplit_step("train", batch,
                                                        batch_idx)
        if self.loss_dict['train'] is None:
            for k, v in step_loss_dict.items():
                step_loss_dict[k] = [v]
            self.loss_dict['train'] = step_loss_dict
        else:
            for k, v in step_loss_dict.items():
                self.loss_dict['train'][k].append(v)
        # for name, param in model.named_parameters():
        #    if param.grad is None:
        #         print(name)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        
        val_loss, step_val_loss_dict = self.allsplit_step("val",
                                                          batch, batch_idx)
        if self.loss_dict['val'] is None:
            for k, v in step_val_loss_dict.items():
                step_val_loss_dict[k] = [v]
            self.loss_dict['val'] = step_val_loss_dict
        else:
            for k, v in step_val_loss_dict.items():
                self.loss_dict['val'][k].append(v)
        # for name, param in model.named_parameters():
        #    if param.grad is None:
        #         print(name)
        return val_loss        

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            
            if '_multi' in loss:
                if 'bodypart' in loss:
                    loss_type, name, multi, _ = loss.split("_")                
                    name = f'{name}_multiple_bp'
                else:
                    loss_type, name, multi = loss.split("_")                
                    name = f'{name}_multiple'
            else:
                loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name


    def norm_and_cat(self, batch, features_types):
        """
        turn batch data into the format the forward() function expects
        """
        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...') 
        input_batch = {}
        ## PREPARE INPUT ##
        motion_condition = any('source' in value for value in batch.keys())
        if motion_condition:
            mo_types = ['source', 'target']
        else:
            mo_types = ['target']
            self.motion_condition = None
        for mot in mo_types:
            
            list_of_feat_tensors = [seq_first(batch[f'{feat_type}_{mot}']) 
                                    for feat_type in features_types]
            # normalise and cat to a unified feature vector
            list_of_feat_tensors_normed = self.norm_inputs(list_of_feat_tensors,
                                                           features_types)
            # list_of_feat_tensors_normed = [x[1:] if 'delta' in nx else x for nx,
                                                # x in zip(features_types, 
                                                # list_of_feat_tensors_normed)]
            
            x_norm, _ = self.cat_inputs(list_of_feat_tensors_normed)
            input_batch[mot] = x_norm
        return input_batch
    
    def norm_and_cat_single_motion(self, batch, features_types):
        """
        turn batch data into the format the forward() function expects
        """
        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...') 
        input_batch = {}
        ## PREPARE INPUT ##
            
        list_of_feat_tensors = [seq_first(batch[feat_type]) 
                                for feat_type in features_types]
        # normalise and cat to a unified feature vector
        list_of_feat_tensors_normed = self.norm_inputs(list_of_feat_tensors,
                                                        features_types)
        # list_of_feat_tensors_normed = [x[1:] if 'delta' in nx else x for nx,
                                            # x in zip(features_types, 
                                            # list_of_feat_tensors_normed)]
        
        x_norm, _ = self.cat_inputs(list_of_feat_tensors_normed)
        input_batch['motion'] = x_norm
        return input_batch
    
    def append_first_frame(self, batch, which_motion):

        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...') 
        list_of_feat_tensors = [seq_first(batch[f'{feat_type}_{which_motion}']) 
                                for feat_type in self.first_pose_feats]
        seqlen, bsz = list_of_feat_tensors[0].shape[:2]
        norm_pose_smpl = self.norm_inputs(list_of_feat_tensors,
                                          self.first_pose_feats)
        norm_pose_smpl = torch.cat(norm_pose_smpl, dim=-1)
        ## PAD THE INITIAL POSE ##
        padding_sz = np.sum(self.input_feats_dims) - np.sum(self.first_pose_feats_dims)
        norm_pose_smpl_pad = torch.zeros(1, bsz,
                                    batch[f'{which_motion}_motion'].shape[-1],
                                    device=self.device)
        norm_pose_smpl_pad[:, :, :norm_pose_smpl.shape[-1]] = norm_pose_smpl[:1]
        batch[f'{which_motion}_motion'] = torch.cat([norm_pose_smpl_pad,
                                                    batch[f'{which_motion}_motion']
                                                    ],
                                                    dim=0)

        return batch

    def norm(self, x, stats):
        if self.norm_type == "standardize":
            mean = stats['mean'].to(self.device)
            std = stats['std'].to(self.device)
            return (x - mean) / (std + 1e-5)
        elif self.norm_type == "min_max":
            max = stats['max'].to(self.device)
            min = stats['min'].to(self.device)
            assert ((x - min) / (max - min + 1e-5)).min() >= 0
            assert ((x - min) / (max - min + 1e-5)).max() <= 1
            return (x - min) / (max - min + 1e-5)

    def unnorm(self, x, stats):
        if self.norm_type == "standardize":
            mean = stats['mean'].to(self.device)
            std = stats['std'].to(self.device)
            return x * (std + 1e-5) + mean
        elif self.norm_type == "min_max":
            max = stats['max'].to(self.device)
            min = stats['min'].to(self.device)
            return x * (max - min + 1e-5) + min

    def unnorm_state(self, state_norm: Tensor) -> Tensor:
        # unnorm state
        return self.cat_inputs(
            self.unnorm_inputs(self.uncat_inputs(state_norm,
                                                 self.first_pose_feats_dims),
                               self.first_pose_feats))[0]
        
    def unnorm_delta(self, delta_norm: Tensor) -> Tensor:
        # unnorm delta
        return self.cat_inputs(
            self.unnorm_inputs(self.uncat_inputs(delta_norm,
                                                 self.input_feats_dims),
                               self.input_feats))[0]

    def norm_state(self, state:Tensor) -> Tensor:
        # normalise state
        return self.cat_inputs(
            self.norm_inputs(self.uncat_inputs(state, 
                                               self.first_pose_feats_dims),
                             self.first_pose_feats))[0]

    def norm_delta(self, delta:Tensor) -> Tensor:
        # normalise delta
        return self.cat_inputs(
            self.norm_inputs(self.uncat_inputs(delta, self.input_feats_dims),
                             self.input_feats))[0]

    def cat_inputs(self, x_list: List[Tensor]):
        """
        cat the inputs to a unified vector and return their lengths in order
        to un-cat them later
        """
        return torch.cat(x_list, dim=-1), [x.shape[-1] for x in x_list]
    
    def uncat_inputs(self, x: Tensor, lengths: List[int]):
        """
        split the unified feature vector back to its original parts
        """
        return torch.split(x, lengths, dim=-1)
    
    def norm_inputs(self, x_list: List[Tensor], names: List[str]):
        """
        Normalise inputs using the self.stats metrics
        """
        x_norm = []
        for x, name in zip(x_list, names):
            x_norm.append(self.norm(x, self.stats[name]))
        return x_norm

    def unnorm_inputs(self, x_list: List[Tensor], names: List[str]):
        """
        Un-normalise inputs using the self.stats metrics
        """
        x_unnorm = []
        for x, name in zip(x_list, names):
            x_unnorm.append(self.unnorm(x, self.stats[name]))
        return x_unnorm

    def allsplit_epoch_end(self, split: str):
        video_names = []
        # RENDER
        if self.renderer is not None:
            if self.global_rank == 0 and self.trainer.current_epoch != 0:
                if split == 'train':
                    if self.trainer.current_epoch%self.render_vids_every_n_epochs == 0:                        
                        video_names = []
                        # self.render_buffer(self.render_data_buffer[split],
                                                        # split=split)
                else:
                    video_names = self.render_buffer(self.render_data_buffer[split],
                                                        split=split)
                    # # log videos to wandb
                    # self.render_buffer(self.render_data_buffer[split],split=split)

        
                if self.logger is not None and video_names:
                    log_render_dic = {}
                    for v in video_names:
                        logname = f'{split}_renders/' + v.replace('.mp4',
                                                        '').split('/')[-1][4:-4]
                        logname = f'{logname}_kid'
                        log_render_dic[logname] = wandb.Video(v, fps=30,
                                                                format='mp4') 
                    self.logger.experiment.log(log_render_dic)
        self.render_data_buffer[split].clear()

        # if split == "val":
        #     metrics_dict = self.metrics.compute()
        #     metrics_dict = {f"Metrics/{metric}": value for metric, value in metrics_dict.items()}
        #     metrics_dict.update({"epoch": float(self.trainer.current_epoch)})
        #     self.log_dict(metrics_dict)

        return

    def on_train_epoch_end(self):
        return self.allsplit_epoch_end("train")
    
    def on_validation_epoch_end(self):
        return self.allsplit_epoch_end("val")

    def on_test_epoch_end(self):
        return self.allsplit_epoch_end("test")

    def configure_optimizers(self):
        optim_dict = {}
        optimizer = torch.optim.AdamW(lr=self.hparams.optim.lr,
                                      params=self.parameters())
        
        optim_dict['optimizer'] = optimizer
        # if self.hparams.NAME 
        if self.hparams.lr_scheduler == 'reduceonplateau':
            optim_dict['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-3)
            optim_dict['monitor'] = 'losses/total/train'
        elif self.hparams.lr_scheduler == 'steplr':
            optim_dict['lr_scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200)

        return optim_dict 

    @torch.no_grad()
    def render_buffer(self, buffer: list[dict], split=False):
        from src.render.video import stack_vids
        novids = self.num_vids_to_render
        # create videos and save full paths
        folder = "epoch_" + str(self.trainer.current_epoch).zfill(3)
        folder =  Path('visuals') / folder / split
        folder.mkdir(exist_ok=True, parents=True)
        stacked_videos = []

        for data in buffer:
            # RUN FWD PASS
            for k in data.keys():
                if isinstance(data[k], dict):
                    motion_type = k
                    for body_repr_name, body_repr_data in data[motion_type].items():            
                        data[motion_type][body_repr_name] = body_repr_data
                else:
                    data[k] = data[k]

            for iid_tor in range(novids):
                flname = folder / str(iid_tor).zfill(3)
                video_names = []
                cur_text = data['text_descr'][iid_tor]
                cur_key = data['keyids'][iid_tor]

                for k, v in data.items():
                    if isinstance(v, dict):
                        mot_to_rend = {k2: v2[iid_tor] for k2, v2 in v.items()}

                        # RENDER THE MOTION
                        fname = render_motion(self.renderer, mot_to_rend,
                                              f'{flname}_{k}_{cur_key}',
                                              text_for_vid=cur_text,
                                              pose_repr='aa',
                                              color=color_map[k])
                        
                        video_names.append(fname)

                stacked_fname = stack_vids(video_names, 
                                           fname=f'{flname}_{cur_key}_stk.mp4',
                                           orient='h')
                stacked_videos.append(stacked_fname)
        return stacked_videos            
    # might be needed not working in multi GPU --> all_split_end
    # Logging per joint things 
    
    # from sinc.info.joints import smplh_joints
    # smplh_joints = smplh_joints[:22]
    # columns = ['Method', 'Epoch']
    # columns.extend(smplh_joints[1:])
    # # self.wandb_table = self.logger.experiment.Table(columns=columns)
    # mnames = ['AVE_pose', 'AVE_joints', 'APE_pose', 'APE_joints']
    # table_joints = []
    # for metric in mnames:
    #     tabdata = metrics_dict[metric]
    #     if '_joints' in metric:
    #         tabdata = tabdata.detach().cpu()
    #         tabdata = tabdata[1:] # discard root for global errors
    #     else:
    #         tabdata = tabdata.detach().cpu()
    #     tabdata = torch.cat((torch.tensor([self.trainer.current_epoch]), tabdata))
    #     tabdata = list(tabdata.numpy())
    #     tabdata = [metric] + tabdata
    #     table_joints.append(tabdata)
    #     # self.logger.experiment.add_data(table_joints)
    #     self.logger.log_table(key="Joints Metrics", data=table_joints, columns=columns)
