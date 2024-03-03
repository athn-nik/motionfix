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
import joblib
from src.model.utils.tools import remove_padding, pack_to_render

# A logger for this file
log = logging.getLogger(__name__)

# Monkey patch SMPLH faster
from src.model.utils.smpl_fast import smpl_forward_fast
from src.utils.file_io import hack_path

class BaseModel(LightningModule):
    def __init__(self, statistics_path: str, nfeats: int, norm_type: str,
                 input_feats: List[str], dim_per_feat: List[int],
                 smpl_path: str, num_vids_to_render: str,
                 loss_on_positions: bool,
                 renderer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, 
                                  ignore=['eval_model','renderer']) # ignore TEMOS score

        # Save visuals, one validation step per validation epoch
        self.render_data_buffer = {"train": [], "val":[]}
        self.set_buf = {
                        'text_cond': [],
                        'mot_cond': [],
                        'full_cond': []
                       }

        self.loss_dict = {'train': None,
                          'val': None,}
        self.stats = self.load_norm_statistics(statistics_path, self.device)
        # from src.model.utils.tools import pack_to_render
        # mr = pack_to_render(aa.detach().cpu(), trans=None)
        # mr = {k: v[0] for k, v in mr.items()}
        # fname = render_motion(aitrenderer, mr,
        #                  "/home/nathanasiou/Desktop/conditional_action_gen/modilex/pose_test",
        #                 pose_repr='aa',
        #                 text_for_vid=str(keyids[0]),
        #                 color=color_map['generated'],
        #                 smpl_layer=smpl_layer)

        self.nfeats = nfeats
        self.dim_per_feat = dim_per_feat
        self.norm_type = norm_type
        self.first_pose_feats_dims = [3, 6, 21*6]
        self.first_pose_feats = ['body_transl', 'body_orient', 'body_pose']
        self.input_feats_dims = list(dim_per_feat)
        self.input_feats = list(input_feats)
        self.num_vids_to_render = num_vids_to_render
        smpl_path = hack_path(smpl_path, keyword='data')

        if loss_on_positions:
            self.body_model = smplx.SMPLHLayer(f'{smpl_path}/smplh',
                                               model_type='smplh',
                                               gender='neutral',
                                               ext='npz').to(self.device).eval();
            setattr(smplx.SMPLHLayer, 'smpl_forward_fast', smpl_forward_fast)
            freeze(self.body_model)
        from aitviewer.models.smpl import SMPLLayer
        if renderer is not None:
            self.smpl_ait = SMPLLayer(model_type='smplh',
                                    ext='npz',
                                    gender='neutral')
        log.info(f'Using these features: {self.input_feats}')
        data_path = Path(hack_path(smpl_path, keyword='data')).parent
        # self.test_subset = joblib.load(data_path / 'test_kinedit.pth.tar')
        self.paths_of_rendered_subset = []
        self.paths_of_rendered_subset_tgt = []
        self.paths_of_rendered_subset_src = []
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

    def run_smpl_fwd(self, body_transl, body_orient, body_pose, fast=True):
        if len(body_transl.shape) > 2:
            body_transl = body_transl.flatten(0, 1)
            body_orient = body_orient.flatten(0, 1)
            body_pose = body_pose.flatten(0, 1)
  
        batch_size = body_transl.shape[0]
        from src.tools.transforms3d import transform_body_pose
        self.body_model.batch_size = batch_size
        if fast:
            return self.body_model.smpl_forward_fast(transl=body_transl,
                                body_pose=transform_body_pose(body_pose,
                                                                'aa->rot'),
                                global_orient=transform_body_pose(body_orient,
                                                                    'aa->rot'))
        else:
            return self.body_model(transl=body_transl,
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
                                    for feat_type in features_types if f'{feat_type}_{mot}' in batch.keys()]
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

    @torch.no_grad()
    def render_gens_set(self, buffer: list[dict]):
        from src.render.video import stack_vids
        from tqdm import tqdm
        novids = self.num_vids_to_render
        # create videos and save full paths
        epo = str(self.trainer.current_epoch)
        folder = "epoch_" + epo.zfill(3)
        folder =  Path('visuals') / folder 
        folder.mkdir(exist_ok=True, parents=True)

        video_names_all = {}

        for data_variant, variant_vals in buffer.items():
            if variant_vals:
                novids = len(variant_vals[0]['keyids'])
                video_names_cur = []
                
                for iid_tor in tqdm(range(novids), 
                                    desc=f'Generating {data_variant} videos'):
                    cur_text = variant_vals[0]['text_descr'][iid_tor]
                    cur_key = variant_vals[0]['keyids'][iid_tor]

                    gen_motion = variant_vals[0]['generation']
                    mot_to_rend = {bd_f: bd_v[iid_tor].detach().cpu()
                                    for bd_f, bd_v in gen_motion.items()}

                    # RENDER THE MOTION
                    fname = render_motion(self.renderer, mot_to_rend,
                                        folder/f'{cur_key}_{data_variant}_{epo}',
                                        pose_repr='aa', 
                                        color=color_map['generation'],
                                        smpl_layer=self.smpl_ait)
                            
                    video_names_cur.append(fname)
            if variant_vals:
                video_names_all[data_variant] = video_names_cur
            else:
                video_names_all[data_variant] = []

        return video_names_all

    def allsplit_epoch_end(self, split: str):
        import os
        from src.render.video import stack_vids, put_text
        video_names = []
        # RENDER
        curep = str(self.trainer.current_epoch)
        # do_render = curep%self.render_vids_every_n_epochs
        if self.renderer is not None:
            if self.global_rank == 0 and self.trainer.current_epoch != 0:
                if split == 'val': # and do_render == 0:
                    folder = "epoch_" + curep.zfill(3)
                    folder =  Path('visuals') / folder 
                    folder.mkdir(exist_ok=True, parents=True)
                    if self.motion_condition == 'source':
                        vids_gt_src, vids_gt_tgt = self.render_subset_gt()
                        all_zipped = list(zip(vids_gt_src, vids_gt_tgt))
                        texts_descrs = self.test_subset['text']

                    video_names_generations = self.render_gens_set(self.set_buf)

                    # Zip the sorted lists
                    log_render_dic = {}

                    for gen_var, vds_paths in video_names_generations.items():
                        stacked_videos = []
                        if self.motion_condition == 'source':
                            for idx, (src, tgt) in enumerate(all_zipped):
                                kid = src.split('/')[-1].split('_')[0]
                                fname = folder / f'{kid}_{gen_var}_{curep}_stk'
                                stacked_fname = stack_vids([src, tgt, vds_paths[idx]], 
                                                        fname=f'{fname}.mp4',
                                                        orient='h')
                                stack_w_text = put_text(self.test_subset['text'][idx],
                                                        stacked_fname,
                                                        f'{fname}_txt.mp4')
                                stacked_videos.append(stack_w_text)
                            for v in stacked_videos:
                                logname = os.path.basename(v).split('_')[:2]
                                logname = '_'.join(logname)
                                logname = f'{gen_var}/' + logname
                                log_render_dic[logname] = wandb.Video(v, fps=30,
                                                                    format='mp4') 
                            if self.logger is not None:
                                self.logger.experiment.log(log_render_dic)
                        else:
                            if self.set_buf[gen_var]:
                                for idx, vd_p in enumerate(vds_paths):
                                    kid = self.set_buf[gen_var][0]['keyids'][idx]
                                    fname = folder / f'{kid}_{gen_var}_{curep}'
                                    stack_w_text = put_text(self.set_buf[gen_var][0]['text_descr'][idx],
                                                            vd_p,
                                                            f'{fname}_txt.mp4')
                                    stacked_videos.append(stack_w_text)
                                for v in stacked_videos:
                                    logname = os.path.basename(v).split('_')[:2]
                                    logname = '_'.join(logname)
                                    logname = f'{gen_var}/' + logname
                                    log_render_dic[logname] = wandb.Video(v, fps=30,
                                                                        format='mp4') 
                                if self.logger is not None:
                                    self.logger.experiment.log(log_render_dic)

        if split == 'val':
            for k, v in self.set_buf.items():
                if v:
                    self.set_buf[k].clear()
    
        #######################################################################
        #     if self.global_rank == 0 and self.trainer.current_epoch != 0:
        #         if split == 'train':
        #             if self.trainer.current_epoch%self.render_vids_every_n_epochs == 0:                        
        #                 video_names = []
        #                 # self.render_buffer(self.render_data_buffer[split],
        #                                                 # split=split)
        #         else:
        #             video_names = self.render_buffer(self.render_data_buffer[split],
        #                                                 split=split)
        #             # # log videos to wandb
        #             # self.render_buffer(self.render_data_buffer[split],split=split)
        #         if self.logger is not None and video_names:
        #             log_render_dic = {}
        #             for v in video_names:
        #                 logname = f'{split}_renders/' + v.replace('.mp4',
        #                                                 '').split('/')[-1][4:-4]
        #                 logname = f'{logname}_kid'
        #                 try:
        #                     log_render_dic[logname] = wandb.Video(v, fps=30,
        #                                                           format='mp4') 
        #                 except:
        #                     break
        #             try:
        #                 self.logger.experiment.log(log_render_dic)
        #             except:
        #                 print('could not log this time!')
        # self.render_data_buffer[split].clear()

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
        
        # optim_dict['optimizer'] = optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                      start_factor=1.0, 
                                                      end_factor=0.1,
                                                      total_iters=1000,
                                                      verbose=True)
        # # if self.hparams.NAME 
        # if self.hparams.lr_scheduler == 'reduceonplateau':
        #     optim_dict['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                                             threshold=1e-3)
        #     optim_dict['monitor'] = 'losses/total/train'
        # elif self.hparams.lr_scheduler == 'steplr':
        #     optim_dict['lr_scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                                                  step_size=200)
        return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                    },
                }
        # return optim_dict
    
    def prepare_mot_masks(self, source_lens, target_lens, max_len=300):
        from src.data.tools.tensors import lengths_to_mask
        import torch.nn.functional as F
        mask_target = lengths_to_mask(target_lens,
                                              self.device)
        mask_source = lengths_to_mask(source_lens, self.device)

        if not source_lens[0] == 1 and max_len is not None:
            padded_mask_target = F.pad(mask_target,
                                    (0, max_len - mask_target.size(1)),
                                    value=0)
            padded_mask_source = F.pad(mask_source,
                                    (0, max_len - mask_source.size(1)),
                                    value=0)
            
            return padded_mask_source, padded_mask_target
        else:
            return mask_source, mask_target

    @torch.no_grad()
    def process_batch(self, batch):
        batch_to_gpu = { k: v.to(self.device) for k,
                            v in self.test_subset.items() 
                         if torch.is_tensor(v) }

        input_batch = self.norm_and_cat(batch_to_gpu, self.input_feats)
        for k, v in input_batch.items():
            if self.input_deltas:
                batch[f'{k}_motion'] = v[1:]
            else:
                batch[f'{k}_motion'] = v
                batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]
        return batch

        
    def render_subset_gt(self):
        batched = self.process_batch(self.test_subset)
        mask_src, mask_tgt = self.prepare_mot_masks(batched['length_source'],
                                                    batched['length_target'])
        src_mot_cond = batched['source_motion']
        gt_texts = batched['text']
        gt_keyids = batched['id']
        folder = "epoch_" + str(self.trainer.current_epoch).zfill(3)
        folder =  Path('visuals') / folder 
        folder.mkdir(exist_ok=True, parents=True)
        if not self.paths_of_rendered_subset_src:
            src_mots, tgt_mots = self.batch2motion(batched,
                                                    pack_to_dict=True,
                                                    slice_til=None)
            for idx, keyid in enumerate(batched['id']):

                src_mot = {k2: v2[idx] for k2,
                            v2 in src_mots.items()}

                # RENDER THE MOTION
                fname = render_motion(self.renderer, src_mot,
                                        folder / f'{keyid}_source',
                                        # text_for_vid=gt_texts[idx],
                                        pose_repr='aa',
                                        color=color_map['source'],
                                        smpl_layer=self.smpl_ait)
                self.paths_of_rendered_subset_src.append(fname)

                tgt_mot = {k2: v2[idx] for k2,
                            v2 in tgt_mots.items()}
                # RENDER THE MOTION
                fname = render_motion(self.renderer, tgt_mot,
                                        folder / f'{keyid}_target',
                                        # text_for_vid=gt_texts[idx],
                                        pose_repr='aa',
                                        color=color_map['target'],
                                        smpl_layer=self.smpl_ait)
                self.paths_of_rendered_subset_tgt.append(fname)
        return self.paths_of_rendered_subset_src, self.paths_of_rendered_subset_tgt

    def batch2motion(self, batch, pack_to_dict=True,
                     slice_til=None, single_motion=False):
        # batch_to_cpu = { k: v.detach().cpu() for k, v in batch.items() 
        #                 if torch.is_tensor(v) }
        tot_dim_deltas = 0
        if self.using_deltas:
            for idx_feat, in_feat in enumerate(self.input_feats):
                if 'delta' in in_feat:
                    tot_dim_deltas += self.input_feats_dims[idx_feat]
        source_motion_gt = None
        if batch['source_motion'] is not None:
            # source motion
            source_motion = batch['source_motion']
            # source_motion = self.unnorm_delta(source_motion)[..., tot_dim_deltas:]
            source_motion = self.diffout2motion(source_motion.detach().permute(1,
                                                                            0,
                                                                            2))
            source_motion = source_motion.detach().cpu()
            if pack_to_dict:
                source_motion_gt = pack_to_render(rots=source_motion[..., 3:],
                                                  trans=source_motion[...,:3])
            else:
                source_motion_gt = source_motion
            if slice_til is not None:
                source_motion_gt = {k: v[slice_til] 
                                    for k, v in source_motion_gt.items()}

        # target motion
        target_motion = batch['target_motion']
        target_motion = self.diffout2motion(target_motion.detach().permute(1,
                                                                        0,
                                                                        2))
        target_motion = target_motion.detach().cpu()
        if pack_to_dict:
            target_motion_gt = pack_to_render(rots=target_motion[..., 3:],
                                            trans=target_motion[...,:3])
        else:
            target_motion_gt = target_motion
        if slice_til is not None:
            target_motion_gt = {k: v[slice_til] 
                                for k, v in target_motion_gt.items()}

        return source_motion_gt, target_motion_gt

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
                                              color=color_map[k],
                                              smpl_layer=self.smpl_ait)
                        
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
