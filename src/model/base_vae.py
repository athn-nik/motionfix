from os import times
from typing import List, Optional, Union
from matplotlib.pylab import cond
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, mode
from torch.distributions.distribution import Distribution
from torch.nn import ModuleDict
from src.data.tools.collate import collate_tensor_with_padding
from torch.nn import functional as F
from src.data.tools.tensors import lengths_to_mask
from src.model.base import BaseModel
from src.model.utils.tools import remove_padding
from src.model.losses.utils import LossTracker
from src.data.tools import lengths_to_mask_njoints
from src.model.losses.compute_mld import MLDLosses
import inspect
from src.model.utils.tools import remove_padding, pack_to_render
from src.render.mesh_viz import render_motion
from src.tools.transforms3d import change_for, transform_body_pose, get_z_rot
from src.tools.transforms3d import apply_rot_delta
from einops import rearrange, reduce
from torch.nn.functional import l1_loss, mse_loss
from src.utils.genutils import dict_to_device
from src.utils.art_utils import color_map
import torch
import torch.distributions as dist
import logging
import wandb
from .vae_motion import CorrectionModule
log = logging.getLogger(__name__)

class MVAE(BaseModel):
    def __init__(self, 
                 text_encoder: DictConfig,
                 motion_condition_encoder: DictConfig,
                 losses: DictConfig,
                 latent_dim: int,
                 nfeats: int,
                 input_feats: List[str],
                 statistics_path: str,
                 dim_per_feat: List[int],
                 norm_type: str,
                 smpl_path: str,
                 tmr_path: str,
                 render_vids_every_n_epochs: Optional[int] = None,
                 num_vids_to_render: Optional[int] = None,
                 condition: Optional[str] = "text",
                 motion_condition: Optional[str] = "source",
                 loss_on_positions: Optional[bool] = False,
                 loss_on_verts: Optional[bool] = False,
                 scale_loss_on_positions: Optional[int] = None,
                 loss_func_pos: str = 'mse', # l1 mse
                 loss_func_feats: str = 'mse', # l1 mse
                 renderer = None,
                 source_encoder: str = 'trans_enc',
                 **kwargs):

        super().__init__(statistics_path, nfeats, norm_type, input_feats,
                         dim_per_feat, smpl_path, num_vids_to_render,
                         loss_on_positions or loss_on_verts,
                         renderer=renderer)

        if set(self.input_feats) == set(["body_transl_delta_pelv_xy",
                                         "body_orient_delta",
                                         "body_pose_delta"]):
            self.input_deltas = True
        else:
            self.input_deltas = False

        if set(["body_transl_delta_pelv_xy", "body_orient_delta",
                "body_pose_delta"]).issubset(self.input_feats):
            self.using_deltas = True
        else:
            self.using_deltas = False

        transl_feats = [x for x in self.input_feats if 'transl' in x]
        if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                  "body_transl_delta_pelv_xy"]):
            self.using_deltas_transl = True
        else:
            self.using_deltas_transl = False

        self.smpl_path = smpl_path
        self.condition = condition
        self.motion_condition = motion_condition
        if self.motion_condition == 'source':
            if source_encoder == 'trans_enc':
                self.motion_cond_encoder = instantiate(motion_condition_encoder)
            else:
                self.motion_cond_encoder = None

        from src.model.tmr_utils.utils import load_model_from_cfg, read_config
        from src.utils.file_io import hack_path
        cfg = read_config(hack_path(tmr_path, keyword='data'))
        ### Fix keys --> move to a function
        import omegaconf
        for k, v in cfg.model.items():
            if isinstance(v, omegaconf.DictConfig):
                for k2, v2 in v.items():
                    if isinstance(v2, str) and 'src.model' in v2:
                        v[k2] = v2.replace('src.model',
                                                         'src.model.tmr_utils')
            else:
                if isinstance(v, str) and 'src.model' in v:
                    cfg.model[k] = v.replace('src.model', 'src.model.tmr_utils')
        self.tmr_model = load_model_from_cfg(cfg, 'last', 
                                             eval_mode=True, 
                                             device='cuda')
        ###\\ Fix keys --> move to a function
        
        self.text_encoder = instantiate(text_encoder)
        self.loss_on_positions = loss_on_positions
        # from torch import nn
        # self.condition_encoder = nn.Linear()
        self.ep_start_scale = scale_loss_on_positions
        # self.motion_decoder = instantiate(motion_decoder, nfeats=nfeats)

        # for k, v in self.render_data_buffer.items():
        #     self.store_examples[k] = {'ref': [], 'ref_features': [], 'keyids': []}
        # self.metrics = ComputeMetrics(smpl_path)
        self.input_feats = input_feats
        self.render_vids_every_n_epochs = render_vids_every_n_epochs
        self.num_vids_to_render = num_vids_to_render
        self.renderer = renderer

        # If we want to overide it at testing time
        self.latent_dim = latent_dim
        # distribution of timesteps
        self.loss_params = losses
        # loss params terrible
        if loss_func_pos == 'l1':
            self.loss_func_pos = l1_loss
        elif loss_func_pos in ['mse', 'l2']:
            self.loss_func_pos = mse_loss

        if loss_func_feats == 'l1':
            self.loss_func_feats = l1_loss
        elif loss_func_feats in ['mse', 'l2']:
            self.loss_func_feats = mse_loss
        from .vae_motion import CorrectionModule, NormalDistDecoder
        self.correction_module = CorrectionModule(latentD=self.latent_dim, 
                                                  text_latentD=self.text_encoder.text_encoded_dim, 
                                                  mode='tirg')
        self.text_src_distr_enc = NormalDistDecoder(self.latent_dim, self.latent_dim)
        self.recons_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.__post_init__()

    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z

    def fix_input_for_tmr(self, batch):
        mots = []
        for k in ['source', 'target']:
            motion_fixed = torch.zeros_like(batch[f'{k}_motion'])
            motion_fixed[..., -6:] = batch[f'{k}_motion'][..., 3:9] 
            motion_fixed[..., 3:-6] = batch[f'{k}_motion'][..., 9:] 
            motion_fixed[..., :3] = batch[f'{k}_motion'][..., :3] 
            mots.append(motion_fixed)
        return mots[0], mots[1]

    def train_vae_forward(self, batch, mask_source_motion,
                                mask_target_motion):
        src, tgt = self.fix_input_for_tmr(batch)
        tgt_dict = {'length': batch['length_target'],
                    'mask': mask_target_motion,
                    'x': tgt.permute(1, 0, 2)}
        src_dict = {'length': batch['length_source'],
                    'mask': mask_source_motion,
                    'x': src.permute(1, 0, 2)}
        src_lat, src_distr = self.tmr_model.encode(src_dict,
                                                   modality='motion',
                                                   return_distribution=True)
        tgt_lat, tgt_distr = self.tmr_model.encode(tgt_dict,
                                                   modality='motion',
                                                   return_distribution=True)

        text_lat, _ = self.text_encoder(batch['text'])
        fused_src_txt = self.correction_module(src_lat, text_lat.squeeze())
        fused_distr = self.text_src_distr_enc(fused_src_txt)

        # during training
        dec_mot = self.tmr_model.decode(tgt_lat, mask=mask_target_motion)
        # sample
        ret = {'fused_src_txt_distr': fused_distr,
               'tgt_distr': torch.distributions.normal.Normal(tgt_distr[0],
                                                              F.softplus(tgt_distr[1])) ,
               'pred_motion_feats': dec_mot,
               'input_motion_feats': tgt}
        return ret
        # encode source motion

        # encode target motion
        # encode text
        # pass through TIRG module 
        # get the output 

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def compute_losses(self, out_dict, motion_mask_source, 
                       motion_mask_target):

        from torch import nn
        from src.data.tools.tensors import lengths_to_mask
        all_losses_dict = {}
        if self.input_deltas:
            pad_mask_jts_pos = motion_mask_target
            pad_mask = motion_mask_target
        else:
            pad_mask_jts_pos = motion_mask_target
            pad_mask = motion_mask_target
        self.input_feats = ['body_transl_delta_pelv', 'body_pose',  'body_orient']
        self.input_feats_dims =  [3, 21*6, 6]
        f_rg = np.cumsum([0] + self.input_feats_dims)
        all_losses_dict = {}
        tot_loss = torch.tensor(0.0, device=self.device)
        bs = out_dict['pred_motion_feats'].shape[0]
        data_loss = self.loss_func_feats(out_dict['pred_motion_feats'],
                                         out_dict['input_motion_feats'].permute(1, 0, 2),
                                         reduction='none')
        full_feature_loss = data_loss

        for i, _ in enumerate(f_rg[:-1]):
            if 'delta' in self.input_feats[i]:
                cur_feat_loss = full_feature_loss[:, 1:, f_rg[i]:f_rg[i+1]
                                                    ].mean(-1)*pad_mask[:, 1:]
                tot_feat_loss = cur_feat_loss.sum() / pad_mask[:, 1:].sum()
                all_losses_dict.update({self.input_feats[i]: tot_feat_loss})
            else:
                cur_feat_loss = full_feature_loss[..., f_rg[i]:f_rg[i+1]
                                                    ].mean(-1)*pad_mask
                tot_feat_loss = cur_feat_loss.sum() / pad_mask.sum()
                all_losses_dict.update({self.input_feats[i]: tot_feat_loss})
            tot_loss += tot_feat_loss
        all_losses_dict['total_loss'] = tot_loss

        loss_joints = torch.tensor(0.0)
        if self.loss_on_positions:
            J = 22
            joints_gt = rearrange(joints_gt, 'b s (j d) -> b s j d', j=J)
            loss_joints, _ = self.compute_joints_loss(out_dict, joints_gt, 
                                                      pad_mask_jts_pos)
            all_losses_dict['total_loss'] += loss_joints
            all_losses_dict['loss_joints'] = loss_joints
        # (KL term between modalities' distributions)
        all_losses_dict['kldft'] = torch.sum(torch.distributions.kl.kl_divergence(out_dict['tgt_distr'],
                                                                         out_dict['fused_src_txt_distr']), dim=[1])# size (batch_size)
        all_losses_dict['kldft'] = all_losses_dict['kldft'].clamp(min=0.0) 
        all_losses_dict['kldft']= 1e-3*all_losses_dict['kldft'].mean()
        # NOTE: `kldft_training` is only to be used at training time;
        # the clamping helps to prevent model collapse

        # (KL regularization term)
        if False:
            n_z = torch.distributions.normal.Normal(loc=torch.zeros((bs,
                                                                    self.latent_dim),
                                                                    device=self.device,
                                                                    requires_grad=False),
                                                    scale=torch.ones((bs,
                                                                    self.latent_dim),
                                                                    device=self.device,
                                                                    requires_grad=False))
            all_losses_dict['kldnp'] = torch.sum(torch.distributions.kl.kl_divergence(out_dict['tgt_distr'],
                                                                         n_z), dim=[1]).mean() # size (batch_size)

        return tot_loss + loss_joints + all_losses_dict['kldft'], all_losses_dict 

    def generate_pose(self, texts_cond, motions_cond,
                        mask_source, mask_target, 
                        init_vec_method='noise', init_vec=None,
                        gd_text=None, gd_motion=None, 
                        return_init_noise=False, 
                        condition_mode='full_cond', num_diff_steps=None):



        bsz, seqlen_tgt = mask_target.shape
        feat_sz = sum(self.input_feats_dims)
        if texts_cond is not None:
            text_emb, text_mask = self.text_encoder(texts_cond)

        cond_emb_motion = None
        cond_motion_mask = None
        
        if self.motion_condition == 'source':
            bsz, seqlen_src = mask_source.shape
            if condition_mode == 'full_cond' or condition_mode == 'mot_cond' :
                source_motion_condition = motions_cond
                cond_emb_motion = source_motion_condition
                cond_motion_mask = mask_source
            else:
                cond_emb_motion = torch.zeros(seqlen_src, bsz, feat_sz,
                                                device=self.device)
                cond_motion_mask = torch.ones((bsz, 1),
                                            dtype=bool, device=self.device)

        if init_vec_method == 'noise_prev':
            init_diff_rev = init_vec
        elif init_vec_method == 'source':
            init_diff_rev = motions_cond
            tgt_len = 1
            src_len = 1
            init_diff_rev = init_diff_rev.permute(1, 0, 2)
        else:
            init_diff_rev = None
            # complete noise
            
        with torch.no_grad():
            if return_init_noise:
                init_noise, diff_out = self._diffusion_reverse(text_emb, 
                                                text_mask,
                                                cond_emb_motion,
                                                cond_motion_mask,
                                                mask_target, 
                                                init_vec=init_diff_rev,
                                                init_from=init_vec_method,
                                                gd_text=gd_text, 
                                                gd_motion=gd_motion,
                                                return_init_noise=return_init_noise,
                                                mode=condition_mode,
                                                steps_num=num_diff_steps)
                return init_noise, diff_out.permute(1, 0, 2)

            else:
                diff_out = self._diffusion_reverse(text_emb, 
                                                text_mask,
                                                cond_emb_motion,
                                                cond_motion_mask,
                                                mask_target, 
                                                init_vec=init_diff_rev,
                                                init_from=init_vec_method,
                                                gd_text=gd_text, 
                                                gd_motion=gd_motion,
                                                return_init_noise=return_init_noise,
                                                mode=condition_mode,
                                                steps_num=num_diff_steps)

            return diff_out.permute(1, 0, 2)

        pass
    
    def generate_motion(self, texts_cond, motions_cond,
                        lens_src, lens_tgt,
                        mask_source, mask_target):

        src_dict = {'length': lens_src,
                    'mask': mask_source,
                    'x': motions_cond.permute(1, 0, 2)}
        src_lat, src_distr = self.tmr_model.encode(src_dict,
                                                   modality='motion',
                                                   return_distribution=True)
        text_lat, _ = self.text_encoder(texts_cond)
        fused_src_txt = self.correction_module(src_lat, text_lat.squeeze())
        fused_distr = self.text_src_distr_enc(fused_src_txt)
        one_sample = fused_distr.rsample()
        # during training
        dec_mot = self.tmr_model.decode(one_sample, mask=mask_target)
        # sample
        ret = {'fused_src_txt_distr': fused_distr,
               'tgt_distr': torch.distributions.normal.Normal(tgt_distr[0],
                                                              F.softplus(tgt_distr[1])) ,
               'pred_motion_feats': dec_mot,
               'input_motion_feats': tgt}
        return ret

    def integrate_feats2motion(self, first_pose_norm, delta_motion_norm):
        """"
        Given a state [translation, orientation, pose] and state deltas,
        properly calculate the next state
        input and output are normalised features hence we first unnormalise,
        perform the calculatios and then normalise again
        """
        # unnorm features

        first_pose = self.unnorm_state(first_pose_norm)
        delta_motion = self.unnorm_delta(delta_motion_norm)

        # apply deltas
        # get velocity in global c.f. and add it to the state position
        assert 'body_transl_delta_pelv_xy' in self.input_feats
        pelvis_orient = first_pose[..., 3:9]
        R_z = get_z_rot(pelvis_orient, in_format="6d")
 
        # rotate R_z
        root_vel = change_for(delta_motion[..., :3],
                              R_z.squeeze(), forward=False)

        new_state_pos = first_pose[..., :3].squeeze() + root_vel

        # apply rotational deltas
        new_state_rot = apply_rot_delta(first_pose[..., 3:].squeeze(), 
                                        delta_motion[..., 3:],
                                        in_format="6d", out_format="6d")

        # cat and normalise the result
        new_state = torch.cat((new_state_pos, new_state_rot), dim=-1)
        new_state_norm = self.norm_state(new_state)
        return new_state_norm

    def integrate_translation(self, pelv_orient_norm, first_trans,
                              delta_transl_norm):
        """"
        Given a state [translation, orientation, pose] and state deltas,
        properly calculate the next state
        input and output are normalised features hence we first unnormalise,
        perform the calculatios and then normalise again
        """
        # B, S, 6d
        pelv_orient_unnorm = self.cat_inputs(self.unnorm_inputs(
                                                [pelv_orient_norm],
                                                ['body_orient'])
                                             )[0]
        # B, S, 3
        delta_trans_unnorm = self.cat_inputs(self.unnorm_inputs(
                                                [delta_transl_norm],
                                                ['body_transl_delta_pelv'])
                                                )[0]
        # B, 1, 3
        first_trans = self.cat_inputs(self.unnorm_inputs(
                                                [first_trans],
                                                ['body_transl'])
                                          )[0]

        # apply deltas
        # get velocity in global c.f. and add it to the state position
        assert 'body_transl_delta_pelv' in self.input_feats
        pelv_orient_unnorm_rotmat = transform_body_pose(pelv_orient_unnorm,
                                                        "6d->rot")
        trans_vel_pelv = change_for(delta_trans_unnorm,
                                    pelv_orient_unnorm_rotmat,
                                    forward=False)

        # new_state_pos = prev_trans_norm.squeeze() + trans_vel_pelv
        full_trans_unnorm = torch.cumsum(trans_vel_pelv,
                                          dim=1) + first_trans
        full_trans_unnorm = torch.cat([first_trans,
                                       full_trans_unnorm], dim=1)
        return full_trans_unnorm

    def diffout2motion(self, diffout, full_deltas=False):
        if diffout.shape[1] == 1:
            rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                            diffout,
                                                            self.input_feats_dims
                                                            ),
                                          self.input_feats))[0]
            full_motion_unnorm = rots_unnorm
        elif full_deltas:
            # FIRST POSE FOR GENERATION & DELTAS FOR INTEGRATION
            first_pose = diffout[:, :1]
            delta_feats = diffout[:, 1:]

            # FORWARD PASS 
            full_mot = [first_pose.squeeze()[None]]
            prev_pose = first_pose
            for i in range(delta_feats.shape[1]):
                cur_pose = self.integrate_feats2motion(prev_pose.squeeze(),
                                                    delta_feats[:, i])
                prev_pose = cur_pose
                full_mot.append(cur_pose[None])
        
            full_motion_norm = torch.cat(full_mot, dim=0)
            full_motion_unnorm = self.unnorm_state(full_motion_norm)

        else:
            # FIRST POSE FOR GENERATION & DELTAS FOR INTEGRATION
            first_trans = torch.zeros(*diffout.shape[:-1], 3,
                                      device=self.device)[:, [0]]
            if "body_orient_delta" in self.input_feats:
                delta_trans = diffout[..., 6:9]
                pelv_orient = diffout[..., 9:15]

                # for i in range(1, delta_trans.shape[1]):
                full_trans_unnorm = self.integrate_translation(pelv_orient[:, :-1],
                                                            first_trans,
                                                            delta_trans[:, 1:])
                rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                                diffout[..., 9:],
                                                        self.input_feats_dims[2:]),
                                                self.input_feats[2:])
                                                )[0]
                full_motion_unnorm = torch.cat([full_trans_unnorm,
                                                rots_unnorm], dim=-1)

            else:
                delta_trans = diffout[..., :3]
                pelv_orient = diffout[..., 3:9]
                # for i in range(1, delta_trans.shape[1]):
                full_trans_unnorm = self.integrate_translation(pelv_orient[:, :-1],
                                                            first_trans,
                                                            delta_trans[:, 1:])
                rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                                diffout[..., 3:],
                                                        self.input_feats_dims[1:]),
                                                self.input_feats[1:])
                                                )[0]
                full_motion_unnorm = torch.cat([full_trans_unnorm,
                                                rots_unnorm], dim=-1)
            

        return full_motion_unnorm

    def allsplit_step(self, split: str, batch, batch_idx):
        from src.data.tools.tensors import lengths_to_mask
        input_batch = self.norm_and_cat(batch, self.input_feats)
        for k, v in input_batch.items():
            if self.input_deltas:
                batch[f'{k}_motion'] = v[1:]
            else:
                batch[f'{k}_motion'] = v
                # batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]
        
        if self.motion_condition:
            mask_source, mask_target = self.prepare_mot_masks(batch['length_source'],
                                                              batch['length_target'],
                                                              max_len=None)
        else:
            mask_target = lengths_to_mask(batch['length_target'],
                                          device=self.device)

            batch['length_source'] = None
            batch['source_motion'] = None
            mask_source = None

        actual_target_lens = batch['length_target']

        gt_lens_tgt = batch['length_target']
        gt_lens_src = batch['length_source']

        gt_texts = batch['text']
        gt_keyids = batch['id']
        self.batch_size = len(gt_texts)

        dif_dict = self.train_vae_forward(batch, mask_source, mask_target)

        # rs_set Bx(S+1)xN --> first pose included
        if self.loss_on_positions:
            total_loss, loss_dict = self.compute_losses(dif_dict,
                                                        batch['body_joints_target'],
                                                        mask_source, 
                                                        mask_target)

        else:
            total_loss, loss_dict = self.compute_losses(dif_dict,
                                                        mask_source, 
                                                        mask_target)

        loss_dict_to_log = {f'losses/{split}/{k}': v for k, v in 
                            loss_dict.items()}
        self.log_dict(loss_dict_to_log, on_epoch=True, 
                      batch_size=self.batch_size)
 
        return total_loss
