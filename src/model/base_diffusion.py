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
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
from src.utils.genutils import dict_to_device
from src.utils.art_utils import color_map
import torch
import torch.distributions as dist
import logging
import wandb
from src.diffusion import create_diffusion

log = logging.getLogger(__name__)

class MD(BaseModel):
    def __init__(self, 
                 text_encoder: DictConfig,
                 motion_condition_encoder: DictConfig,
                 denoiser: DictConfig,
                 losses: DictConfig,
                 diff_params: DictConfig,
                 latent_dim: int,
                 nfeats: int,
                 input_feats: List[str],
                 statistics_path: str,
                 dim_per_feat: List[int],
                 norm_type: str,
                 smpl_path: str,
                 render_vids_every_n_epochs: Optional[int] = None,
                 num_vids_to_render: Optional[int] = None,
                 reduce_latents: Optional[str] = None,
                 condition: Optional[str] = "text",
                 motion_condition: Optional[str] = "source",
                 loss_func_pos: str = 'mse', # l1 mse
                 loss_func_feats: str = 'mse', # l1 mse
                 renderer = None,
                 pad_inputs = False,
                 source_encoder: str = 'trans_enc',
                 zero_len_source: bool = True,
                 copy_target: bool = False,
                 old_way: bool = False,
                 **kwargs):

        super().__init__(statistics_path, nfeats, norm_type, input_feats,
                         dim_per_feat, smpl_path, num_vids_to_render,
                         renderer=renderer)


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
        self.zero_len_source = zero_len_source
        self.copy_target = copy_target
        self.old_way = old_way
        self.smpl_path = smpl_path
        self.condition = condition
        self.motion_condition = motion_condition
        if self.motion_condition == 'source':
            if source_encoder == 'trans_enc':
                self.motion_cond_encoder = instantiate(motion_condition_encoder)
            else:
                self.motion_cond_encoder = None
        self.pad_inputs = pad_inputs 
        self.text_encoder = instantiate(text_encoder)

        # for k, v in self.render_data_buffer.items():
        #     self.store_examples[k] = {'ref': [], 'ref_features': [], 'keyids': []}
        # self.metrics = ComputeMetrics(smpl_path)
        self.input_feats = input_feats
        self.render_vids_every_n_epochs = render_vids_every_n_epochs
        self.num_vids_to_render = num_vids_to_render
        self.renderer = renderer

        # If we want to overide it at testing time
        self.reduce_latents = reduce_latents
        self.latent_dim = latent_dim
        self.diff_params = diff_params
        denoiser.motion_condition = self.motion_condition
        self.denoiser = instantiate(denoiser)
        from src.diffusion import create_diffusion

        from src.diffusion.gaussian_diffusion import ModelMeanType, ModelVarType
        from src.diffusion.gaussian_diffusion import LossType

        # ARGUMENTS FOR DIFFUSION
        # timestep_respacing --> None just the default linear things
        # noise_schedule="linear", squaredcos_cap_v2
        # use_kl=False,
        # sigma_small=False,
        # predict_xstart=False,
        # learn_sigma=True,
        # rescale_learned_sigmas=False,
        # diffusion_steps=1000
        # default: 1000 steps, linear noise schedule
        self.diffusion_process = create_diffusion(timestep_respacing=None,
                                     learn_sigma=False,
                                     sigma_small=True,
                                     diffusion_steps=self.diff_params.num_train_timesteps,
                                     noise_schedule=self.diff_params.noise_schedule,
                                     predict_xstart=False if self.diff_params.predict_type == 'noise' else True) # noise vs sample

        # self.diffusion = GaussianDiffusion(
        #       betas=betas,
        #       model_mean_type=ModelMeanType.EPSILON,
        #       model_var_type=ModelVarType.FIXED_SMALL,
        #       loss_type=LossType.MSE
        #         )
        # self.infer_scheduler = instantiate(infer_scheduler)
        # self.train_scheduler = instantiate(train_scheduler)
        # distribution of timesteps
        shape = 2.0
        scale = 1.0
        self.tsteps_distr = dist.Gamma(torch.tensor(shape),
                                       torch.tensor(scale))
        # Keep track of the losses
        # if train_scheduler.prediction_type == 'sample':
        #     self.predict_noise = False
        # else:
        #     self.predict_noise = True
        # self._losses = ModuleDict({split: instantiate(losses)
        #     for split in ["losses_train", "losses_test", "losses_val"]
        # })
        # self.losses = {key: self._losses["losses_" + key] for key in ["train",
        #                                                               "val",
        #                                                               "test"]}
        self.loss_params = losses
        # loss params terrible
        if loss_func_pos == 'l1':
            self.loss_func_pos = l1_loss
        elif loss_func_pos in ['mse', 'l2']:
            self.loss_func_pos = mse_loss
        elif loss_func_pos in ['sl1']:
            self.loss_func_pos = smooth_l1_loss

        if loss_func_feats == 'l1':
            self.loss_func_feats = l1_loss
        elif loss_func_feats in ['mse', 'l2']:
            self.loss_func_feats = mse_loss
        elif loss_func_feats in ['sl1']:
            self.loss_func_feats = smooth_l1_loss
        self.validation_step_outputs = []

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

    def _diffusion_reverse(self,
                           text_embeds, text_masks_from_enc, 
                           motion_embeds, cond_motion_masks,
                           inp_motion_mask, diff_process,
                           init_vec=None,
                           init_from='noise',
                           gd_text=None, gd_motion=None, 
                           mode='full_cond',
                           return_init_noise=False,
                           steps_num=None,
                           inpaint_dict=None,
                           use_linear=False,
                           prob_way='3way'):
        # guidance_scale_text: 7.5 #
        #  guidance_scale_motion: 1.5
        # init latents

        bsz = inp_motion_mask.shape[0]
        assert mode in ['full_cond', 'text_cond', 'mot_cond']
        assert inp_motion_mask is not None
        # len_to_gen = max(lengths) if not self.input_deltas else max(lengths) + 1
        if init_vec is None:
            initial_latents = torch.randn(
                (bsz, inp_motion_mask.shape[1], self.nfeats),
                device=inp_motion_mask.device,
                dtype=torch.float,
            )
        else:
            initial_latents = init_vec

        if gd_text is None:
            gd_scale_text = self.diff_params.guidance_scale_text
        else:
            gd_scale_text = gd_text

        if gd_motion is None:
            gd_scale_motion = self.diff_params.guidance_scale_motion
        else:
            gd_scale_motion = gd_motion

        if text_embeds is not None:
            max_text_len = text_embeds.shape[1]
        else:
            max_text_len = 0
        if self.motion_condition == 'source' and motion_embeds is not None:
            max_motion_len = cond_motion_masks.shape[1]
            text_masks = text_masks_from_enc.clone()
            if self.zero_len_source or self.old_way:
                nomotion_mask = torch.zeros(bsz, max_motion_len, 
                            dtype=torch.bool).to(self.device)
            else:
                nomotion_mask = torch.ones(bsz, max_motion_len,
                            dtype=torch.bool).to(self.device)
            motion_masks = torch.cat([nomotion_mask, 
                                      cond_motion_masks, 
                                      cond_motion_masks],
                                    dim=0)
            aug_mask = torch.cat([text_masks,
                                  motion_masks],
                                 dim=1)

        else:
            if max_text_len > 1:
                # aug_mask = text_mask
                # text_mask_aux = torch.ones(2*bsz, max_text_len, 
                #             dtype=torch.bool).to(self.device)
                aug_mask = text_masks_from_enc
            else:
                aug_mask = torch.ones(2*bsz, max_text_len, 
                            dtype=torch.bool).to(self.device)

        # Setup classifier-free guidance:
        if motion_embeds is not None:
            z = torch.cat([initial_latents, initial_latents, initial_latents], 0)
        else:
            z = torch.cat([initial_latents, initial_latents], 0)

        # y_null = torch.tensor([1000] * n, device=device)
        # y = torch.cat([y, y_null], 0)
        if use_linear:
            max_steps_diff = diff_process.num_timesteps
        else:
            max_steps_diff = None
        if motion_embeds is not None:
            model_kwargs = dict(# noised_motion=latent_model_input,
                                # timestep=t,
                                in_motion_mask=torch.cat([inp_motion_mask,
                                                        inp_motion_mask,
                                                        inp_motion_mask], 0),
                                text_embeds=text_embeds,
                                condition_mask=aug_mask,
                                motion_embeds=torch.cat([torch.zeros_like(motion_embeds),
                                                        motion_embeds,
                                                        motion_embeds], 1),
                                guidance_motion=gd_motion,
                                guidance_text_n_motion=gd_text,
                                inpaint_dict=inpaint_dict,
                                max_steps=max_steps_diff,
                                prob_way=prob_way)
        else:
            model_kwargs = dict(# noised_motion=latent_model_input,
                    # timestep=t,
                    in_motion_mask=torch.cat([inp_motion_mask,
                                            inp_motion_mask], 0),
                    text_embeds=text_embeds,
                    condition_mask=aug_mask,
                    motion_embeds=None,
                    guidance_motion=gd_motion,
                    guidance_text_n_motion=gd_text,
                    inpaint_dict=inpaint_dict,
                    max_steps=max_steps_diff)

        # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        # Sample images:
        samples = diff_process.p_sample_loop(self.denoiser.forward_with_guidance,
                                             z.shape, z, 
                                             clip_denoised=False, 
                                             model_kwargs=model_kwargs,
                                             progress=True,
                                             device=initial_latents.device,)
        if motion_embeds is not None:
            _, _, samples = samples.chunk(3, dim=0)  # Remove null class samples
        else:
            # print('YAO TEXT CONDITIONS>>>>>>')
            _, samples = samples.chunk(2, dim=0)
        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]

        final_diffout = samples.permute(1, 0, 2)
        if return_init_noise:
            return initial_latents, final_diffout
        else:
            return final_diffout
    
    def sample_timesteps(self, samples: int, sample_mode=None):
        if sample_mode is None:
            if self.trainer.current_epoch / self.trainer.max_epochs > 0.5:

                gamma_samples = self.tsteps_distr.sample((samples,))
                lower_bound = 0
                upper_bound = self.diffusion_process.num_timesteps
                scaled_samples = upper_bound * (gamma_samples / gamma_samples.max()) 
                # Convert the samples to integers
                timesteps_sampled = scaled_samples.floor().int().to(self.device)
            else:
                timesteps_sampled = torch.randint(0,
                                    self.diffusion_process.num_timesteps,
                                     (samples, ),
                                    device=self.device)
        # elif isinstance(sample_mode, list):
            
        else:
            
            if sample_mode == 'uniform':
                timesteps_sampled = torch.randint(0,
                                        self.diffusion_process.num_timesteps,
                                        (samples, ),
                                        device=self.device)
        return timesteps_sampled

    def _diffusion_process(self, input_motion_feats,
                           mask_in_mot,
                           text_encoded,
                           mask_for_condition,
                           motion_encoded=None,
                           sample=None,
                           lengths=None):
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
 
        # source_latents = self.motion_encoder.skel_embedding(source_motion_feats)    

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        input_motion_feats = input_motion_feats.permute(1, 0, 2)
        bsz = input_motion_feats.shape[0]
        # Sample a random timestep for each motion
        timesteps = self.sample_timesteps(samples=bsz,
                                          sample_mode='uniform')
        timesteps = timesteps.long()
        model_args = dict(in_motion_mask=mask_in_mot,
                        #   timestep=timesteps,
                          text_embeds=text_encoded,
                          condition_mask=mask_for_condition,
                          motion_embeds=motion_encoded)
        diff_outs = self.diffusion_process.training_losses(self.denoiser,
                                                           input_motion_feats,
                                                           timesteps,
                                                           model_args)
        return diff_outs

    def train_diffusion_forward(self, batch, mask_source_motion,
                                mask_target_motion):

        cond_emb_motion = None
        batch_size = len(batch["text"])

        if self.motion_condition == 'source':
            source_motion_condition = batch['source_motion']
            if self.motion_cond_encoder is not None:
                cond_emb_motion = self.motion_cond_encoder(source_motion_condition,
                                                           mask_source_motion)
                cond_emb_motion = cond_emb_motion.unsqueeze(0)
                mask_source_motion = torch.ones((batch_size, 1),
                                                 dtype=bool,
                                                 device=self.device)
            else:
                cond_emb_motion = source_motion_condition

        feats_for_denois = batch['target_motion']
        target_lens = batch['length_target']

        text_list = batch["text"]
        perc_uncondp = self.diff_params.prob_uncondp
        perc_drop_text = self.diff_params.prob_drop_text
        perc_drop_motion = self.diff_params.prob_drop_motion
        perc_keep_both = 1 - perc_uncondp - perc_drop_motion - perc_drop_text
        # text encode
        # cond_emb_text, text_mask = self.text_encoder(text)
        
        # ALWAYS --> [ text condition || motion condition ] 
        # row order (rows=batch size) --> ---------------
        #                                 | rows_mixed  |
        #                                 | rows_uncond |
        #                                 |rows_txt_only|
        #                                 |rows_mot_only|
        #                                 ---------------
        bs_cond = feats_for_denois.shape[1]
        if cond_emb_motion is not None:
            max_motion_len = cond_emb_motion.shape[0]
        if self.motion_condition == 'source':
            # motion should be alwasys S, B
            # text should be B, S

            mask = (torch.rand(bs_cond, 1, 1, device=cond_emb_motion.device) > perc_drop_motion).float()
            cond_emb_motion = cond_emb_motion.permute(1, 0, 2) * mask
            cond_emb_motion = cond_emb_motion.permute(1, 0, 2)
            mask_source_motion = (mask_source_motion * mask.squeeze(-1)).bool()

            text_list = [
                "" if np.random.rand(1) < perc_drop_text else i
                for i in text_list
            ]

            mask_both = (torch.rand(bs_cond, 1, 1, device=cond_emb_motion.device) > (1-perc_keep_both)).float()
            zeroed_rows_indices = torch.nonzero(mask_both.squeeze() == 0).view(-1).tolist()
            for idx in zeroed_rows_indices:
                text_list[idx] = ""
            cond_emb_motion = cond_emb_motion.permute(1, 0, 2) * mask_both
            cond_emb_motion = cond_emb_motion.permute(1, 0, 2)
            if not self.old_way:
                mask_source_motion = (mask_source_motion * mask_both.squeeze(-1)).bool()
        else:
            text_list = [
                "" if np.random.rand(1) < self.diff_params.prob_uncondp else i
                for i in text_list
            ]
        cond_emb_text, text_mask = self.text_encoder(text_list)
        max_text_len = cond_emb_text.shape[1]

        # random permutation along the batch dimension same for all
        if self.motion_condition == 'source':
            # After
            aug_mask = torch.cat([text_mask if max_text_len > 1 
                                  else torch.ones_like(text_mask),
                                  mask_source_motion], dim=1).to(self.device)

        else:
            if max_text_len > 1:
                aug_mask = text_mask
                # text_mask_aux = torch.ones(bs_cond, max_text_len, 
                #             dtype=torch.bool).to(self.device)
                # aug_mask = text_mask_aux
            else:
                aug_mask = torch.ones(bs_cond, max_text_len, 
                                      dtype=torch.bool).to(self.device)

        # diffusion process return with noise and noise_pred
        diff_outs = self._diffusion_process(feats_for_denois,
                                            mask_in_mot=mask_target_motion,
                                            text_encoded=cond_emb_text, 
                                            motion_encoded=cond_emb_motion,
                                            mask_for_condition=aug_mask)
        diff_outs['motion_mask_target'] = mask_target_motion
        return diff_outs 

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def compute_losses(self, out_dict, dataset_names):
        from torch import nn
        from src.data.tools.tensors import lengths_to_mask

        pad_mask_jts_pos = out_dict['motion_mask_target']
        pad_mask = out_dict['motion_mask_target']
        f_rg = np.cumsum([0] + self.input_feats_dims)
        all_losses_dict = {}
        tot_loss = torch.tensor(0.0, device=self.device)
        data_loss = self.loss_func_feats(out_dict['target'],
                                            out_dict['model_output'],
                                            reduction='none')
        first_pose_loss = torch.tensor(0.0)
        full_feature_loss = data_loss
        # maybe i should do weighted average .. maybe not
        unique_datasets = list(set(dataset_names))
        dataset_to_idx = {name: i for i, name in enumerate(unique_datasets)}
        dataset_indices = torch.tensor([dataset_to_idx[name] for name in dataset_names])

        # Dictionary to store losses per dataset
        dataset_losses = {name: 0.0 for name in unique_datasets}
        # Main loss calculation loop
        for i, _ in enumerate(f_rg[:-1]):
            if 'delta' in self.input_feats[i]:
                cur_feat_loss = full_feature_loss[:, 1:, f_rg[i]:f_rg[i+1]].mean(-1) * pad_mask[:, 1:]
                tot_feat_loss = cur_feat_loss.sum() / pad_mask[:, 1:].sum()
            else:
                cur_feat_loss = full_feature_loss[..., f_rg[i]:f_rg[i+1]].mean(-1) * pad_mask
                tot_feat_loss = cur_feat_loss.sum() / pad_mask.sum()

            # Update all_losses_dict with overall tot_feat_loss
            all_losses_dict.update({self.input_feats[i]: tot_feat_loss})
            tot_loss += tot_feat_loss  # Overall loss across datasets

            # Compute per-dataset losses
            for name, idx in dataset_to_idx.items():
                dataset_mask = (dataset_indices == idx)
                dataset_pad_mask = pad_mask[dataset_mask]
                if dataset_mask.any():
                    dataset_cur_feat_loss = cur_feat_loss[dataset_mask]
                    if dataset_pad_mask.sum() > 0:  # Avoid division by zero
                        dataset_tot_feat_loss = dataset_cur_feat_loss.sum() / dataset_pad_mask.sum()
                        dataset_losses[name] += dataset_tot_feat_loss  # Accumulate per-dataset loss

        # Optionally convert accumulated scalars to tensors after loop
        for name in dataset_losses:
            dataset_losses[name] = dataset_losses[name].clone().detach() / len(self.input_feats)

        tot_loss /= len(self.input_feats)
        all_losses_dict['total_loss'] = tot_loss
        all_losses_dict = all_losses_dict | dataset_losses
        return tot_loss, all_losses_dict 

    def generate_motion(self, texts_cond, motions_cond,
                        mask_source, mask_target,
                        diffusion_process, 
                        init_vec_method='noise', init_vec=None,
                        gd_text=None, gd_motion=None, 
                        return_init_noise=False, 
                        condition_mode='full_cond',
                        num_diff_steps=None, 
                        inpaint_dict=None,
                        use_linear=False,
                        prob_way='3way'
                        ):
        # uncond_tokens = [""] * len(texts_cond)
        # if self.condition == 'text':
        #     uncond_tokens.extend(texts_cond)
        # elif self.condition == 'text_uncond':
        #     uncond_tokens.extend(uncond_tokens)
        bsz, seqlen_tgt = mask_target.shape
        feat_sz = sum(self.input_feats_dims)
        if texts_cond is not None:
            no_of_texts = len(texts_cond)
            texts_cond = ['']*no_of_texts + texts_cond
            if self.motion_condition == 'source':
                texts_cond = ['']*no_of_texts + texts_cond
            text_emb, text_mask = self.text_encoder(texts_cond)

        cond_emb_motion = None
        cond_motion_mask = None
        if self.motion_condition == 'source':
            bsz, seqlen_src = mask_source.shape
            if condition_mode == 'full_cond' or condition_mode == 'mot_cond' :
                if self.motion_cond_encoder is not None:
                    source_motion_condition = motions_cond

                    cond_emb_motion = self.motion_cond_encoder(source_motion_condition, 
                                                               mask_source)
                    # assuming encoding of a single token!
                    cond_emb_motion = cond_emb_motion.unsqueeze(0)
                    cond_motion_mask = torch.ones((mask_source.shape[0], 1),
                                                dtype=bool, device=self.device)
                else:
                    source_motion_condition = motions_cond
                    cond_emb_motion = source_motion_condition
                    cond_motion_mask = mask_source
            else:
                if self.motion_cond_encoder is not None:
                    cond_emb_motion = torch.zeros(1, bsz,
                                                  self.denoiser.latent_dim,
                                                   device=self.device)
                    cond_motion_mask = torch.ones((bsz, 1),
                                                dtype=bool, device=self.device)
                else:
                    cond_emb_motion = torch.zeros(seqlen_src, bsz, feat_sz,
                                                   device=self.device)
                    cond_motion_mask = torch.ones((bsz, 1),
                                                dtype=bool, device=self.device)

        if init_vec_method == 'noise_prev':
            init_diff_rev = init_vec
        elif init_vec_method == 'source':
            init_diff_rev = motions_cond
            tgt_len = mask_target.shape[-1]
            src_len = mask_source.shape[-1]
            if tgt_len > src_len:
                init_diff_rev = torch.cat([init_diff_rev,
                                            torch.zeros((tgt_len-src_len,
                                                          *init_diff_rev.shape[1:]),
                                                        device=self.device)],
                                            dim=0)
                init_diff_rev = init_diff_rev.permute(1, 0, 2)
            else:
                init_diff_rev = init_diff_rev[:tgt_len]
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
                                                diffusion_process,
                                                init_vec=init_diff_rev,
                                                init_from=init_vec_method,
                                                gd_text=gd_text, 
                                                gd_motion=gd_motion,
                                                return_init_noise=return_init_noise,
                                                mode=condition_mode,
                                                steps_num=num_diff_steps,
                                                inpaint_dict=inpaint_dict,
                                                use_linear=use_linear,
                                                prob_way=prob_way)
                return init_noise, diff_out.permute(1, 0, 2)

            else:
                diff_out = self._diffusion_reverse(text_emb, 
                                                text_mask,
                                                cond_emb_motion,
                                                cond_motion_mask,
                                                mask_target, 
                                                diffusion_process,
                                                init_vec=init_diff_rev,
                                                init_from=init_vec_method,
                                                gd_text=gd_text, 
                                                gd_motion=gd_motion,
                                                return_init_noise=return_init_noise,
                                                mode=condition_mode,
                                                steps_num=num_diff_steps,
                                                inpaint_dict=inpaint_dict,
                                                use_linear=use_linear)

            return diff_out.permute(1, 0, 2)

    # def integrate_feats2motion(self, first_pose_norm, delta_motion_norm):
    #     """"
    #     Given a state [translation, orientation, pose] and state deltas,
    #     properly calculate the next state
    #     input and output are normalised features hence we first unnormalise,
    #     perform the calculatios and then normalise again
    #     """
    #     # unnorm features

    #     first_pose = self.unnorm_state(first_pose_norm)
    #     delta_motion = self.unnorm_delta(delta_motion_norm)

    #     # apply deltas
    #     # get velocity in global c.f. and add it to the state position
    #     assert 'body_transl_delta_pelv_xy' in self.input_feats
    #     pelvis_orient = first_pose[..., 3:9]
    #     R_z = get_z_rot(pelvis_orient, in_format="6d")
 
    #     # rotate R_z
    #     root_vel = change_for(delta_motion[..., :3],
    #                           R_z.squeeze(), forward=False)

    #     new_state_pos = first_pose[..., :3].squeeze() + root_vel

    #     # apply rotational deltas
    #     new_state_rot = apply_rot_delta(first_pose[..., 3:].squeeze(), 
    #                                     delta_motion[..., 3:],
    #                                     in_format="6d", out_format="6d")

    #     # cat and normalise the result
    #     new_state = torch.cat((new_state_pos, new_state_rot), dim=-1)
    #     new_state_norm = self.norm_state(new_state)
    #     return new_state_norm


    # def integrate_translation(self, pelv_orient_norm, first_trans,
    #                           delta_transl_norm):
    #     """"
    #     Given a state [translation, orientation, pose] and state deltas,
    #     properly calculate the next state
    #     input and output are normalised features hence we first unnormalise,
    #     perform the calculatios and then normalise again
    #     """
    #     # B, S, 6d
    #     pelv_orient_unnorm = self.cat_inputs(self.unnorm_inputs(
    #                                             [pelv_orient_norm],
    #                                             ['body_orient'])
    #                                          )[0]
    #     # B, S, 3
    #     delta_trans_unnorm = self.cat_inputs(self.unnorm_inputs(
    #                                             [delta_transl_norm],
    #                                             ['body_transl_delta_pelv'])
    #                                             )[0]
    #     # B, 1, 3
    #     first_trans = self.cat_inputs(self.unnorm_inputs(
    #                                             [first_trans],
    #                                             ['body_transl'])
    #                                       )[0]

    #     # apply deltas
    #     # get velocity in global c.f. and add it to the state position
    #     assert 'body_transl_delta_pelv' in self.input_feats
    #     pelv_orient_unnorm_rotmat = transform_body_pose(pelv_orient_unnorm,
    #                                                     "6d->rot")
    #     trans_vel_pelv = change_for(delta_trans_unnorm,
    #                                 pelv_orient_unnorm_rotmat,
    #                                 forward=False)

    #     # new_state_pos = prev_trans_norm.squeeze() + trans_vel_pelv
    #     full_trans_unnorm = torch.cumsum(trans_vel_pelv,
    #                                       dim=1) + first_trans
    #     full_trans_unnorm = torch.cat([first_trans,
    #                                    full_trans_unnorm], dim=1)
    #     return full_trans_unnorm


    def diffout2motion(self, diffout):
        if diffout.shape[1] == 1:
            rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                            diffout,
                                                            self.input_feats_dims
                                                            ),
                                          self.input_feats))[0]
            full_motion_unnorm = rots_unnorm
        else:
            # - "body_transl_delta_pelv_xy_wo_z"
            # - "body_transl_z"
            # - "z_orient_delta"
            # - "body_orient_xy"
            # - "body_pose"
            # - "body_joints_local_wo_z_rot"
            feats_unnorm = self.cat_inputs(self.unnorm_inputs(
                                            self.uncat_inputs(diffout,
                                                self.input_feats_dims),
                                            self.input_feats))[0]
            # FIRST POSE FOR GENERATION & DELTAS FOR INTEGRATION
            if "body_joints_local_wo_z_rot" in self.input_feats:
                idx = self.input_feats.index("body_joints_local_wo_z_rot")
                feats_unnorm = feats_unnorm[..., :-self.input_feats_dims[idx]]

            first_trans = torch.zeros(*diffout.shape[:-1], 3,
                                      device=self.device)[:, [0]]
            if 'z_orient_delta' in self.input_feats:
                first_orient_z = torch.eye(3, device=self.device).unsqueeze(0)  # Now the shape is (1, 1, 3, 3)
                first_orient_z = first_orient_z.repeat(feats_unnorm.shape[0], 1, 1)  # Now the shape is (B, 1, 3, 3)
                first_orient_z = transform_body_pose(first_orient_z, 'rot->6d') 

                # --> first_orient_z convert to 6d
                # integrate z orient delta --> z component tof orientation
                z_orient_delta = feats_unnorm[..., 9:15]

                from src.tools.transforms3d import apply_rot_delta, remove_z_rot, get_z_rot, change_for
                prev_z = first_orient_z 
                full_z_angle = [first_orient_z[:, None]]
                for i in range(1, z_orient_delta.shape[1]):
                    curr_z = apply_rot_delta(prev_z, z_orient_delta[:, i])
                    prev_z = curr_z.clone()
                    full_z_angle.append(curr_z[:,None])
                full_z_angle = torch.cat(full_z_angle, dim=1)
                full_z_angle_rotmat = get_z_rot(full_z_angle)
                # full_orient = torch.cat([full_z_angle, xy_orient], dim=-1)
                xy_orient = feats_unnorm[..., 3:9]
                xy_orient_rotmat = transform_body_pose(xy_orient, '6d->rot')
                # xy_orient = remove_z_rot(xy_orient, in_format="6d")

                # GLOBAL ORIENTATION
                # full_z_angle = transform_body_pose(full_z_angle_rotmat,
                #                                    'rot->6d')

                # full_global_orient = apply_rot_delta(full_z_angle,
                #                                      xy_orient)
                full_global_orient_rotmat = full_z_angle_rotmat @ xy_orient_rotmat
                full_global_orient = transform_body_pose(full_global_orient_rotmat,
                                                         'rot->6d')

                first_trans = self.cat_inputs(self.unnorm_inputs(
                                                        [first_trans],
                                                        ['body_transl'])
                                                )[0]

                # apply deltas
                # get velocity in global c.f. and add it to the state position
                assert 'body_transl_delta_pelv' in self.input_feats
                pelvis_delta = feats_unnorm[..., :3]
                trans_vel_pelv = change_for(pelvis_delta[:, 1:],
                                            full_global_orient_rotmat[:, :-1],
                                            forward=False)

                # new_state_pos = prev_trans_norm.squeeze() + trans_vel_pelv
                full_trans = torch.cumsum(trans_vel_pelv, dim=1) + first_trans
                full_trans = torch.cat([first_trans, full_trans], dim=1)

                #  "body_transl_delta_pelv_xy_wo_z"
                # first_trans = self.cat_inputs(self.unnorm_inputs(
                #                                         [first_trans],
                #                                         ['body_transl'])
                #                                 )[0]

                # pelvis_xy = pelvis_delta_xy
                # FULL TRANSLATION
                # full_trans = torch.cat([pelvis_xy, 
                #                         feats_unnorm[..., 2:3][:,1:]], dim=-1)  
                #############
                full_rots = torch.cat([full_global_orient, 
                                       feats_unnorm[...,-21*6:]],
                                      dim=-1)
                full_motion_unnorm = torch.cat([full_trans,
                                                full_rots], dim=-1)

            elif "body_orient_delta" in self.input_feats:
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
        if 'hml3d' in batch['dataset_name']:
            idx_t2m = [i for i, x in enumerate( batch['dataset_name']) 
                       if x == 'hml3d']
        # print('-------SincSynth------')
        # print(batch['dataset_name'].count('sinc_synth') / len(batch['dataset_name']))
        # print('-------H3D------')
        # print(batch['dataset_name'].count('hml3d') / len(batch['dataset_name']))
        #print('-------AMT------')
        # print(batch['dataset_name'].count('bodilex') / len(batch['dataset_name']))
        # unique_datasets = list(set(batch['dataset_name']))  # Extract unique names
        # dataset_to_idx = {name: i for i, name in enumerate(unique_datasets)}
        # # Map each batch item to its dataset index
        # dset_idxs = torch.tensor([dataset_to_idx[name] for name in dataset_names])
        # return None
        for k, v in input_batch.items():
            batch[f'{k}_motion'] = v
            # batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]
            if v.shape[0] > 1 and self.pad_inputs:
                batch[f'{k}_motion'] = torch.nn.functional.pad(v, (0, 0, 0, 0, 0,
                                                               300 - v.size(0)),
                                                           value=0)
        #if self.old_way:
        #    max_source_len = max(batch['length_source'])
        #    sliced_tensors = []
        #    for i in range(batch['source_motion'].shape[1]):
        #        sliced_tensors.append(batch['source_motion'][:max_source_len, i][:, None])
        #        batch['source_motion'] = torch.cat(sliced_tensors, dim=1)
        if 'length_source' in batch: #hml_3d only case
            if 0 in batch['length_source']:
                if self.old_way:
                    max_source_len = max(batch['length_source'])
                    sliced_tensors = []
                    for i in range(batch['source_motion'].shape[1]):
                        sliced_tensors.append(batch['source_motion'][:max_source_len, i][:, None])
                    batch['source_motion'] = torch.cat(sliced_tensors, dim=1)
                # Assuming batch['source_motion'] is already a CUDA tensor
                # Convert batch['length_source'] from list to a CUDA tensor for efficient operations
                else:
                    length_source_tensor = torch.tensor(batch['length_source'], 
                                                    device=self.device)

                    # Determine the maximum length from length_source_tensor
                    max_source_len = torch.max(length_source_tensor)
                    # Efficient slicing: cut off all sequences to the max length in one operation
                    # batch['source_motion'] = batch['source_motion'][:max_source_len]
                    # Create a mask for indices that need to be zeroed out (assuming idx_t2m is some list of indices)
                    idx_t2m_mask = torch.zeros_like(length_source_tensor, 
                                                dtype=torch.bool)
                    idx_t2m_mask[idx_t2m] = True
                    tgt_lens_to_pad = torch.tensor(batch['length_target'], device=self.device)
                    # Apply the mask to modify 'length_source_tensor' efficiently
                    length_source_tensor[idx_t2m_mask] = tgt_lens_to_pad[idx_t2m_mask]

                # Zero out the specific indices in 'source_motion'
                # Ensure 'source_motion' can handle boolean indexing along the second dimension
                    if self.copy_target:
                        batch['source_motion'][:, idx_t2m_mask] = batch['target_motion'][:, idx_t2m_mask].clone()
                    else:
                        batch['source_motion'][:, idx_t2m_mask] = 0

                    # Convert length_source_tensor back to a list and store it back in batch if necessary
                    # Note: Conversion to list happens on the CPU, so move tensor to CPU before converting
                    batch['length_source'] = length_source_tensor.tolist()
                    if self.zero_len_source:
                        batch['length_source'] = [0 if ii in idx_t2m else ll 
                                              for ii, ll in enumerate(batch['length_source']) 
                                             ]
        if self.motion_condition is not None:
            if self.pad_inputs:
                mask_source, mask_target = self.prepare_mot_masks(batch['length_source'],
                                                                  batch['length_target'],
                                                                  max_len=300)
            else:
                mask_source, mask_target = self.prepare_mot_masks(batch['length_source'],
                                                                  batch['length_target'],
                                                                  max_len=None)
        else:

            mask_target = lengths_to_mask(batch['length_target'],
                                          device=self.device)
            if v.shape[0] > 1 and self.pad_inputs:
                mask_target = F.pad(mask_target, (0, 300 - mask_target.size(1)),
                                value=0)

            batch['length_source'] = None
            batch['source_motion'] = None
            mask_source = None

        actual_target_lens = batch['length_target']

        # batch['text'] = ['']*len(batch['text'])

        gt_lens_tgt = batch['length_target']
        gt_lens_src = batch['length_source']
        batch['text'] = [el.lower() for el in batch['text']]
        gt_texts = batch['text']
        gt_keyids = batch['id']
        self.batch_size = len(gt_texts)
        dif_dict = self.train_diffusion_forward(batch,
                                                mask_source,
                                                mask_target)


        # rs_set Bx(S+1)xN --> first pose included
        total_loss, loss_dict = self.compute_losses(dif_dict, batch['dataset_name'])

        # if self.trainer.current_epoch % 100 == 0 and self.trainer.current_epoch != 0:
        #     if self.global_rank == 0 and split=='train' and batch_idx == 0:
        #         if self.renderer is not None:
        #             self.visualize_diffusion(dif_dict, actual_target_lens, 
        #                                     gt_keyids, gt_texts, 
        #                                     self.trainer.current_epoch)

        # self.losses[split](rs_set)
        # if loss is None:
        #     raise ValueError("Loss is None, this happend with torchmetrics > 0.7")
        loss_dict_to_log = {
            f'total_losses/{split}/{k}' if k not in self.input_feats
            else f'feature_losses/{split}/{k}': v 
            for k, v in loss_dict.items()
        }

        # loss_dict_to_log = {f'losses/{split}/{k}': v for k, v in 
        #                     loss_dict.items()}
        self.log_dict(loss_dict_to_log, on_epoch=True, 
                      batch_size=self.batch_size)
        import random
        if split == 'val' and self.global_rank == 0:
            from tqdm import tqdm
            gd_text = [1.5] # 3.0, 7.0]
            gd_motion = [1.5] #, 3.0, 7.0]
            guidances_mix = [(x, y) for x in gd_text for y in gd_motion]
            infer_steps = self.diffusion_process.num_timesteps

            if batch_idx == 0:
                gen_mots = {f'{s_t}txt_{s_m}mot': [] 
                            for s_t, s_m in guidances_mix}


            # prepare the motions
            # compute the metrics

            for guid_text, guid_motion in tqdm(guidances_mix):
                diffout = self.generate_motion(gt_texts, batch['source_motion'],
                                               mask_source, mask_target,
                                               self.diffusion_process,
                                               gd_motion=guid_motion,
                                               gd_text=guid_text,
                                               num_diff_steps=infer_steps)
                gen_mo = self.diffout2motion(diffout)
                gen_mots[f'{guid_text}txt_{guid_motion}mot'].append(gen_mo)
            self.validation_step_outputs.append(gen_mots)
            return {'val_motions': gen_mots}

        return total_loss
