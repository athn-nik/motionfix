from os import times
from typing import List, Optional, Union
from matplotlib.pylab import cond
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.nn import ModuleDict
from src.data.tools.collate import collate_tensor_with_padding
from torch.nn import functional as F
from src.data.tools.tensors import lengths_to_mask
from src.model.base import BaseModel
from src.model.metrics import ComputeMetrics
from src.model.utils.tools import remove_padding
from src.model.losses.utils import LossTracker
from src.data.tools import lengths_to_mask_njoints
from torchmetrics import MetricCollection
from src.model.losses.compute_mld import MLDLosses
import inspect
from src.model.utils.tools import remove_padding, pack_to_render
from src.tools.transforms3d import change_for, transform_body_pose, get_z_rot
from src.tools.transforms3d import apply_rot_delta
from einops import rearrange, reduce
from torch.nn.functional import l1_loss, mse_loss
from src.utils.genutils import dict_to_device
from src.utils.art_utils import color_map
import torch
import torch.distributions as dist

import wandb

class MD(BaseModel):
    def __init__(self, 
                 text_encoder: DictConfig,
                 motion_condition_encoder: DictConfig,
                 infer_scheduler: DictConfig,
                 train_scheduler: DictConfig,
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
                 loss_on_positions: Optional[bool] = False,
                 scale_loss_on_positions: Optional[int] = None,
                 loss_func_pos: str = 'mse', # l1 mse
                 loss_func_feats: str = 'mse', # l1 mse
                 renderer= None,
                 **kwargs):

        super().__init__(statistics_path, nfeats, norm_type, input_feats,
                         dim_per_feat, smpl_path, num_vids_to_render)

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
            self.motion_cond_encoder = instantiate(motion_condition_encoder)
        self.text_encoder = instantiate(text_encoder)
        self.loss_on_positions = loss_on_positions
        # from torch import nn
        # self.condition_encoder = nn.Linear()
        self.ep_start_scale = scale_loss_on_positions
        # self.motion_decoder = instantiate(motion_decoder, nfeats=nfeats)

        # for k, v in self.render_data_buffer.items():
        #     self.store_examples[k] = {'ref': [], 'ref_features': [], 'keyids': []}
        self.metrics = ComputeMetrics(smpl_path)
        self.input_feats = input_feats
        self.render_vids_every_n_epochs = render_vids_every_n_epochs
        self.num_vids_to_render = num_vids_to_render
        self.renderer = renderer

        # If we want to overide it at testing time
        self.reduce_latents = reduce_latents
        self.latent_dim = latent_dim
        self.diff_params = diff_params
        denoiser['use_deltas'] = self.input_deltas
        self.denoiser = instantiate(denoiser)

        self.infer_scheduler = instantiate(infer_scheduler)
        self.train_scheduler = instantiate(train_scheduler)        
        # distribution of timesteps
        shape = 2.0
        scale = 1.0
        self.tsteps_distr = dist.Gamma(torch.tensor(shape),
                                       torch.tensor(scale))
        # Keep track of the losses
        if train_scheduler.prediction_type == 'sample':
            self.predict_noise = False
        else:
            self.predict_noise = True
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

        if loss_func_feats == 'l1':
            self.loss_func_feats = l1_loss
        elif loss_func_feats in ['mse', 'l2']:
            self.loss_func_feats = mse_loss

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

    def _diffusion_reverse(self, text_embeds, text_masks, 
                           motion_embeds, motion_masks,
                           inp_motion_mask=None):
        # guidance_scale_text: 7.5 #
        #  guidance_scale_motion: 1.5
        # init latents
        bsz = text_embeds.shape[0]
        class_free_both = self.diff_params.guidance_scale_motion > 1.0 and\
              self.diff_params.guidance_scale_text > 1.0 
        class_free_motion = self.diff_params.guidance_scale_motion > 1.0 and\
              self.diff_params.guidance_scale_text < 1.0 
        class_free_text = self.diff_params.guidance_scale_motion < 1.0 and\
              self.diff_params.guidance_scale_text > 1.0 

        # if class_free_both:
        #     bsz = bsz // 3
        if self.motion_condition == 'source' or  self.condition in ['text',
                                                                'text_uncondp']:
            bsz = bsz // 2

        assert inp_motion_mask is not None, "no vae (diffusion only) need lengths for diffusion"
        # len_to_gen = max(lengths) if not self.input_deltas else max(lengths) + 1
        latents = torch.randn(
            (bsz, inp_motion_mask.shape[1], self.nfeats),
            device=text_embeds.device,
            dtype=torch.float,
        )

        # scale the initial noise by the standard deviation required by the scheduler

        latents = latents * self.infer_scheduler.init_noise_sigma
        # set timesteps
        self.infer_scheduler.set_timesteps(
            self.diff_params.num_inference_timesteps)
        timesteps = self.infer_scheduler.timesteps.to(text_embeds.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]

        # extra_step_kwargs = {}
        # if "eta" in set(
        #         inspect.signature(self.scheduler.step).parameters.keys()):
        #     extra_step_kwargs["eta"] = 0.0 # self.diff_params.scheduler.eta



        inp_motion_mask = torch.cat([inp_motion_mask] * 2)
        #  both_rows, uncond_rows, text_rows1, motion_rows2
        if self.motion_condition == 'source':
            condition_mask_wo_motion = self.filter_conditions(
                                        max_text_len=text_embeds.shape[1],
                                        max_motion_len=motion_embeds.shape[0],
                                        batch_size=text_embeds.shape[0], 
                                        perc_only_text=0.5,
                                        perc_only_motion=0.0,
                                        perc_text_n_motion=0.0,
                                        perc_uncond=0.5, 
                                        randomize=False)

            condition_mask_wo_motion[:, :text_embeds.shape[1]] *= text_masks
            condition_mask_both = self.filter_conditions(
                            max_text_len=text_embeds.shape[1],
                            max_motion_len=motion_embeds.shape[0],
                            batch_size=text_embeds.shape[0] // 2, 
                            perc_only_text=0.0,
                            perc_only_motion=0.0,
                            perc_text_n_motion=1.0, 
                            perc_uncond=0.0,
                            randomize=False)
            # might need to adjust for motion if it is more than 1 token
            condition_mask_both[:, :text_embeds.shape[1]] *= text_masks[bsz:]

        elif self.condition in ['text', 'text_uncondp']:
            condition_mask_only_text = self.filter_conditions(
                                        max_text_len=text_embeds.shape[1],
                                        max_motion_len=0,
                                        batch_size=text_embeds.shape[0], 
                                        perc_only_text=0.5,
                                        perc_only_motion=0.0,
                                        perc_text_n_motion=0.0, 
                                        perc_uncond=0.5,
                                        randomize=False)
            condition_mask_only_text[bsz:, :text_embeds.shape[1]] *= text_masks[bsz:]

        # reverse
        for i, t in enumerate(timesteps):

            if class_free_motion or class_free_text or class_free_both:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            # # expand the latents if we are doing classifier free guidance
            # latent_model_input = torch.cat(
            #     [latents] * 2) if class_free else latents


            # predict the noise residual
            if self.motion_condition == 'source':
                mot_pred_both = self.denoiser(noised_motion=latent_model_input[bsz:],
                                              in_motion_mask=inp_motion_mask[bsz:],
                                              timestep=t,
                                              text_embeds=text_embeds[bsz:],
                                              condition_mask=condition_mask_both,
                                              motion_embeds=motion_embeds[:, 
                                                                          bsz:]
                                                                          )

                mot_pred_uncond_text = self.denoiser(
                                        noised_motion=latent_model_input,
                                        in_motion_mask=inp_motion_mask,
                                        timestep=t,
                                        text_embeds=text_embeds,
                                        condition_mask=condition_mask_wo_motion[:, :-motion_embeds.shape[0]],
                                        motion_embeds=None,
                                        )
            elif self.condition in ['text', 'text_uncondp']:
                mot_pred_uncond_cond = self.denoiser(noised_motion=latent_model_input,
                                                     in_motion_mask=inp_motion_mask,
                                                     timestep=t,
                                                     text_embeds=text_embeds,
                                                     condition_mask=condition_mask_only_text,
                                                     motion_embeds=None,
                                                     )

            # perform guidance
            if self.motion_condition == 'source':
                mot_pred_uncond, mot_pred_text = mot_pred_uncond_text.chunk(2)

                motion_pred = mot_pred_uncond +\
                             self.diff_params.guidance_scale_motion * (
                             mot_pred_text - mot_pred_uncond)*\
                             self.diff_params.guidance_scale_text*(
                                mot_pred_both - mot_pred_text
                             )
            # elif class_free_motion or class_free_text:
            #     latent_model_input = torch.cat([latents] * 2)
            #     lengths_reverse = lengths * 2
            elif self.condition in ['text', 'text_uncondp']:
                mot_pred_uncond, mot_pred_text = mot_pred_uncond_cond.chunk(2)

                motion_pred = mot_pred_uncond +\
                             self.diff_params.guidance_scale_text*(
                                mot_pred_text - mot_pred_uncond
                             )


            # text_embeddings_for_guidance = encoder_hidden_states.chunk(
            #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.infer_scheduler.step(motion_pred, 
                                                t, latents).prev_sample
        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]

        latents = latents.permute(1, 0, 2)
        return latents
    
    def sample_timesteps(self, samples: int, sample_mode=None):
        if sample_mode is None:
            if self.trainer.current_epoch / self.trainer.max_epochs > 0.5:

                gamma_samples = self.tsteps_distr.sample((samples,))
                lower_bound = 0
                upper_bound = self.train_scheduler.config.num_train_timesteps
                scaled_samples = upper_bound * (gamma_samples / gamma_samples.max()) 
                # Convert the samples to integers
                timesteps_sampled = scaled_samples.floor().int().to(self.device)
            else:
                timesteps_sampled = torch.randint(0,
                                    self.train_scheduler.config.num_train_timesteps,
                                     (samples, ),
                                    device=self.device)
        else:
            if sample_mode == 'uniform':
                timesteps_sampled = torch.randint(0,
                                        self.train_scheduler.config.num_train_timesteps,
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
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
 
        # source_latents = self.motion_encoder.skel_embedding(source_motion_feats)    

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        input_motion_feats = input_motion_feats.permute(1, 0, 2)
        noise = torch.randn_like(input_motion_feats)
        bsz = input_motion_feats.shape[0]
        # Sample a random timestep for each motion
        timesteps = self.sample_timesteps(samples=bsz,
                                          sample_mode=sample)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_motion = self.train_scheduler.add_noise(input_motion_feats.clone(),
                                                      noise,
                                                      timesteps)
        # Predict the noise residual
        diffusion_fw_out = self.denoiser(noised_motion=noisy_motion,
                                         in_motion_mask=mask_in_mot,
                                         timestep=timesteps,
                                         text_embeds=text_encoded,
                                         motion_embeds=motion_encoded,
                                        #  lengths=lengths,
                                         condition_mask=mask_for_condition,
                                         return_dict=False)


        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        # if self.losses.lmd_prior != 0.0:
        #     noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
        #     noise, noise_prior = torch.chunk(noise, 2, dim=0)
        # else:
        if self.predict_noise:
            n_set = {
                "noise": noise,
                "noise_pred": diffusion_fw_out,
            }

        else:
            n_set = {
                "pred_motion_feats": diffusion_fw_out,
                "noised_motion_feats": noisy_motion,
                "input_motion_feats": input_motion_feats,
                "timesteps": timesteps
            }


        return n_set

    def filter_conditions(self, max_text_len, max_motion_len,
                          batch_size, 
                          perc_only_text=0.05,  perc_only_motion=0.05,
                          perc_text_n_motion=0.85, perc_uncond=0.05,
                          randomize=True):

        # Define the dimensions of the tensor
        M = batch_size # Number of rows (adjust as needed)
        N = max_text_len + max_motion_len  # Total number of columns

        # Calculate the number of rows for each category
        
        # rows_ones = int(M * perc_text_n_motion)
        rows_uncond = int(round(M * perc_uncond))
        rows_text_only = int(round(M * perc_only_text))
        rows_motion_only = int(round(M * perc_only_motion))
        rows_both = M - rows_text_only - rows_uncond - rows_motion_only
        all_masks = []
        if rows_both:
            # Create rows with all ones
            both_rows = torch.ones((rows_both, N), dtype=torch.float)
            all_masks.append(both_rows)
        if rows_uncond:
            # Create rows with all zeros
            uncond_rows = torch.zeros((rows_uncond, N), dtype=torch.float)
            all_masks.append(uncond_rows)

        if rows_text_only:
            # Create rows with k zeros and w ones
            single_row_text = torch.cat((torch.ones(max_text_len,
                                              dtype=torch.float),
                                         torch.zeros(max_motion_len,
                                               dtype=torch.float),
                                         ))
            text_only_row = torch.cat([single_row_text[None]] * rows_text_only,
                                      dim=0)
            all_masks.append(text_only_row)

        if rows_motion_only:
            single_row_mot = torch.cat((torch.zeros(max_text_len, 
                                                    dtype=torch.float),
                                        torch.ones(max_motion_len,
                                                   dtype=torch.float)))
            motion_only_row = torch.cat([single_row_mot[None]] * rows_motion_only,
                                        dim=0)
            all_masks.append(motion_only_row)

        # Combine the rows
        final_mask = torch.cat(all_masks, dim=0)

        if randomize:
            # Shuffle the tensor if needed
            final_mask = final_mask[torch.randperm(final_mask.size(0))]

        return final_mask.bool().to(self.device)

    def denoise_forward(self, batch, mask_source_motion,
                        mask_target_motion,
                        sample_schema='uniform'):

        cond_emb_motion = None
        if self.motion_condition == 'source':
            source_motion_condition = batch['source_motion']
            cond_emb_motion = self.motion_cond_encoder(source_motion_condition,
                                                       mask_source_motion)
            cond_emb_motion = cond_emb_motion.unsqueeze(0)

        feats_for_denois = batch['target_motion']
        target_lens = batch['length_target']
        # motion encode
        # with torch.no_grad():
            
        # motion_feats = feats_ref.permute(1, 0, 2)
        batch_size = len(batch["text"])

        text = batch["text"]

        perc_uncondp = self.diff_params.prob_uncondp
        # text encode
        cond_emb_text, text_mask = self.text_encoder(text)
        # ALWAYS --> [ text condition || motion condition ] 
        # row order (rows=batch size) --> ---------------
        #                                 | rows_mixed  |
        #                                 | rows_uncond |
        #                                 |rows_txt_only|
        #                                 |rows_mot_only|
        #                                 ---------------

        if self.motion_condition == 'source':
            aug_mask = self.filter_conditions(max_text_len=cond_emb_text.shape[1],
                                              max_motion_len=cond_emb_motion.shape[0],
                                              batch_size=batch_size, 
                                              perc_only_text=0.05,
                                              perc_only_motion=0.05,
                                              perc_text_n_motion=0.85,
                                              perc_uncond=perc_uncondp, 
                                              randomize=False)
            
            idx_text_only = int(round(batch_size * 0.05))
            idx_motion_only = int(round(batch_size * 0.05))
            idx_uncondp = int(round(batch_size * perc_uncondp))
            idx_mix = batch_size - idx_text_only - idx_uncondp - idx_motion_only
            # FIXME if motion is more than SEQ === 1 
            aug_mask[
                     (idx_uncondp+idx_mix):(idx_uncondp+idx_mix+idx_text_only), :-1] *= text_mask[(idx_uncondp+idx_mix):(idx_uncondp+idx_mix+idx_text_only)]
            # aug_mask[-idx_motion_only:] *= motion_source_mask[-idx_motion_only:]
            aug_mask = aug_mask[torch.randperm(batch_size)]
        else:
            aug_mask = self.filter_conditions(max_text_len=cond_emb_text.shape[1],
                                              max_motion_len=0,
                                              batch_size=batch_size,
                                              perc_only_text=0.9,
                                              perc_only_motion=0.00,
                                              perc_text_n_motion=0.0,
                                              perc_uncond=perc_uncondp,
                                              randomize=False)
            # final_mask = final_mask[torch.randperm(final_mask.size(0))]
            no_of_uncond = int(round(batch_size * perc_uncondp))
            aug_mask[no_of_uncond:] *= text_mask[no_of_uncond:]
            aug_mask = aug_mask[torch.randperm(batch_size)]

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(feats_for_denois,
                                        mask_in_mot=mask_target_motion,
                                        text_encoded=cond_emb_text, 
                                        motion_encoded=cond_emb_motion,
                                        mask_for_condition=aug_mask,
                                        lengths=target_lens,
                                        sample=sample_schema)
        return {**n_set}

    def train_diffusion_forward(self, batch, mask_source_motion,
                                mask_target_motion):

        cond_emb_motion = None
        if self.motion_condition == 'source':
            source_motion_condition = batch['source_motion']
            cond_emb_motion = self.motion_cond_encoder(source_motion_condition,
                                                       mask_source_motion)
            cond_emb_motion = cond_emb_motion.unsqueeze(0)

        feats_for_denois = batch['target_motion']
        target_lens = batch['length_target']
        # motion encode
        # with torch.no_grad():
            
        # motion_feats = feats_ref.permute(1, 0, 2)
        batch_size = len(batch["text"])

        text = batch["text"]
        # classifier free guidance: randomly drop text during training

        # text = [ "" if np.random.rand(1) < self.diff_params.guidance_uncondp
        #         else i for i in text]
        # # cond_emb_motion = [ "" np.random.rand(1) < self.diff_params.guidance_uncondp
        # #         else i for i in text]

        # text = [ "" if np.random.rand(1) < self.diff_params.guidance_uncondp
        #         else i for i in text]
        # text = [ "" if np.random.rand(1) < self.diff_params.guidance_uncondp
        #         else i for i in text]
        perc_uncondp = self.diff_params.prob_uncondp
        # text encode
        cond_emb_text, text_mask = self.text_encoder(text)
        # ALWAYS --> [ text condition || motion condition ] 
        # row order (rows=batch size) --> ---------------
        #                                 | rows_mixed  |
        #                                 | rows_uncond |
        #                                 |rows_txt_only|
        #                                 |rows_mot_only|
        #                                 ---------------

        if self.motion_condition == 'source':
            aug_mask = self.filter_conditions(max_text_len=cond_emb_text.shape[1],
                                              max_motion_len=cond_emb_motion.shape[0],
                                              batch_size=batch_size, 
                                              perc_only_text=0.05,
                                              perc_only_motion=0.05,
                                              perc_text_n_motion=0.85,
                                              perc_uncond=perc_uncondp, 
                                              randomize=False)
            
            idx_text_only = int(round(batch_size * 0.05))
            idx_motion_only = int(round(batch_size * 0.05))
            idx_uncondp = int(round(batch_size * perc_uncondp))
            idx_mix = batch_size - idx_text_only - idx_uncondp - idx_motion_only
            # FIXME if motion is more than SEQ === 1 
            aug_mask[
                     (idx_uncondp+idx_mix):(idx_uncondp+idx_mix+idx_text_only), :-1] *= text_mask[(idx_uncondp+idx_mix):(idx_uncondp+idx_mix+idx_text_only)]
            # aug_mask[-idx_motion_only:] *= motion_source_mask[-idx_motion_only:]
            aug_mask = aug_mask[torch.randperm(batch_size)]
        else:
            aug_mask = self.filter_conditions(max_text_len=cond_emb_text.shape[1],
                                              max_motion_len=0,
                                              batch_size=batch_size,
                                              perc_only_text=0.9,
                                              perc_only_motion=0.00,
                                              perc_text_n_motion=0.0,
                                              perc_uncond=perc_uncondp,
                                              randomize=False)
            # final_mask = final_mask[torch.randperm(final_mask.size(0))]
            no_of_uncond = int(round(batch_size * perc_uncondp))
            aug_mask[no_of_uncond:] *= text_mask[no_of_uncond:]
            aug_mask = aug_mask[torch.randperm(batch_size)]

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(feats_for_denois,
                                        mask_in_mot=mask_target_motion,
                                        text_encoded=cond_emb_text, 
                                        motion_encoded=cond_emb_motion,
                                        mask_for_condition=aug_mask,
                                        lengths=target_lens)
        return {**n_set}

    #     return self.allsplit_epoch_end("train")

    # def on_validation_epoch_end(self):
    #     # # ToDo
    #     # # re-write vislization checkpoint?
    #     # # visualize validation
    #     # parameters = {"xx",xx}
    #     # vis_path = viz_epoch(self, dataset, epoch, parameters, module=None,
    #     #                         folder=parameters["folder"], writer=None, exps=f"_{dataset_val.dataset_name}_"+val_set)
    #     return self.allsplit_epoch_end("val")

    # def on_test_epoch_end(self):
    #     return self.allsplit_epoch_end("test")

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    @property
    def jts_scale(self):
        warm_up_epochs = self.ep_start_scale
        if self.trainer.current_epoch <= warm_up_epochs:
            return 1.0
        else:
            max_loss_lambda = 10
            return min(((self.trainer.current_epoch - warm_up_epochs + 1)
                        / max_loss_lambda) * max_loss_lambda, max_loss_lambda)

    def compute_joints_loss(self, out_motion, joints_gt, padding_mask):

        from src.render.mesh_viz import render_skeleton, render_motion
        from src.model.utils.tools import remove_padding, pack_to_render
        #from src.render.video import get_offscreen_renderer
        
        if self.input_deltas:
            motion_unnorm = self.diffout2motion(out_motion['pred_motion_feats'],
                                                full_deltas=True)
            motion_unnorm = motion_unnorm.permute(1, 0, 2)
        elif self.using_deltas_transl: 
            motion_unnorm = self.diffout2motion(out_motion['pred_motion_feats'])
            # motion_unnorm = motion_unnorm.permute(1, 0, 2)
        else:
            # motion_unnorm = self.unnorm_delta(out_motion['pred_motion_feats'])
            motion_unnorm = self.unnorm_delta(out_motion['pred_motion_feats'])
            motion_norm = out_motion['pred_motion_feats']
        B, S = motion_unnorm.shape[:2]

        if False: # self.trainer.current_epoch % 20 == 0:
            iid = f'epoch-{self.trainer.current_epoch}'
            tot_dim_deltas = 0
            if self.using_deltas:
                for idx_feat, in_feat in enumerate(self.input_feats):
                    if 'delta' in in_feat:
                        tot_dim_deltas += self.input_feats_dims[idx_feat]
                motion_unnorm = motion_unnorm[..., tot_dim_deltas:]

            motion_unnorm_rd = pack_to_render(rots=motion_unnorm[..., 3:],
                                           trans=motion_unnorm[..., :3])

            B, S = motion_unnorm_rd['body_transl'].shape[:2]

            jts_unnorm = self.run_smpl_fwd(motion_unnorm_rd['body_transl'],
                                            motion_unnorm_rd['body_orient'],
                            motion_unnorm_rd['body_pose'].reshape(B, S, 63)
                                           ).joints

            jts_unnorm = rearrange(jts_unnorm[:, :22], '(b s) ... -> b s ...',
                                    s=S, b=B)
            
            render_skeleton(self.renderer,
                            positions=jts_unnorm[0].detach().cpu().numpy(),
                            filename=f'jts_unnorm_0{iid}')
            render_skeleton(self.renderer,
                            positions=jts_unnorm[1].detach().cpu().numpy(),
                            filename=f'jts_unnorm_1{iid}')

            motion_norm_rd = pack_to_render(rots=motion_norm[..., 3:],
                                         trans=motion_norm[..., :3])

            jts_norm = self.run_smpl_fwd(motion_norm_rd['body_transl'],
                                            motion_norm_rd['body_orient'],
                                            motion_norm_rd['body_pose'].reshape(B,
                                                                                S, 
                                                                                63)).joints
            jts_norm = rearrange(jts_norm[:, :22], '(b s) ... -> b s ...',
                                    s=S, b=B)
            render_skeleton(self.renderer, positions=jts_norm[0].detach().cpu().numpy(),
                            filename=f'jts_norm_0{iid}')
            render_skeleton(self.renderer, positions=jts_norm[1].detach().cpu().numpy(),
                            filename=f'jts_norm_1{iid}')


            render_skeleton(self.renderer, positions=joints_gt[0].detach().cpu().numpy(),
                            filename=f'jts_gt_0{iid}')
            render_skeleton(self.renderer, positions=joints_gt[1].detach().cpu().numpy(),
                            filename=f'jts_gt_1{iid}')
        tot_dim_deltas = 0
        if self.using_deltas:
            for idx_feat, in_feat in enumemorate(self.input_feats):
                if 'delta' in in_feat:
                    tot_dim_deltas += self.input_feats_dims[idx_feat]
            motion_unnorm = motion_unnorm[..., tot_dim_deltas:]

        pred_smpl_params = pack_to_render(rots=motion_unnorm[..., 3:],
                                          trans=motion_unnorm[...,:3])

        pred_joints = self.run_smpl_fwd(pred_smpl_params['body_transl'],
                                        pred_smpl_params['body_orient'],
                                        pred_smpl_params['body_pose'].reshape(B,
                                                                              S, 
                                                                              63)).joints
# self.run_smpl_fwd(pred_smpl_params['body_transl'],pred_smpl_params['body_orient'],pred_smpl_params['body_pose'].reshape(B, S, 63)).joints
        pred_joints = rearrange(pred_joints[:, :22], '(b s) ... -> b s ...',
                                s=S, b=B)
        

        loss_joints = self.jts_scale * self.loss_func_pos(pred_joints, 
                                                          joints_gt,
                                                          reduction='none')
        loss_joints = reduce(loss_joints, 's b j d -> s b', 'mean')
        loss_joints = (loss_joints * padding_mask).sum() / padding_mask.sum()
         
        # import numpy as np
        # np.save('gt.npz',joints_gt[0].detach().cpu().numpy())  
        # np.save('pred.npz', pred_joints[0].detach().cpu().numpy())

        return loss_joints, pred_smpl_params

    def compute_losses(self, out_dict, joints_gt, motion_mask_source, 
                       motion_mask_target):
        from torch import nn
        from src.data.tools.tensors import lengths_to_mask

        if self.input_deltas:
            pad_mask_jts_pos = motion_mask_target
            pad_mask = motion_mask_target
        else:
            pad_mask_jts_pos = motion_mask_target
            pad_mask = motion_mask_target
        f_rg = np.cumsum([0] + self.input_feats_dims)
        all_losses_dict = {}
        tot_loss = torch.tensor(0.0, device=self.device)
        if self.loss_params['predict_epsilon']:
            noise_loss = self.loss_func_feats(out_dict['noise_pred'],
                                         out_dict['noise'],
                                         reduction='none')
        # predict x
        else:

            data_loss = self.loss_func_feats(out_dict['pred_motion_feats'],
                                       out_dict['input_motion_feats'],
                                       reduction='none')
            if self.input_deltas:
                first_pose_loss = data_loss[:, 0].mean(-1)
                first_pose_loss = first_pose_loss.mean()
                full_feature_loss = data_loss[:, 1:]
            else:
                first_pose_loss = torch.tensor(0.0)
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
            # trans_loss = full_feature_loss[..., :lparts[0]].mean(-1)*pad_mask
            # trans_loss = trans_loss.sum() / pad_mask.sum()

            # orient_loss = full_feature_loss[..., 
            #                           lparts[0]:lparts[1]].mean(-1)*pad_mask
            # orient_loss = orient_loss.sum() / pad_mask.sum()

            # pose_loss = full_feature_loss[...,
            #                         lparts[1]:lparts[2]].mean(-1)*pad_mask
            # pose_loss = pose_loss.sum() / pad_mask.sum()            

            # total_loss = pose_loss + trans_loss + orient_loss + first_pose_loss
        
            # total_loss = first_pose_loss
        

        loss_joints = torch.tensor(0.0)                                 
        if self.loss_on_positions:
            J = 22
            joints_gt = rearrange(joints_gt, 'b s (j d) -> b s j d', j=J)
            loss_joints, _ = self.compute_joints_loss(out_dict, joints_gt, 
                                                      pad_mask_jts_pos)
        
        # from src.tools.transforms3d import transform_body_pose

        # pred_smpl_params = transform_body_pose(torch.cat(
        #                                             [pred_smpl['body_orient'],
        #                                              pred_smpl['body_pose']],
    #                                              dim=-1), "aa->6d")

        # gt_pose_loss_non_deltas = loss_func_data(pred_smpl_params, 
        #                                          tgt_smpl_params)
        # gt_pose_loss_non_deltas = gt_pose_loss_non_deltas.mean(-1)
        # gt_pose_loss_non_deltas = gt_pose_loss_non_deltas.sum() / pad_mask.sum()
        all_losses_dict['total_loss'] += loss_joints
        all_losses_dict['loss_joints'] = loss_joints
        return tot_loss + loss_joints, all_losses_dict 
    
    
    # {'total_loss': total_loss,
    #                         self.input_feats[2]: pose_loss,
    #                         self.input_feats[1]: orient_loss,
    #                         self.input_feats[0]: trans_loss,
    #                         'first_pose_loss': first_pose_loss,
    #                         'global_joints_loss': loss_joints,
    #                         }

    def batch2motion(self, batch, slice_til=None):
        # batch_to_cpu = { k: v.detach().cpu() for k, v in batch.items() 
        #                 if torch.is_tensor(v) }
        tot_dim_deltas = 0
        if self.using_deltas:
            for idx_feat, in_feat in enumerate(self.input_feats):
                if 'delta' in in_feat:
                    tot_dim_deltas += self.input_feats_dims[idx_feat]
        # source motion
        source_motion = batch['source_motion']
        # source_motion = self.unnorm_delta(source_motion)[..., tot_dim_deltas:]
        source_motion = self.diffout2motion(source_motion.detach())
        source_motion = source_motion.permute(1, 0, 2).detach().cpu()
        source_motion_gt = pack_to_render(rots=source_motion[..., 3:],
                                          trans=source_motion[...,:3])

        # target motion
        target_motion = batch['target_motion']
        target_motion = self.diffout2motion(target_motion.detach())        
        target_motion = target_motion.permute(1, 0, 2).detach().cpu()
        target_motion_gt = pack_to_render(rots=target_motion[..., 3:],
                                          trans=target_motion[...,:3])
        
        if slice_til is not None:
            source_motion_gt = {k: v[:slice_til] 
                                for k, v in source_motion_gt.items()}
            target_motion_gt = {k: v[:slice_til] 
                                for k, v in target_motion_gt.items()}

        return source_motion_gt, target_motion_gt

    def generate_motion(self, texts_cond, motions_cond,
                        mask_source, mask_target):
        uncond_tokens = [""] * len(texts_cond)
        if self.condition == 'text':
            uncond_tokens.extend(texts_cond)
        elif self.condition == 'text_uncond':
            uncond_tokens.extend(uncond_tokens)

        text_emb, text_mask = self.text_encoder(uncond_tokens)

        cond_emb_motion = None
        cond_motion_mask = None
        if self.motion_condition == 'source':
            source_motion_condition = motions_cond
            cond_emb_motion = self.motion_cond_encoder(source_motion_condition, 
                                                       mask_source)
            cond_motion_mask = torch.ones((cond_emb_motion.shape[0],
                                           1),
                                          dtype=bool, device=self.device)
            cond_emb_motion = torch.cat([cond_emb_motion, cond_emb_motion],
                                        dim=0).unsqueeze(0)
            
        with torch.no_grad():
            diff_out = self._diffusion_reverse(text_emb, text_mask,
                                               cond_emb_motion,
                                               cond_motion_mask,
                                               mask_target)
        return diff_out.permute(1, 0, 2)

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
        if full_deltas:
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
    
    
    def visualize_diffusion(self, dif_out, target_lens, keyids, texts_diff,
                            curepoch, return_fnames=False):       
        ##### DEBUG THE MODEL #####
        import os
        cur_epoch = curepoch
        # if not self.training
        curdir = f'debug/epoch-{cur_epoch}'
        os.makedirs(curdir, exist_ok=True)
        input_motion_feats = dif_out['input_motion_feats']
        timesteps = dif_out['timesteps']
        noisy_motion = dif_out['noised_motion_feats']
        diffusion_fw_out = dif_out['pred_motion_feats']
        if self.using_deltas:
            tot_dim_deltas = 0
            for idx_feat, in_feat in enumerate(self.input_feats):
                if 'delta' in in_feat:
                    tot_dim_deltas += self.input_feats_dims[idx_feat]

        if self.input_deltas:
            # integrate all motions
            mot_from_deltas = self.diffout2motion(input_motion_feats.detach(), 
                                                  full_deltas=True)
            noisy_mot_from_deltas = self.diffout2motion(noisy_motion.detach(), 
                                                  full_deltas=True)
            denois_mot_deltas = self.diffout2motion(diffusion_fw_out.detach(), 
                                                  full_deltas=True)
            mot_from_deltas = mot_from_deltas.permute(1, 0, 2)
            noisy_mot_from_deltas = noisy_mot_from_deltas.permute(1, 0, 2)
            denois_mot_deltas = denois_mot_deltas.permute(1, 0, 2)
        elif self.using_deltas_transl:
            mot_from_deltas = self.diffout2motion(input_motion_feats.detach())
            noisy_mot_from_deltas = self.diffout2motion(noisy_motion.detach())
            denois_mot_deltas = self.diffout2motion(diffusion_fw_out.detach())
            # mot_from_deltas = mot_from_deltas.permute(1, 0, 2)
            # noisy_mot_from_deltas = noisy_mot_from_deltas.permute(1, 0, 2)
            # denois_mot_deltas = denois_mot_deltas.permute(1, 0, 2)
        else:
            # integrate all motions
            mot_from_deltas = self.unnorm_delta(input_motion_feats.detach())
            noisy_mot_from_deltas = self.unnorm_delta(noisy_motion.detach())
            denois_mot_deltas = self.unnorm_delta(diffusion_fw_out.detach())

        log_render_dic_debug = {}
        filenames_lst = []
        for idx in range(2):
            keyid_ts_str = f'{keyids[idx]}_ts_{str(timesteps[idx].item())}'
            tstep = f'timestep: {str(timesteps[idx].item())}'
            from src.render.mesh_viz import render_motion
            from src.render.video import stack_vids
            text_vid = f'{texts_diff[idx]}'

            one_mot_from_deltas = mot_from_deltas[idx, :target_lens[idx]]
            if self.using_deltas:
                one_mot_from_deltas = one_mot_from_deltas[...,
                                                          tot_dim_deltas:]

            uno_vid = pack_to_render(rots=one_mot_from_deltas[...,
                                                        3:].detach().cpu(),
                                        trans=one_mot_from_deltas[...,
                                                    :3].detach().cpu())
            in_fl = render_motion(self.renderer, uno_vid, 
                                  f'{curdir}/input_{keyid_ts_str}', 
                                  text_for_vid=text_vid, 
                                  pose_repr='aa',
                                  color=color_map['input'])


            one_noisy_mot_from_deltas = noisy_mot_from_deltas[idx, :target_lens[idx]]
            if self.using_deltas:
                one_noisy_mot_from_deltas = one_noisy_mot_from_deltas[...,
                                                          tot_dim_deltas:]
            no_vid = pack_to_render(rots=one_noisy_mot_from_deltas[...,
                                                        3:].detach().cpu(),
                                        trans=one_noisy_mot_from_deltas[...,
                                                    :3].detach().cpu())
            noised_fl = render_motion(self.renderer, no_vid,
                                      f'{curdir}/noised_{keyid_ts_str}',
                                      text_for_vid=text_vid,
                                      pose_repr='aa',
                                      color=color_map['noised'])


            one_denois_mot_deltas = denois_mot_deltas[idx, :target_lens[idx]]
            if self.using_deltas:
                one_denois_mot_deltas = one_denois_mot_deltas[...,
                                                          tot_dim_deltas:]
            deno_vid = pack_to_render(rots=one_denois_mot_deltas[...,
                                                        3:].detach().cpu(),
                                        trans=one_denois_mot_deltas[...,
                                                    :3].detach().cpu())
            denoised_fl = render_motion(self.renderer, deno_vid, 
                                        f'{curdir}/denoised_{keyid_ts_str}',
                                        text_for_vid=text_vid,
                                        pose_repr='aa',
                                        color=color_map['denoised'])


            fname_for_stack = f'{curdir}/stak_{keyid_ts_str}_{idx}.mp4'
            stacked_name = stack_vids([in_fl, noised_fl, denoised_fl],
                                      fname=fname_for_stack,
                                      orient='h')
            logname = f'debug_renders/' + f'ep-{cur_epoch}_{keyids[idx]}_{idx}'
            log_render_dic_debug[logname] = wandb.Video(stacked_name, fps=30,
                                                        format='mp4',
                                                        caption=tstep) 
        if return_fnames:
            return filenames_lst
        else:
            self.logger.experiment.log(log_render_dic_debug)

        ##### DEBUG THE MODEL #####
    def prepare_mot_masks(self, source_lens, target_lens):
        # mask_for_features_source = []
        # mask_for_features_target = []

        # mask_source = lengths_to_mask([l - 1 
        #                                       for l in source_lens],
        #                                       self.device)
        # mask_target = lengths_to_mask([l - 1 
        #                                       for l in target_lens],
        #                                       self.device)

        # mask_deltas_source = torch.cat((torch.zeros(len(mask_deltas_source),
        #                                 1, dtype=torch.bool,
        #                                 device=self.device), 
        #                                 mask_deltas_source),
        #                                dim=1)
        # mask_deltas_target = torch.cat((torch.zeros(len(mask_deltas_source),
        #                                             1, dtype=torch.bool,
        #                                 device=self.device), mask_deltas_target),
        #                                dim=1)

        mask_target = lengths_to_mask(target_lens,
                                              self.device)
        mask_source = lengths_to_mask(source_lens,
                                              self.device)
        # for feat_name in self.input_feats:
        #     if 'delta' in feat_name:
        #         mask_for_features_source.append(mask_deltas_source)
        #         mask_for_features_target.append(mask_deltas_target)
        #     else:
        #         mask_for_features_source.append(mask_source)
        #         mask_for_features_target.append(mask_target)
        # mask_source = torch.cat(mask_for_features_source,
        #                         dim=1)
        # mask_target = torch.cat(mask_for_features_target,
        #                         dim=1)
        return mask_source, mask_target


    def allsplit_step(self, split: str, batch, batch_idx):
        # bs = len(texts)
        # number of texts for each motion
        # state_features:
        # - "body_transl"
        # - "body_orient"
        # - "body_pose"

        # delta_features:
        # - "body_transl_delta_pelv_xy"
        # - "body_orient_delta"
        # - "body_pose_delta"

        # x_features:
        # - "body_transl_z"
        # - "body_orient_xy"
        # - "body_pose"
        
        input_batch = self.norm_and_cat(batch, self.input_feats)
        for k, v in input_batch.items():
            if self.input_deltas:
                batch[f'{k}_motion'] = v[1:]
            else:
                batch[f'{k}_motion'] = v
                batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]

        mask_source, mask_target = self.prepare_mot_masks(batch['length_source'],
                                                        batch['length_target'])
        if self.input_deltas:
            batch = self.append_first_frame(batch, which_motion='target')
            batch['length_target'] = [leng - 1 for leng in batch['length_target']]
            actual_target_lens = [leng + 1 for leng in batch['length_target']]
        else:
            actual_target_lens = batch['length_target']

        # batch['text'] = ['']*len(batch['text'])

        gt_lens_tgt = batch['length_target']
        gt_lens_src = batch['length_source']

        gt_texts = batch['text']
        gt_keyids = batch['id']
        self.batch_size = len(gt_texts)

        dif_dict = self.train_diffusion_forward(batch,
                                                mask_source,
                                                mask_target)

        if self.trainer.current_epoch % 50 == 0 and self.global_rank == 0 \
            and split=='train' and batch_idx == 0:
            self.visualize_diffusion(dif_dict, actual_target_lens, 
                                     gt_keyids, gt_texts, 
                                     self.trainer.current_epoch)
        # rs_set Bx(S+1)xN --> first pose included 
        target_smpl = torch.cat([batch['body_orient_target'], 
                                 batch['body_pose_target']],
                                 dim=-1)

        total_loss, loss_dict = self.compute_losses(dif_dict,
                                                    batch['body_joints_target'],
                                                    mask_source, 
                                                    mask_target)


        # self.losses[split](rs_set)
        # if loss is None:
        #     raise ValueError("Loss is None, this happend with torchmetrics > 0.7")
        loss_dict_to_log = {f'losses/{split}/{k}': v for k, v in 
                            loss_dict.items()}
        self.log_dict(loss_dict_to_log, on_epoch=True, 
                      batch_size=self.batch_size)
        if split == 'val':
            source_motion_gt, target_motion_gt = self.batch2motion(batch)
            with torch.no_grad():
                motion_out = self.generate_motion(gt_texts, 
                                                  batch['source_motion'],
                                                  mask_source, mask_target)
                if self.input_deltas:
                    motion_unnorm = self.diffout2motion(motion_out,
                                                        full_deltas=True)
                    motion_unnorm = motion_unnorm.permute(1, 0, 2)
                if self.using_deltas_transl:
                    motion_unnorm = self.diffout2motion(motion_out)
                    # motion_unnorm = motion_unnorm.permute(1, 0, 2)
                else:
                    motion_unnorm = self.unnorm_delta(motion_out)
                # do something with the full motion
                tot_dim_deltas = 0
                if self.using_deltas:
                    for idx_feat, in_feat in enumerate(self.input_feats):
                        if 'delta' in in_feat:
                            tot_dim_deltas += self.input_feats_dims[idx_feat]
                    motion_unnorm = motion_unnorm[..., tot_dim_deltas:]

                gen_to_render = pack_to_render(rots=motion_unnorm[...,
                                                    3:].detach().cpu(),
                                               trans=motion_unnorm[...,
                                                    :3].detach().cpu())
    
            self.metrics(dict_to_device(source_motion_gt, self.device), 
                         dict_to_device(gen_to_render, self.device),
                         dict_to_device(target_motion_gt, self.device),
                         gt_lens_src, actual_target_lens)

        if batch_idx == 0 and self.global_rank == 0:
            nvds = self.num_vids_to_render
            source_motion_gt, target_motion_gt = self.batch2motion(batch, 
                                                                slice_til=nvds)
            motion_out = self.generate_motion(gt_texts[:nvds],
                                              batch['source_motion'][:, :nvds],
                                              mask_source[:nvds], 
                                              mask_target[:nvds])
            if self.input_deltas:
                motion_unnorm = self.diffout2motion(motion_out,
                                                    full_deltas=True)
                motion_unnorm = motion_unnorm.permute(1, 0, 2)
            elif self.using_deltas_transl:
                motion_unnorm = self.diffout2motion(motion_out)
                # motion_unnorm = motion_unnorm.permute(1, 0, 2)
            else:
                motion_unnorm = self.unnorm_delta(motion_out)
            
            tot_dim_deltas = 0
            if self.using_deltas:
                for idx_feat, in_feat in enumerate(self.input_feats):
                    if 'delta' in in_feat:
                        tot_dim_deltas += self.input_feats_dims[idx_feat]
                motion_unnorm = motion_unnorm[..., tot_dim_deltas:]

            gen_to_render = pack_to_render(rots=motion_unnorm[...,
                                                            3:].detach().cpu(),
                                           trans=motion_unnorm[...,
                                                            :3].detach().cpu())

            self.render_data_buffer[split].append({
                                             'source_motion': source_motion_gt,
                                             'target_motion': target_motion_gt,
                                             'generation': gen_to_render,
                                             'text_diff': gt_texts[:nvds],
                                             'keyids': gt_keyids[:nvds]}
                                             )
        return total_loss
