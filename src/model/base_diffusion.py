from typing import List, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.nn import ModuleDict
from src.data.tools.collate import collate_tensor_with_padding
from torch.nn import functional as F
from src.model.base import BaseModel
from src.model.metrics import ComputeMetrics
from src.model.utils.tools import remove_padding
from src.model.losses.utils import LossTracker
from src.data.tools import lengths_to_mask_njoints
from torchmetrics import MetricCollection
from src.model.losses.compute_mld import MLDLosses
import inspect
from src.model.utils.tools import remove_padding, pack_to_render

class MD(BaseModel):
    def __init__(self, 
                 text_encoder: DictConfig,
                 motion_encoder: DictConfig,
                 motion_decoder: DictConfig,
                 diffusion_scheduler: DictConfig,
                 noise_scheduler: DictConfig,
                 denoiser: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 diff_params: DictConfig,
                 vae: bool,
                 latent_dim: int,
                 motion_branch: bool,
                 separate_latents: Optional[bool] = False,
                 render_vids_every_n_epochs: Optional[int] = None,
                 teacher_forcing: Optional[bool] = False,
                 reduce_latents: Optional[str] = None,
                 condition: Optional[str] = "text",
                 renderer= None,
                 **kwargs):

        super().__init__()
        self.condition = condition    
        self.text_encoder = instantiate(text_encoder)
        if motion_branch:
            self.motion_encoder = instantiate(motion_encoder, nfeats=135)

        self.motion_decoder = instantiate(motion_decoder, nfeats=135)
        # for k, v in self.render_data_buffer.items():
        #     self.store_examples[k] = {'ref': [], 'ref_features': [], 'keyids': []}
        self.metrics = ComputeMetrics()
        
        self.render_vids_every_n_epochs = render_vids_every_n_epochs
        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = 1.0
        self.teacher_forcing = teacher_forcing
        self.reduce_latents = reduce_latents
        self.motion_branch = motion_branch
        self.separate_latents = separate_latents
        self.latent_dim = latent_dim
        self.diff_params = diff_params
        self.renderer = renderer
        self.denoiser = instantiate(denoiser)
        if not self.diff_params.predict_epsilon:
            diffusion_scheduler['prediction_type'] = 'sample'
            noise_scheduler['prediction_type'] = 'sample'
        
        self.scheduler = instantiate(diffusion_scheduler)
        self.noise_scheduler = instantiate(noise_scheduler)        
        # Keep track of the losses
        self._losses = ModuleDict({split: instantiate(losses)
            for split in ["losses_train", "losses_test", "losses_val"]
        })
        self.losses = {key: self._losses["losses_" + key] for key in ["train",
                                                                      "val",
                                                                      "test"]}
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

    def forward(self, text_prompts, lens):
        lengths = lens
        # diffusion reverse
        uncond_tokens = [""] * len(text_prompts)
        if self.condition == 'text':
            uncond_tokens.extend(text_prompts)
        elif self.condition == 'text_uncond':
            uncond_tokens.extend(uncond_tokens)
        texts = uncond_tokens
        text_emb = self.text_encoder.get_last_hidden_state(texts)
        motion_feats = self._diffusion_reverse(text_emb, lengths)

        with torch.no_grad():
            # ToDo change mcross actor to same api
            motion_feats = motion_feats.permute(1, 0, 2)

        return motion_feats
        #return remove_padding(joints, lengths)

    def recon_from_motion(self, batch):
        feats_ref = batch["motion"]
        length = batch["length"]

        z, dist = self.vae.encode(feats_ref, length)
        feats_rst = self.vae.decode(z, length)

        # feats => joints
        joints = self.feats2joints(feats_rst.detach().cpu())
        joints_ref = self.feats2joints(feats_ref.detach().cpu())
        return remove_padding(joints,
                              length), remove_padding(joints_ref, length)

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        bsz = bsz // 2
        assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
        latents = torch.randn(
            (bsz, max(lengths), 135),
            device=encoder_hidden_states.device,
            dtype=torch.float,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.diff_params.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = 0.0 # self.diff_params.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] *
                2)
            lengths_reverse = lengths * 2
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.diff_params.guidance_scale * (
                noise_pred_text - noise_pred_uncond)
            # text_embeddings_for_guidance = encoder_hidden_states.chunk(
            #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample
        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]

        latents = latents.permute(1, 0, 2)
        return latents
    

    def _diffusion_process(self, source_motion_feats, text_encoded, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
 
        # source_latents = self.motion_encoder.skel_embedding(source_motion_feats)    
        source_motion_feats = source_motion_feats.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(source_motion_feats)
        bsz = source_motion_feats.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=source_motion_feats.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_motion = self.noise_scheduler.add_noise(source_motion_feats.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        diffusion_fw_out = self.denoiser(sample=noisy_motion, timestep=timesteps,
                                   encoder_hidden_states=text_encoded,
                                   lengths=lengths, return_dict=False,)[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        # if self.losses.lmd_prior != 0.0:
        #     noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
        #     noise, noise_prior = torch.chunk(noise, 2, dim=0)
        # else:
        noise_pred_prior = 0
        noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": diffusion_fw_out,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.diff_params.predict_epsilon:
            n_set["pred"] = diffusion_fw_out
            n_set["diff_in"] = source_motion_feats
        return n_set


    def train_diffusion_forward(self, batch):

        feats_ref = batch['motion_s']
        lengths = batch["length_s"]
        # motion encode
        # with torch.no_grad():
            
        motion_feats = feats_ref.permute(1, 0, 2)

        text = batch["text"]
        # classifier free guidance: randomly drop text during training
        text = [ "" if np.random.rand(1) < self.diff_params.guidance_uncondp
                else i for i in text]

        # text encode
        cond_emb = self.text_encoder.get_last_hidden_state(text)
        # diffusion process return with noise and noise_pred

        n_set = self._diffusion_process(motion_feats.squeeze(2), cond_emb, lengths)
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]

        # get text embeddings
        uncond_tokens = [""] * len(lengths)
        if self.condition == 'text':
            texts = batch["text"]
            uncond_tokens.extend(texts)
        elif self.condition == 'text_uncond':
            uncond_tokens.extend(uncond_tokens)
        texts = uncond_tokens
        cond_emb = self.text_encoder.get_last_hidden_state(texts)

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            feats_rst = z.permute(1, 0, 2)

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats] <= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                motion_z = feats_ref
                recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set

    
    # def on_train_epoch_end(self):
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

    # def predict_step(self, batch, batch_idx):
    #     return self.forward(batch)

    # def allsplit_epoch_end(self, split: str ):
        # dico = {}

        # if split in ["train", "val"]:
        #     losses = self.losses[split]
        #     loss_dict = losses.compute()
        #     losses.reset()
        #     dico.update({
        #         losses.loss2logname(loss, split): value.item()
        #         for loss, value in loss_dict.items() if not torch.isnan(value)
        #     })

        # if split in ["val"]:
        #     pass
        #     # metrics_dict = self.metrics.compute()

        # if split != "test":
        #     dico.update({
        #         "epoch": float(self.trainer.current_epoch),
        #         "step": float(self.trainer.current_epoch),
        #     })
        # # don't write sanity check into log
        # if not self.trainer.sanity_checking:
        #     self.log_dict(dico, sync_dist=True, rank_zero_only=True)
    
    def batch2motion(self, batch):
        batch_to_cpu = { k: v.detach().cpu() for k, v in batch.items() 
                        if torch.is_tensor(v) }

        # source motion
        source_motion_gt_pose = torch.cat([batch_to_cpu['body_orient_s'], 
                                           batch_to_cpu['body_pose_s']],
                                           dim=-1)
        source_motion_gt_trans = batch_to_cpu['body_transl_s']
        source_motion_gt = pack_to_render(rots=source_motion_gt_pose,
                                          trans=source_motion_gt_trans)
        # target motion
        target_motion_gt_pose = torch.cat([batch_to_cpu['body_orient_t'], 
                                           batch_to_cpu['body_pose_t']],
                                           dim=-1)
        target_motion_gt_trans = batch_to_cpu['body_transl_t']
        target_motion_gt = pack_to_render(rots=target_motion_gt_pose,
                                          trans=target_motion_gt_trans)

        return source_motion_gt, target_motion_gt
    
    def allsplit_step(self, split: str, batch, batch_idx):
        # bs = len(texts)
        # number of texts for each motion

        batch['motion_s'] = torch.cat([batch['body_transl_s'],
                                     batch['body_orient_s'],
                                     batch['body_pose_s']], -1)
        batch['motion_t'] = torch.cat([batch['body_transl_t'],
                                     batch['body_orient_t'],
                                     batch['body_pose_t']], -1)

        # gt_motion_feats = batch["datastruct"]
        gt_lens = batch['length_t']
        gt_texts = batch['text']
        # batch.clear()
        # bs = len(gt_lens)
        if split in ["train", "val"]:
            rs_set = self.train_diffusion_forward(batch)

            loss = self.losses[split](rs_set)
            # if loss is None:
            #     raise ValueError("Loss is None, this happend with torchmetrics > 0.7")
        if split == 'val':
            if batch_idx == 0 and self.global_rank == 0:
                uncond_tokens = [""] * len(gt_texts)
                if self.condition == 'text':
                    uncond_tokens.extend(gt_texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts_augm = uncond_tokens
                text_emb = self.text_encoder.get_last_hidden_state(texts_augm)
                diff_out = self._diffusion_reverse(text_emb, gt_lens)
                diff_out = diff_out.permute(1, 0, 2)
                with torch.no_grad():
                    
                    source_motion_gt, target_motion_gt = self.batch2motion(batch)

                    render_dict = pack_to_render(rots=diff_out[..., 3:].detach().cpu(),
                                                 trans=diff_out[..., :3].detach().cpu())


                if batch_idx == 0 and self.global_rank == 0:
                    source_motion_gt, target_motion_gt = self.batch2motion(batch)
                    self.render_data_buffer[split].append({
                        'source_motion': source_motion_gt,
                        'target_motion': target_motion_gt,
                        'generation': render_dict})


            # self.metrics(gt_motion_feats.detach(),
            #              datastruct_from_text.detach(),
            #         #  datastruct_from_text.detach(),
            #         #  gt_motion_feats.detach(), 
            #             gt_lens)
        # loss = self.losses[split].compute()
        else:
            ## SAVE DATA FOR RENDERING LATER ##
            if batch_idx == 0 and self.global_rank == 0:
                # convert groundtruth to render-ready
                source_motion_gt, target_motion_gt = self.batch2motion(batch)
                self.render_data_buffer[split].append({
                    'source_motion': source_motion_gt,
                    'target_motion': target_motion_gt,
                    'generation': None})

        return loss
