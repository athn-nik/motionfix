import inspect
import os
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from os.path import join as pjoin
from src.model.losses.compute_mld import MLDLosses
from src.model.base import BaseModel
from src.model.utils.tools import remove_padding
from .base import BaseModel
import spacy
from torch import nn
from hydra.utils import instantiate
from src.model.metrics import ComputeMetricsSinc

# NLP_PROC = spacy.load("en_core_web_sm")

class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, conf_for_transforms, path_conf, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.transforms = instantiate(conf_for_transforms)
        self.Datastruct = self.transforms.Datastruct

        self.stage = 'diffusion'
        self.condition = 'text'
        self.is_vae = True
        self.predict_epsilon = True
        self.latent_dim = [1, 256]
        self.guidance_scale = 7.5
        self.guidance_uncodp = 0.1
        self.metrics = ComputeMetricsSinc()

        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")
        from src.model.textencoder.distilbert import DistilbertEncoderBase
        self.text_encoder = DistilbertEncoderBase(modelpath=f'{path_conf.deps}/distilbert-base-uncased')

        from src.model.motionencoder.actor import ActorAgnosticEncoder
        from src.model.motiondecoder.actor import ActorAgnosticDecoder
        self.vae_encoder = ActorAgnosticEncoder(vae=True, 
                                                nfeats=135, 
                                                num_heads=4, 
                                                num_layers=6)
        self.vae_decoder = ActorAgnosticDecoder(vae=True, 
                                                nfeats=135, 
                                                num_heads=4,
                                                num_layers=6)

        # Don't train the motion encoder and decoder
        self.vae_encoder.training = False
        for p in self.vae_encoder.parameters():
            p.requires_grad = False


        # Don't train the motion encoder and decoder
        self.vae_decoder.training = False
        for p in self.vae_decoder.parameters():
            p.requires_grad = False
        
        from src.model.mld_denoiser import MldDenoiser
        self.denoiser = MldDenoiser()

        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'
        from diffusers import DDIMScheduler
        from diffusers import DDPMScheduler

        self.scheduler = DDIMScheduler(num_train_timesteps=1000,
                                       beta_start=0.00085,
                                       beta_end=0.012,
                                       beta_schedule='scaled_linear',
                                       clip_sample=False,
                                       set_alpha_to_one=False,
                                       steps_offset=1)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                             beta_start=0.00085,
                                             beta_end=0.012,
                                             beta_schedule='scaled_linear',
                                             variance_type='fixed_small',
                                             clip_sample=False)

        self.optimizer = AdamW(lr=1e-4,
                                params=self.parameters())

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }



        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

    def gerund_augment(self, text_list):
        text_dur_gerund = []
        for x in text_list:

            occ = 0
            sample_ger = []

            for wd in NLP_PROC(x):
                if wd.pos_ == 'VERB':
                    occ += 1
                if occ == 2 and wd.pos_ == 'VERB':
                    if wd.text.endswith('ing'):
                        sample_ger.append(wd.text)
                    else:
                        sample_ger.append(f'{wd.text}ing') 
                else:
                    sample_ger.append(wd.text) 

            sample_ger = ' '.join(sample_ger)
            text_dur_gerund.append(sample_ger)
        return text_dur_gerund

    def rule_based_concat(self, texts, conj_word=None):
        conj_word_dict = {'while':0, 'sim': 1, ',': 2, 'same_time':3, 'during': 4}        
        from random import randint
        texts_wl = [" while ".join(x) for x in texts]
        texts_wl = self.gerund_augment(texts_wl)

        texts_sim = [(" <and> ".join(x),)  for x in texts]
        texts_sim = [ f"simultaneously {x[0]}" if '<and>' in x[0] else x[0] \
                        for x in texts_sim ]
        texts_sim = [ x.replace('<and>', 'and') for x in texts_sim ]

        texts_com = [", ".join(x) for x in texts]

        texts_and_same = [ (f" <and> ".join(x),) for x in texts]
        texts_and_same = [ f"{x[0]} at the same time"\
            if '<and>' in x[0] else x[0] for x in texts_and_same ]
        texts_and_same = [ x.replace('<and>', 'and') for x in texts_and_same ]
        
        texts_dur = [ f" during ".join(x) for x in texts]
        texts_dur = self.gerund_augment(texts_dur)        
        
        text_aug_batch = []
        conj_word = 'same_time'
        for augm_text_el in zip(texts_wl, texts_sim,
                                               texts_com, texts_and_same,
                                               texts_dur):
            if self.training:
                text_aug_batch.append((augm_text_el[randint(0, 4)],))
            else:
                if conj_word is None:
                    augm_idx = 0
                else:
                    augm_idx = conj_word_dict[conj_word]
                text_aug_batch.append((augm_text_el[augm_idx],))

        assert len(text_aug_batch) == len(texts)

        assert sum([len(x) for x in text_aug_batch]) == len(texts)
        return text_aug_batch
    
    @torch.no_grad()
    def transform_batch_to_mixed_synthetic(self, batch):
        from src.tools.frank import combine_motions
        from src.data.tools.collate import collate_tensor_with_padding

        motion_lst = []
        lens_mot = []
        self.transforms.rots2rfeats = self.transforms.rots2rfeats.to('cpu')
        for idx, x in enumerate(batch['datastruct_a']):
            if 'synth' in batch['keyid'][idx]:
                motion_a = self.transforms.rots2rfeats.inverse(batch['datastruct_a'][idx].detach().cpu())
                motion_b = self.transforms.rots2rfeats.inverse(batch['datastruct_b'][idx].detach().cpu())
               
                smpl_comb = combine_motions(motion_a, motion_b, 
                                           batch['bp-gpt'][idx][0],
                                           batch['bp-gpt'][idx][1],
                                           center=False)
                
                feats_to_train_idx = self.transforms.rots2rfeats(smpl_comb)
                curlen = len(feats_to_train_idx)
            else:
                curlen = batch['length'][idx]
                feats_to_train_idx = batch['datastruct'][idx, :curlen].detach().cpu()
            motion_lst.append(feats_to_train_idx)
            lens_mot.append(curlen)
        mot_collated = collate_tensor_with_padding(motion_lst)
        mot_collated = self.transforms.Datastruct(features=mot_collated)
        return lens_mot, mot_collated


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

    def forward(self, text_pairs, lens):
        # texts = batch["text"]
        # lengths = batch["length"]
        lengths = lens
        texts = [" while ".join(x) for x in text_pairs]
        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(texts)
                if self.condition == 'text':
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            text_emb = self.text_encoder.get_last_hidden_state(texts)
            z = self._diffusion_reverse(text_emb, lengths)
        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)
        with torch.no_grad():
            # ToDo change mcross actor to same api
            if self.vae_type in ["mld","actor"]:
                feats_rst = self.vae_decoder(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)

        rotations_datastruct = self.transforms.rots2rfeats.inverse(feats_rst.detach().cpu())
        # from sinc.render import render_animation

        # self.transforms.rots2joints.jointstype = 'smplh'
        # joints = self.transforms.rots2joints(rotations)
        # import ipdb; ipdb.set_trace()
        # render_animation(joints, output='./some_anim.mp4', title='what', fps=30)
        return rotations_datastruct
        #return remove_padding(joints, lengths)

    def gen_from_latent(self, batch):
        z = batch["latent"]
        lengths = batch["length"]

        feats_rst = self.vae.decode(z, lengths)
        rotations = self.transforms.rots2rfeats.inverse(feats_rst.detach().cpu())
        joints = self.transforms.rots2joints(rotations.detach().cpu())
        # feats => joints
        return remove_padding(joints, lengths)

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
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
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
    
    def _diffusion_reverse_tsne(self, encoder_hidden_states, lengths=None):
        # init latents
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        if self.vae_type == "no":
            assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
            latents = torch.randn(
                (bsz, max(lengths), self.cfg.DATASET.NFEATS),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
        else:
            latents = torch.randn(
                (bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        # reverse
        latents_t = []
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample
            # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
            latents_t.append(latents.permute(1,0,2))
        # [1, batch_size, latent_dim] -> [t, batch_size, latent_dim]
        latents_t = torch.cat(latents_t)
        return latents_t

    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0]
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        if self.vae_type in ["mld", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        # prepare for metric
        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        if self.condition == "text":
            joints_rst = self.feats2joints(feats_rst)
            joints_ref = self.feats2joints(feats_ref)
        elif self.condition == "action":
            mask = batch["mask"]
            joints_rst = self.feats2joints(feats_rst, mask)
            joints_ref = self.feats2joints(feats_ref, mask)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set

    def train_diffusion_forward(self, batch):

        feats_ref = batch["datastruct"].features
        lengths = batch["length"]
        # motion encode
        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                dist = self.vae_encoder(feats_ref, lengths)
                z = self.sample_from_distribution(dist)
            elif self.vae_type == "no":
                z = feats_ref.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor")

        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            # classifier free guidance: randomly drop text during training
            text = [
                "" if np.random.rand(1) < self.guidance_uncodp else i
                for i in text
            ]
            text = [ "" if np.random.rand(1) < self.guidance_uncodp else i for i in text]

            # text encode
            cond_emb = self.text_encoder.get_last_hidden_state(text)
        elif self.condition in ['action']:
            action = batch['action']
            # text encode
            cond_emb = action
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z.squeeze(2), cond_emb, lengths)
        return {**n_set}

    def test_diffusion_forward(self, batch, finetune_decoder=False):
        lengths = batch["length"]

        if self.condition in ["text", "text_uncond"]:
            # get text embeddings
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(lengths)
                if self.condition == 'text':
                    texts = batch["text"]
                    uncond_tokens.extend(texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts = uncond_tokens
            cond_emb = self.text_encoder.get_last_hidden_state(texts)
        elif self.condition in ['action']:
            cond_emb = batch['action']
            if self.do_classifier_free_guidance:
                cond_emb = torch.cat(
                    cond_emb,
                    torch.zeros_like(batch['action'],
                                     dtype=batch['action'].dtype))
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # diffusion reverse
        with torch.no_grad():
            z = self._diffusion_reverse(cond_emb, lengths)

        with torch.no_grad():
            if self.vae_type in ["mld", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
            else:
                raise TypeError("vae_type must be mcross or actor or mld")

        joints_rst = self.feats2joints(feats_rst)

        rs_set = {
            "m_rst": feats_rst,
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_t": z.permute(1, 0, 2),
            "joints_rst": joints_rst,
        }

        # prepare gt/refer for metric
        if "motion" in batch.keys() and not finetune_decoder:
            feats_ref = batch["motion"].detach()
            with torch.no_grad():
                if self.vae_type in ["mld", "vposert", "actor"]:
                    motion_z, dist_m = self.vae.encode(feats_ref, lengths)
                    recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
                elif self.vae_type == "no":
                    motion_z = feats_ref
                    recons_z = feats_rst

            joints_ref = self.feats2joints(feats_ref)

            rs_set["m_ref"] = feats_ref
            rs_set["lat_m"] = motion_z.permute(1, 0, 2)
            rs_set["lat_rm"] = recons_z.permute(1, 0, 2)
            rs_set["joints_ref"] = joints_ref
        return rs_set



    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    @torch.no_grad()
    def transform_batch_to_mixed_synthetic(self, batch):
        from src.tools.frank import combine_motions
        from src.data.tools.collate import collate_tensor_with_padding
        motion_lst = []
        lens_mot = []
        self.transforms.rots2rfeats = self.transforms.rots2rfeats.to('cpu')
        for idx, x in enumerate(batch['datastruct_a']):
            if 'synth' in batch['keyid'][idx]:
                motion_a = self.transforms.rots2rfeats.inverse(batch['datastruct_a'][idx].detach().cpu())
                motion_b = self.transforms.rots2rfeats.inverse(batch['datastruct_b'][idx].detach().cpu())
               
                smpl_comb = combine_motions(motion_a, motion_b, 
                                           batch['bp-gpt'][idx][0],
                                           batch['bp-gpt'][idx][1],
                                           center=False)
                
                feats_to_train_idx = self.transforms.rots2rfeats(smpl_comb)
                curlen = len(feats_to_train_idx)
            else:
                curlen = batch['length'][idx]
                feats_to_train_idx = batch['datastruct'][idx, :curlen].detach().cpu()
            motion_lst.append(feats_to_train_idx)
            lens_mot.append(curlen)
        mot_collated = collate_tensor_with_padding(motion_lst)
        mot_collated = self.transforms.Datastruct(features=mot_collated)
        return lens_mot, mot_collated
    
    def on_train_epoch_end(self):
        return self.allsplit_epoch_end("train")

    def on_validation_epoch_end(self):
        # # ToDo
        # # re-write vislization checkpoint?
        # # visualize validation
        # parameters = {"xx",xx}
        # vis_path = viz_epoch(self, dataset, epoch, parameters, module=None,
        #                         folder=parameters["folder"], writer=None, exps=f"_{dataset_val.dataset_name}_"+val_set)
        return self.allsplit_epoch_end("val")

    def on_test_epoch_end(self):

        return self.allsplit_epoch_end("test")

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def allsplit_epoch_end(self, split: str ):
        dico = {}

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })

        if split in ["val"]:
            metrics_dict = self.metrics.compute()
            dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items() if '_mean_' in metric})
            if 'Temos_Score' in metrics_dict:
                dico.update({f"Metrics/Temos_Score": metrics_dict['Temos_Score']})
        if split != "test":
            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.current_epoch),
            })
        # don't write sanity check into log
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def allsplit_step(self, split: str, batch, batch_idx):
        import ipdb; ipdb.set_trace()
        lens, motions_ds = self.transform_batch_to_mixed_synthetic(batch)
        del batch['datastruct_a']
        del batch['datastruct_b']
        del batch['datastruct']
        
        texts = self.rule_based_concat(batch['text'], conj_word=None)

        bs = len(texts)
        # number of texts for each motion

        # pack indices to use them for packing the batch correctly after
        text_and_idx = [(text, y) for texts, y in zip(texts, range(bs))
                        for text in texts]
        # text and in which batch element each text belongs
        texts, indices = zip(*text_and_idx)
        texts = list(texts)

        batch['datastruct'] = motions_ds.to(self.device)
        batch['length'] = lens
        batch['text'] = texts
                
        gt_motion_feats = batch["datastruct"]
        gt_lens = batch['length']
        gt_texts = batch['text']
        gpt_parts = batch['bp-gpt']
        # batch.clear()
        
        bs = len(gt_lens)
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")
        if split == 'val':
            if self.do_classifier_free_guidance:
                uncond_tokens = [""] * len(gt_texts)
                if self.condition == 'text':
                    uncond_tokens.extend(gt_texts)
                elif self.condition == 'text_uncond':
                    uncond_tokens.extend(uncond_tokens)
                texts_augm = uncond_tokens
            text_emb = self.text_encoder.get_last_hidden_state(texts_augm)
            z = self._diffusion_reverse(text_emb, gt_lens)
            with torch.no_grad():
                features_gen = self.vae_decoder(z.permute(1, 0, 2), gt_lens)
                datastruct_from_text = self.Datastruct(features=features_gen)

            self.metrics(gt_motion_feats.detach(),
                         datastruct_from_text.detach(),
                    #  datastruct_from_text.detach(),
                    #  gt_motion_feats.detach(), 
                        gt_lens)

        return loss
