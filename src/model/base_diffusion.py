from os import times
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
from src.tools.transforms3d import change_for, transform_body_pose, get_z_rot
from src.tools.transforms3d import apply_rot_delta
from einops import rearrange, reduce
from torch.nn.functional import l1_loss, mse_loss


class MD(BaseModel):
    def __init__(self, 
                 text_encoder: DictConfig,
                 motion_decoder: DictConfig,
                 diffusion_scheduler: DictConfig,
                 noise_scheduler: DictConfig,
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
                 reduce_latents: Optional[str] = None,
                 condition: Optional[str] = "text",
                 renderer= None,
                 **kwargs):

        super().__init__(statistics_path, nfeats, norm_type, input_feats,
                         dim_per_feat, smpl_path)
        self.condition = condition    
        self.text_encoder = instantiate(text_encoder)
        # from torch import nn
        # self.condition_encoder = nn.Linear()
        self.motion_decoder = instantiate(motion_decoder, nfeats=nfeats)
        # for k, v in self.render_data_buffer.items():
        #     self.store_examples[k] = {'ref': [], 'ref_features': [], 'keyids': []}
        self.metrics = ComputeMetrics()
        self.input_feats = input_feats
        self.render_vids_every_n_epochs = render_vids_every_n_epochs
        self.renderer = renderer

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = 1.0
        self.reduce_latents = reduce_latents
        self.latent_dim = latent_dim
        self.diff_params = diff_params
        self.denoiser = instantiate(denoiser)
        if not self.diff_params.predict_epsilon:
            diffusion_scheduler['prediction_type'] = 'sample'
            noise_scheduler['prediction_type'] = 'sample'
        
        self.scheduler = instantiate(diffusion_scheduler)
        self.noise_scheduler = instantiate(noise_scheduler)        
        # Keep track of the losses

        # self._losses = ModuleDict({split: instantiate(losses)
        #     for split in ["losses_train", "losses_test", "losses_val"]
        # })
        # self.losses = {key: self._losses["losses_" + key] for key in ["train",
        #                                                               "val",
        #                                                               "test"]}
        self.loss_params = losses
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
            (bsz, max(lengths) + 1, self.nfeats),
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
            noise_pred = self.denoiser(noised_motion=latent_model_input,
                                       timestep=t,
                                       encoder_hidden_states=encoder_hidden_states,
                                       lengths=lengths_reverse)[0]

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
    

    def _diffusion_process(self, input_motion_feats, text_encoded,
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
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=input_motion_feats.device,
        )
        timesteps = timesteps.long()
        

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_motion = self.noise_scheduler.add_noise(input_motion_feats.clone(),
                                                      noise,
                                                      timesteps)
        # Predict the noise residual
        diffusion_fw_out = self.denoiser(noised_motion=noisy_motion,
                                         timestep=timesteps,
                                         encoder_hidden_states=text_encoded,
                                         lengths=lengths, return_dict=False,)[0]


        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        # if self.losses.lmd_prior != 0.0:
        #     noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
        #     noise, noise_prior = torch.chunk(noise, 2, dim=0)
        # else:
        if not self.diff_params.predict_epsilon:
            n_set = {
                "pred_motion_feats": diffusion_fw_out,
                "noised_motion_feats": noisy_motion,
                "input_motion_feats": input_motion_feats,
                "timesteps": timesteps
            }

        else:
            n_set = {
                "noise": noise,
                "noise_pred": diffusion_fw_out,
            }

        return n_set


    def train_diffusion_forward(self, batch, batch_idx):
        
        feats_ref = batch['target_motion']
        lengths = batch['length_target']
        # motion encode
        # with torch.no_grad():
            
        # motion_feats = feats_ref.permute(1, 0, 2)

        text = batch["text"]
        # classifier free guidance: randomly drop text during training
        text = [ "" if np.random.rand(1) < self.diff_params.guidance_uncondp
                else i for i in text]

        # text encode
        cond_emb = self.text_encoder.get_last_hidden_state(text)
        
        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(feats_ref,
                                        cond_emb, lengths=lengths)
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
    def compute_joints_loss(self, out_motion, joints_gt, padding_mask):
        motion_unnorm = self.diffout2motion(out_motion['pred_motion_feats'])
        motion_unnorm = motion_unnorm.permute(1, 0, 2)
        pred_smpl_params = pack_to_render(rots=motion_unnorm[..., 3:],
                                          trans=motion_unnorm[...,:3])

        B, S = pred_smpl_params['body_transl'].shape[:2]
        pred_joints = self.run_smpl_fwd(pred_smpl_params['body_transl'],
                                        pred_smpl_params['body_orient'],
                                        pred_smpl_params['body_pose'].reshape(B,
                                                                              S, 
                                                                              63)).joints
        pred_joints = rearrange(pred_joints[:, :22], '(b s) ... -> b s ...',
                                s=S, b=B)
        loss_joints = mse_loss(pred_joints, joints_gt, reduction='none')
        loss_joints = reduce(loss_joints, 's b j d -> s b', 'mean')
        loss_joints = (loss_joints * padding_mask).sum() / padding_mask.sum()
        # import numpy as np
        # np.save('gt.npz',joints_gt[0].detach().cpu().numpy())  
        # np.save('pred.npz', pred_joints[0].detach().cpu().numpy())
        return loss_joints

    def compute_losses(self, out_dict, joints_gt, lengths):
        from torch import nn
        from src.data.tools.tensors import lengths_to_mask

        pad_mask = lengths_to_mask(lengths, self.device)
        pad_mask_jts_pos = lengths_to_mask([ll+1 for ll in lengths] , self.device)

        lparts = np.cumsum(self.input_feats_dims)
        if self.loss_params['predict_epsilon']:
            loss_func_noise = nn.MSELoss(reduction='mean')
            noise_loss = loss_func_noise(out_dict['noise_pred'],
                                         out_dict['noise'])
        # predict x
        else:
            loss_func_data = nn.MSELoss(reduction='none')

            data_loss = loss_func_data(out_dict['pred_motion_feats'],
                                       out_dict['input_motion_feats'])
            first_pose_loss = data_loss[:, 0].mean(-1)
            first_pose_loss = first_pose_loss.mean()

            deltas_pose = data_loss[:, 1:]

            trans_loss = deltas_pose[..., :lparts[0]].mean(-1)*pad_mask
            trans_loss = trans_loss.sum() / pad_mask.sum()

            orient_loss = deltas_pose[..., lparts[0]:lparts[1]].mean(-1)*pad_mask
            orient_loss = orient_loss.sum() / pad_mask.sum()

            pose_loss = deltas_pose[..., lparts[1]:lparts[2]].mean(-1)*pad_mask
            pose_loss = pose_loss.sum() / pad_mask.sum()            

            total_loss = pose_loss + trans_loss + orient_loss + first_pose_loss
        
            # total_loss = first_pose_loss
        
        if self.loss_params['lmd_prior'] != 0.0:
            # loss - prior loss
            loss_func_prior = nn.MSELoss(reduction='mean')
            prior_loss = loss_func_prior(out_dict['noise_prior'],
                                         out_dict['dist_m1'])
        loss_joints = 0
        J = 22
        joints_gt = rearrange(joints_gt, 'b s (j d) -> b s j d', j=J)

        loss_joints = self.compute_joints_loss(out_dict['pred_motion_feats'],
                                               joints_gt, 
                                               pad_mask_jts_pos)
        total_loss = total_loss + loss_joints

        return total_loss, {'loss': total_loss,
                            'pose': pose_loss,
                            'orientation': orient_loss,
                            'translation': trans_loss,
                            'first_pose_loss': first_pose_loss,
                            'joints_loss': loss_joints}

    def batch2motion(self, batch):
        batch_to_cpu = { k: v.detach().cpu() for k, v in batch.items() 
                        if torch.is_tensor(v) }

        # source motion
        source_motion_gt_pose = torch.cat([batch_to_cpu['body_orient_source'], 
                                           batch_to_cpu['body_pose_source']],
                                           dim=-1)
        source_motion_gt_trans = batch_to_cpu['body_transl_source']
        source_motion_gt = pack_to_render(rots=source_motion_gt_pose,
                                          trans=source_motion_gt_trans)
        # target motion
        target_motion_gt_pose = torch.cat([batch_to_cpu['body_orient_target'], 
                                           batch_to_cpu['body_pose_target']],
                                           dim=-1)
        target_motion_gt_trans = batch_to_cpu['body_transl_target']
        target_motion_gt = pack_to_render(rots=target_motion_gt_pose,
                                          trans=target_motion_gt_trans)

        return source_motion_gt, target_motion_gt

    def generate_motion(self, texts, lengths):
        uncond_tokens = [""] * len(texts)
        if self.condition == 'text':
            uncond_tokens.extend(texts)
        elif self.condition == 'text_uncond':
            uncond_tokens.extend(uncond_tokens)

        text_emb = self.text_encoder.get_last_hidden_state(uncond_tokens)
        diff_out = self._diffusion_reverse(text_emb, lengths)
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
        root_vel = change_for(delta_motion[..., :3], R_z.squeeze(), forward=False)

        new_state_pos = first_pose[..., :3].squeeze() + root_vel

        # apply rotational deltas
        new_state_rot = apply_rot_delta(first_pose[..., 3:].squeeze(), 
                                        delta_motion[..., 3:],
                                        in_format="6d", out_format="6d")

        # cat and normalise the result
        new_state = torch.cat((new_state_pos, new_state_rot), dim=-1)
        new_state_norm = self.norm_state(new_state)
        return new_state_norm

    def diffout2motion(self, diffout):
        # FIRST POSE FOR GENERATION & DELTAS FOR INTEGRATION
        first_pose = diffout[:, :1,]
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
        return full_motion_unnorm
    
    
    
    def visualize_diffusion(self, dif_out):       
        ##### DEBUG THE MODEL #####
        import os
        curdir = f'debug/epoch-{self.trainer.current_epoch}'
        os.makedirs(curdir, exist_ok=True)
        cur_ep = self.trainer.current_epoch
        input_motion_feats = dif_out['input_motion_feats']
        timesteps = dif_out['timesteps']
        noisy_motion = dif_out['noised_motion_feats']
        diffusion_fw_out = dif_out['pred_motion_feats']
        
        for idx in range(dif_out['input_motion_feats'].shape[0]-2):
            from src.render.mesh_viz import render_motion

            mot_from_deltas = self.diffout2motion(input_motion_feats.detach())
            mot_from_deltas = mot_from_deltas.permute(1, 0, 2)
            mot_from_deltas = mot_from_deltas[idx]
            uno_vid = pack_to_render(rots=mot_from_deltas[...,
                                                        3:].detach().cpu(),
                                        trans=mot_from_deltas[...,
                                                    :3].detach().cpu())
            render_motion(self.renderer, uno_vid, 
                            f'{curdir}/input_{idx}', 
                        text_for_vid=str(timesteps[idx].item()), 
                        pose_repr='aa')


            noisy_mot_from_deltas = self.diffout2motion(noisy_motion.detach())
            noisy_mot_from_deltas = noisy_mot_from_deltas.permute(1, 0, 2)
            noisy_mot_from_deltas = noisy_mot_from_deltas[idx]
            no_vid = pack_to_render(rots=noisy_mot_from_deltas[...,
                                                        3:].detach().cpu(),
                                        trans=noisy_mot_from_deltas[...,
                                                    :3].detach().cpu())


            render_motion(self.renderer, no_vid, f'{curdir}/noised_{idx}',
                        text_for_vid=str(timesteps[idx].item()),
                        pose_repr='aa')



            denois_mot_deltas = self.diffout2motion(diffusion_fw_out.detach())
            denois_mot_deltas = denois_mot_deltas.permute(1, 0, 2)
            denois_mot_deltas = denois_mot_deltas[idx]
            deno_vid = pack_to_render(rots=denois_mot_deltas[...,
                                                        3:].detach().cpu(),
                                        trans=denois_mot_deltas[...,
                                                    :3].detach().cpu())


            render_motion(self.renderer, deno_vid, f'{curdir}/denoised_{idx}',
                        text_for_vid=str(timesteps[idx].item()),
                        pose_repr='aa')

        ##### DEBUG THE MODEL #####


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

        batch = self.norm_and_cat(batch, self.input_feats)

        batch = self.append_first_frame(batch, which_motion='target')
        batch['length_target'] = [leng - 1 for leng in batch['length_target']]
        batch['text'] = ['']*len(batch['text'])

        gt_lens = batch['length_target']
        gt_texts = batch['text']
        self.batch_size = len(gt_texts)
        # batch['source_motion'] = batch['source_motion'].permute(1, 0, 2)
        # batch['target_motion'] = batch['target_motion'].permute(1, 0, 2)
        # batch.clear()
        # bs = len(gt_lens)
        if split in ['train',
                     'val']:
            
            dif_dict = self.train_diffusion_forward(batch, batch_idx)
            if self.trainer.current_epoch % 10 == 0 and split=='train':
                self.visualize_diffusion(dif_dict)
            # rs_set Bx(S+1)xN --> first pose included 
            total_loss, loss_dict = self.compute_losses(dif_dict,
                                                        batch['body_joints_target'],
                                                        gt_lens)

            # self.losses[split](rs_set)
            # if loss is None:
            #     raise ValueError("Loss is None, this happend with torchmetrics > 0.7")
            loss_dict_to_log = {f'losses/{split}/{k}': v for k, v in 
                                loss_dict.items()}
            self.log_dict(loss_dict_to_log, on_epoch=True,
                           batch_size=self.batch_size)
    
        if batch_idx == 0 and self.global_rank == 0 and split == 'val':
            source_motion_gt, target_motion_gt = self.batch2motion(batch)
            with torch.no_grad():
                motion_out = self.generate_motion(gt_texts, gt_lens)
                motion_unnorm = self.diffout2motion(motion_out)
                # do something with the full motion
                gen_to_render = pack_to_render(rots=motion_unnorm[...,
                                                                    3:].detach().cpu(),
                                                    trans=motion_unnorm[...,
                                                                :3].detach().cpu())
            self.render_data_buffer[split].append({
                # 'source_motion': source_motion_gt,
                'target_motion': target_motion_gt,
                'generation': gen_to_render})

        return total_loss
