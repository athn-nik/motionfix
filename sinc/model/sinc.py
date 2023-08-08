from typing import List, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.nn import ModuleDict
from sinc.data.tools.collate import collate_tensor_with_padding
from torch.nn import functional as F
from sinc.model.base import BaseModel
from sinc.model.metrics import ComputeMetricsSinc
from sinc.model.utils.tools import remove_padding
from sinc.model.losses.utils import LossTracker
from sinc.data.tools import lengths_to_mask_njoints
import spacy
NLP_PROC = spacy.load("en_core_web_sm")

class SINC(BaseModel):
    def __init__(self, 
                 textencoder: DictConfig,
                 motionencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 transforms: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 motion_branch: bool,
                 eval_model = None,
                 separate_latents: Optional[bool] = False,
                 nvids_to_save: Optional[int] = None,
                 teacher_forcing: Optional[bool] = False,
                 reduce_latents: Optional[str] = None,
                 concat_text_word: Optional[str] = None,
                 single_text_desc: Optional[bool] = False,
                 **kwargs):

        super().__init__()
        self.textencoder = instantiate(textencoder, nfeats=nfeats)
        if motion_branch:
            self.motionencoder = instantiate(motionencoder, nfeats=nfeats)

        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct
        self.motiondecoder = instantiate(motiondecoder, nfeats=nfeats)

        for k, v in self.store_examples.items():
            self.store_examples[k] = {'ref': [], 'ref_features': [], 'keyids': []}
        self.temos_path = '/is/cluster/fast/nathanasiou/data/motion-language/sinc-checkpoints/temos_score/bs32'
        self.eval_model = eval_model
        # eval_model = self.load_temos()
        self.metrics = ComputeMetricsSinc(eval_model)

        self.nvids_to_save = nvids_to_save
        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = 1.0
        self.teacher_forcing = teacher_forcing
        self.reduce_latents = reduce_latents
        self.motion_branch = motion_branch
        self.separate_latents = separate_latents
        self.latent_dim = latent_dim
        self.concat_text_word = concat_text_word
        self.single_text_desc = single_text_desc
                
        # Keep track of the losses
        self._losses = ModuleDict({split: instantiate(losses, vae=vae,
                                                        separate_latents=separate_latents,
                                                      _recursive_=False)
                                   for split in ["losses_train", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "val"]}

        self.__post_init__()

    # def load_temos(self):
    #     from pathlib import Path

    #     from omegaconf import OmegaConf

    #     from hydra.utils import instantiate
    #     temos_path = Path(self.temos_path)
    #     temoscfg = OmegaConf.load(temos_path / ".hydra/config.yaml")
    #     cfg = OmegaConf.merge(temoscfg, OmegaConf.load( / ".hydra/overrides.yaml")

    #     # Overload it
    #     # Instantiate all modules specified in the configs
    #     temos_model = instantiate(temoscfg.model,
    #                               nfeats=135,
    #                               logger_name="none",
    #                               nvids_to_save=None,
    #                               _recursive_=False)

    #     last_ckpt_path = temos_path / "checkpoints/last.ckpt"
    #     # Load the last checkpoint
    #     temos_model = temos_model.load_from_checkpoint(last_ckpt_path)
    #     temos_model.eval()
    #     return temos_model, temoscfg
    
    
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

    def text_to_motion_forward(self, texts: list[str], lengths: list[int], *, return_latent: bool = False,
                               return_mask: bool = False, return_motion: Optional[str] = None,
                               gpt_parts: Optional[List] = None, 
                               conjuct_word: Optional[str]=None) -> List[Tensor]:
        if self.single_text_desc:
            texts = self.rule_based_concat(texts, conjuct_word)

        if self.concat_text_word is not None:
            texts = [(f" {self.concat_text_word} ".join(x),) for x in texts]
        bs = len(texts)
        if self.concat_text_word is not None:
            texts = [(f" {self.concat_text_word} ".join(x),) for x in texts]
        # number of texts for each motion
        ntexts_lens = [len(lst) for lst in texts]

        # pack indices to use them for packing the batch correctly after
        text_and_idx = [(text, y) for texts, y in zip(texts, range(bs))
                        for text in texts]

        # text and in which batch element each text belongs
        texts, indices = zip(*text_and_idx)
        texts = list(texts)

        if self.separate_latents:
            # max sim. actions
            max_number_of_sim_texts = max(ntexts_lens)
            # masks: [bs, max_number_of_z_texts] : [[T, F, F], [T, T, F], [T, T, T] ]
            ntexts_lens = torch.tensor(ntexts_lens, device=self.device)
            mask_texts_for_dec_mem = torch.arange(max_number_of_sim_texts,
                                                  device=self.device).expand(len(ntexts_lens),
                                                                            max_number_of_sim_texts) < ntexts_lens.unsqueeze(1)
            mem_mask_njoints = mask_texts_for_dec_mem

            # Encode the text to the latent space
            if self.hparams.vae:
                distributions = self.textencoder(texts)
                latent_vectors = self.sample_from_distribution(distributions)
                mus_text, sigmas_text = distributions.loc, distributions.scale
            else:
                distribution = None
                latent_vectors = self.textencoder(texts)
            # extract each embedding for each sentences
            latent_vectors_batch = [[] for _ in range(bs)]
            mus_batch = [[] for _ in range(bs)]
            sigmas_batch = [[] for _ in range(bs)]

            for i, index in enumerate(indices):
                latent_vectors_batch[index].append(latent_vectors[i])
                mus_batch[index].append(mus_text[i])
                sigmas_batch[index].append(sigmas_text[i])

            for i, latent_vector_batch in enumerate(latent_vectors_batch):
                latent_vectors_batch[i] = torch.stack(latent_vector_batch)
                mus_batch[i] = torch.stack(mus_batch[i])
                sigmas_batch[i] = torch.stack(sigmas_batch[i])

            # extract each distribution for each sentences
            distributions_batch = []
            for mu, sigma in zip(mus_batch, sigmas_batch):
                distributions_batch.append(torch.distributions.Normal(mu,
                                                                    sigma))

            # => reduce actions weighted by GPT
            if self.reduce_latents == 'actions':
                gpt_parts = [torch.as_tensor(el, device=self.device).float() for el in gpt_parts]
                latent_vectors_weighted = torch.zeros((bs, 6, self.hparams.latent_dim),
                                                device=latent_vectors.device)
                for idx, lvs in enumerate(latent_vectors_batch): # get the GPT here give it as arg
                    probs = F.softmax(gpt_parts[idx], dim=0)
                    latent_vectors_weighted[idx] = (lvs * probs[..., None]).sum(0)

                    # lvs -> 2,6, 256, probs[]
                #    (Nt, 6, d) [Nt, 6] -> [6,d]
                latent_vectors_padded = latent_vectors_weighted # collate_tensor_with_padding(latent_vectors_weighted)
                mem_mask_njoints = None
            # => reduce bodyparts weigthed by GPT
            elif self.reduce_latents == 'bodyparts':
                gpt_parts = [torch.as_tensor(el, device=self.device).float() for el in gpt_parts]
                latent_vectors_weighted = []
                for idx, lvs in enumerate(latent_vectors_batch): # get the GPT here give it as arg
                    probs = F.softmax(gpt_parts[idx], dim=0)
                    latent_vectors_weighted.append((lvs * probs[..., None]).sum(1))
                latent_vectors_padded = collate_tensor_with_padding(latent_vectors_weighted)
                mem_mask_njoints = mask_texts_for_dec_mem
            # => reduce bodyparts weigthed by GPT and actions by average
            elif self.reduce_latents == 'bodyparts_average':
                gpt_parts = [torch.as_tensor(el, device=self.device).float() for el in gpt_parts]
                latent_vectors_weighted = []
                for idx, lvs in enumerate(latent_vectors_batch): # get the GPT here give it as arg
                    probs = F.softmax(gpt_parts[idx], dim=0)
                    latent_vectors_weighted.append((lvs * probs[..., None]).sum(1).mean(0))

                latent_vectors_padded = collate_tensor_with_padding(latent_vectors_weighted)
                latent_vectors_padded = latent_vectors_padded.unsqueeze(1)
                mem_mask_njoints = None
            # => average everything
            elif self.reduce_latents == 'average':
                latent_vectors_padded = torch.stack([lvs.mean(0) for lvs in latent_vectors_batch]).mean(1, keepdim=True)
                mem_mask_njoints = None
            else:
                latent_vectors_padded = collate_tensor_with_padding(latent_vectors_batch)
            # Decode the latent vector to a motion
            features = self.motiondecoder(latent_vectors_padded, lengths, mem_mask_njoints)
            datastruct = self.Datastruct(features=features)
            latent_vectors = latent_vectors_padded
            distributions = distributions_batch
        else:
            # Encode the text to the latent space
            if self.hparams.vae:
                distributions = self.textencoder(texts, mapping=indices)
                latent_vectors = self.sample_from_distribution(distributions)
            else:
                distributions = None
                latent_vectors = self.textencoder(texts)
            # Decode the latent vector to a motion
            features = self.motiondecoder(latent_vectors[:, None], lengths)
            datastruct = self.Datastruct(features=features)
            mask_texts_for_dec_mem = None

        if return_motion:
            return self.motion_from_datastruct(datastruct, return_type=return_motion)
        if not return_latent:
            return datastruct
        if return_mask:
            return datastruct, latent_vectors, distributions, mask_texts_for_dec_mem

        #     datastruct = self.Datastruct(features=output_features_T[0])

        #     motion = datastruct.rots
        #     rots, transl = motion.rots, motion.trans

        #     from sinc.transforms.smpl import RotTransDatastruct
        #     final_datastruct = self.Datastruct(
        #         rots_=RotTransDatastruct(rots=rots, trans=transl))

        #     if return_type == "vertices":
        #         final_motion.append(final_datastruct.vertices)
        #     elif return_type == "joints":
        #         final_motion.append(final_datastruct.joints)
        #     else:
        #         raise NotImplementedError
        # return final_motion

    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def motion_from_datastruct(self, datastruct, return_type='joints'):
        motion = datastruct.rots
        rots, transl = motion.rots, motion.trans

        from sinc.transforms.smpl import RotTransDatastruct
        final_datastruct = self.Datastruct(
            rots_=RotTransDatastruct(rots=rots, trans=transl))

        if return_type == 'vertices':
            final_motion = final_datastruct.vertices
        elif return_type in ['joints', 'smplh']:
            final_motion = final_datastruct.joints
        elif return_type == 'rots':
            return rots, transl
        elif return_type == 'rotsd':
            return motion
        elif return_type == 'rotsd+vertices':
            return motion, final_datastruct.vertices
        else:
            raise NotImplementedError

        return final_motion

    def motion_to_motion_forward(self, datastruct,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False,
                                 return_motion: Optional[str] = None):
        # Make sure it is on the good device
        datastruct.transforms = self.transforms
        # Encode the motion to the latent space
        if return_motion:
            datastruct_to_encode = datastruct
        else:
            datastruct_to_encode = datastruct.features

        if self.hparams.vae:
            distribution = self.motionencoder(datastruct_to_encode, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(datastruct_to_encode, lengths)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector[:, None], lengths)
        datastruct = self.Datastruct(features=features)
        if return_motion:
            return self.motion_from_datastruct(datastruct,
                                               return_type=return_motion)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution
    
    @torch.no_grad()
    def transform_batch_to_mixed_synthetic(self, batch):
        from sinc.tools.frank import combine_motions
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


    def allsplit_step(self, split: str, batch, batch_idx):
        # Prepare the generated motion features
        # input_motion_feats_0: [batch_size, max_time, feats]
        # gt_motion_feats = batch["datastruct"]
        # gt_lens = batch['length']
        # gt_texts = batch['text']
        # gpt_parts = batch['bp-gpt']
        # if self.hparams.synthetic:
        lens, motions_ds = self.transform_batch_to_mixed_synthetic(batch)
        del batch['datastruct_a']
        del batch['datastruct_b']
        del batch['datastruct']
        batch['datastruct'] = motions_ds.to(self.device)
        batch['length'] = lens
        
        gt_motion_feats = batch["datastruct"]
        gt_lens = batch['length']
        gt_texts = batch['text']
        gpt_parts = batch['bp-gpt']
        # batch.clear()
        
        bs = len(gt_lens)
        # text
        #  b, tuple of size 1,2,3,4...
        # [(a, b), (g,), (e, r, t)]
        
        if self.reduce_latents:
            datastruct_from_text,\
            latent_vectors_text,\
            distributions_from_text,\
            mask_texts = self.text_to_motion_forward(gt_texts, gt_lens, 
                                                     return_latent=True,
                                                     return_mask=True,
                                                     gpt_parts=gpt_parts)

        else:
            datastruct_from_text,\
            latent_vectors_text,\
            distributions_from_text,\
            mask_texts = self.text_to_motion_forward(gt_texts, gt_lens, 
                                                     return_latent=True,
                                                     return_mask=True)
        # Motion part
        if self.motion_branch:
            # Encode the first motion
            # Encode the motion/decode to a motion
            ret = self.motion_to_motion_forward(gt_motion_feats,
                                                gt_lens,
                                                return_latent=True)
            datastruct_from_motion, latent_from_motion, distribution_from_motion = ret

        # output_features_T_lst = [feats[:len0] for feats, len0 in zip(output_features_T, length)]
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            if self.separate_latents:
                distribution_ref_text_lst = []
                for ds_text in distributions_from_text:
                    mu_ref = torch.zeros_like(ds_text.loc)
                    scale_ref = torch.ones_like(ds_text.scale)
                    ds_ref = torch.distributions.Normal(mu_ref, scale_ref)
                    distribution_ref_text_lst.append(ds_ref)
                mu_ref_mot = torch.zeros_like(distribution_from_motion.loc)
                scale_ref_mot = torch.ones_like(distribution_from_motion.scale)
                distribution_ref_mot = torch.distributions.Normal(mu_ref_mot, scale_ref_mot)

            else:
                dist_T = distributions_from_text
                mu_ref = torch.zeros_like(dist_T.loc)
                scale_ref = torch.ones_like(dist_T.scale)
                distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None
        # may use them for GPT-based loss
        merged_annots_weights = None
        bp_list_ref = None
        bp_list_text = None
        bp_list_motion = None

        if self.separate_latents:
            # TODO fix this 2 latents
            total_loss, loss_dict = self.losses[split](ds_ref=gt_motion_feats,
                                                       ds_text=datastruct_from_text,
                                                       ds_motion=datastruct_from_motion,
                                                       multi_dis_text=distributions_from_text,
                                                       dis_motion=distribution_from_motion,
                                                       dis_ref_text=distribution_ref_text_lst,
                                                       dis_ref_mot=distribution_ref_mot,
                                                       multi_lat_text=latent_vectors_text,
                                                       lat_motion=latent_from_motion,
                                                       lengths=gt_lens,
                                                       texts_mask=mask_texts,
                                                       gpt_wts=merged_annots_weights,
                                                       bp_list_ref=bp_list_ref,
                                                       bp_list_text=bp_list_text,
                                                       bp_list_motion=bp_list_motion)
        else:
            total_loss, loss_dict = self.losses[split](ds_ref=gt_motion_feats,
                                                       ds_text=datastruct_from_text,
                                                       ds_motion=datastruct_from_motion,
                                                       dis_text=distributions_from_text,
                                                       dis_motion=distribution_from_motion,
                                                       dis_ref=distribution_ref,
                                                       dis_ref_mot=distribution_ref,
                                                       lat_text=latent_vectors_text,
                                                       lat_motion=latent_from_motion,
                                                       lengths=gt_lens,
                                                       gpt_wts=merged_annots_weights,
                                                       bp_list_ref=bp_list_ref,
                                                       bp_list_text=bp_list_text,
                                                       bp_list_motion=bp_list_motion)

        if split == 'val':
            self.metrics(gt_motion_feats.detach(),
                         datastruct_from_text.detach(),
                        #  datastruct_from_text.detach(),
                        #  gt_motion_feats.detach(), 
                         gt_lens)
        if batch_idx == 0:
            nvids = self.hparams.nvids_to_save
            if nvids is not None and nvids != 0:
                if split in self.store_examples:
                    # del self.store_examples[split]
                    self.store_examples[split].clear()

                lengths_for_viz = gt_lens[:nvids]
                keyids_for_viz = batch['keyid'][:nvids]
                def prepare_pos(x):
                    x = x.detach().joints[:nvids]
                    x = x.cpu().numpy()
                    return remove_padding(x, lengths_for_viz)
                def prepare_verts(x):
                    x = x.detach().vertices[:nvids]
                    x = x.cpu().to(dtype=torch.float32).numpy()
                    return remove_padding(x, lengths_for_viz)
                def prepare_features(x, lens):
                    x = x.detach().features[:nvids]
                    # x =  x.cpu()
                    x = remove_padding(x, lens)
                    x = collate_tensor_with_padding(x)
                    return x
                # ['transforms', '_joints2jfeats', 'features', 'joints_', 'jfeats_']
                #
                # ['transforms', '_rots2rfeats', '_rots2joints', '_joints2jfeats',
                # 'features', 'rots_', 'rfeats_', 'joints_', 'jfeats_']

                self.store_examples[split] = { "text": gt_texts[:nvids] }
                # if 'vertices_' in output_features_M.keys():
                #     # get SMPL features for viz
                #     self.store_examples[split].update({
                #         'ref': prepare_verts(input_motion_feats),
                #         'ref_features': motion_features.detach(),
                #         'keyids': keyids
                #     })
                # else:
                self.store_examples[split].update({
                    'ref': prepare_verts(gt_motion_feats),
                    'ref_features': prepare_features(gt_motion_feats, lengths_for_viz),
                    'keyids': keyids_for_viz,
                    'lengths': lengths_for_viz
                })

        # self.tracker[split].update(loss_dict)
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
        
        del gt_motion_feats, datastruct_from_text, datastruct_from_motion
        del distributions_from_text, distribution_from_motion
        del latent_vectors_text, latent_from_motion
        del batch
        torch.cuda.empty_cache()
        return total_loss, loss_dict
