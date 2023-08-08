from typing import List, Optional, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions.distribution import Distribution
from torch.nn import ModuleDict

from sinc.model.base import BaseModel
from sinc.model.metrics import ComputeMetricsSpace
from sinc.model.utils.tools import remove_padding
from sinc.model.losses.utils import LossTracker

class SPACESpatial(BaseModel):
    def __init__(self, textencoder: DictConfig,
                 motionencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                 transforms: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 motion_branch: bool,
                 separate_latents: Optional[bool] = False,
                 nvids_to_save: Optional[int] = None,
                 teacher_forcing: Optional[bool] = False,
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

        self.metrics = ComputeMetricsSpace()

        self.nvids_to_save = nvids_to_save
        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = 1.0
        self.teacher_forcing = teacher_forcing
        self.motion_branch = motion_branch
        self.separate_latents = separate_latents
        # Keep track of the losses
        self._losses = ModuleDict({split: instantiate(losses, vae=vae,
                                                      separate_latents=separate_latents,
                                                      _recursive_=False)
                                   for split in ["losses_train", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "val"]}
        
        self._tracker = ModuleDict({"tracker_" + split: LossTracker(self.losses[split].losses)
                                            for split in ["train", "val"]})
        self.tracker = {key: self._tracker["tracker_" + key] for key in ["train", "val"]}
        self.__post_init__()

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int], *,
                               return_latent: bool = False):
        bs = len(text_sentences)
        ntexts_batch = [len(lst) for lst in text_sentences]

        max_number_of_z_texts = max(ntexts_batch)
        
        # one big forward path
        # merge all the sep texts
        # keep the indices
        text_and_idx = [(text, y) for texts, y in zip(text_sentences, range(bs))
                        for text in texts]

        texts, indices = zip(*text_and_idx)
        texts = list(texts)

        # Encode the text to the latent space
        if self.hparams.vae:
            distributions, att = self.textencoder(texts)
            latent_vectors = self.sample_from_distribution(distributions)
        else:
            distribution = None
            latent_vectors = self.textencoder(texts)
        
        
        # Similar for all the architectures    
        # [(texta_1, texta_2, texta_3), (textb_1, textb_2), (textc_1,)]
        # flatten => [texta_1, texta_2, texta_3, textb_1, text_b2, textc_1]
        # distilbert => [tokensa_1, tokensa_2, tokensa_3, tokensb_1, tokensb_2, tokensc_1] + masks
        
        # FOR FULLY INDEPENDANT/SEPARATE
        # transformer encoder + key_padding_mask = masks => [distriba_1, distriba_2, distriba_3, distribb_1, distribb_2, distribc_1]
        # reorganize => [(distriba_1, distriba_2, distriba_3), (distribb_1, distribb_2), (distribc_1,)]
        # padding => [(distriba_1, distriba_2, distriba_3), (distribb_1, distribb_2, 00000), (distribc_1, 00000, 00000)]
        # same for latent vector:
        # padding => [(latenta_1, latenta_2, latenta_3), (latentb_1, latentb_2, 00000), (latentc_1, 00000, 00000)]          
        # + number_of_actions mask => [(True, True, True), (True, True, False), (True, False, False)]
        # this mask is the mem_key_padding_mask for the transformer decoder
        # TEMOS losses for reconstruction
        # For latent sinc.
        ## 1. KL divergence between the motion distribution and the texts distributions
        ### => can be done in parralel
        ### Z_t (BS, max_number_of_z_texts), z_m (BS)
        ### compute_loss: create a function named 
        # z_m 
        
        
        # FOR JOINT MODEL
        # reorganize => [(tokensa_1, tokensa_2, tokensa_3), (tokensb_1, tokensb_2), (tokensc_1,)] + reorganized_masks [(TTT, TFF, TTF), (TTF, TTF), (TTT,))])
        # padding =>  [(tokensa_1, tokensa_2, tokensa_3), (tokensb_1, tokensb_2, 00000), (tokensc_1, 00000, 00000)] + [(TTT, TFF, TTF), (TTF, TTF, FFF), (TTT, FFF, FFF))])
        # => create a tensor => BS x MAX_LEN x NUMBER_TOKENS_MAX
        # joint the tokens => BS x (NUMBER_TOKENS_MAX x MAX_LEN + MAX_LEN-1) (need to create the good mask) => [[TTT T TFF T TTF), (TTF T TTF F FFF), (TTT F FFF F FFF))])
        ## transformer encoder => BS latent vectors (1 latent vector per list of text, for example: (texta_1, texta_2, texta_3) => latent_a, (textb_1, textb_2) => latent_b etc)
        # LIKE TEMOS
        
        

        # extract each embedding for each sentences
        latent_vectors_batch = [[] for _ in range(bs)]
        for i, index in enumerate(indices):
            latent_vectors_batch[index].append(latent_vectors[i])

        for i, latent_vector_batch in enumerate(latent_vectors_batch):
            latent_vectors_batch[i] = torch.stack(latent_vector_batch)

        # extract each distribution for each sentences
        distributions_batch = [[] for _ in range(bs)]
        for i, index in enumerate(indices):
            distributions_batch[index].append(torch.distributions.Normal(distributions.loc[i],
                                                                         distributions.scale[i])
                                              )

        from sinc.data.tools.collate import collate_tensor_with_padding
        latent_vectors_padded = collate_tensor_with_padding(latent_vectors_batch)

        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vectors_padded, lengths, ntexts_batch)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vectors_batch, distributions_batch, att

    def motion_to_motion_forward(self, datastruct,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False
                                 ):
        # Make sure it is on the good device
        datastruct.transforms = self.transforms
        # Encode the motion to the latent space
        if self.hparams.vae:
            distribution = self.motionencoder(datastruct.features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(datastruct.features, lengths)

        latent_vector = latent_vector[:, None]
        # Decode the latent vector to a motion
        features = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution

    
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


    def encode_data(self, data: Union[List[str], Tensor],
                    *, return_latent: bool = False, inference=False):
        # Use the text branch
        # and encode the text to the latent space
        if isinstance(data, list):
            distribution = self.textencoder(data)
        else:
            # it is a motion and check for inference
            if inference:
                distribution = self.motionencoder(data)
            else:
                distribution = self.motionencoder(data.features)

        if self.hparams.vae:
            latent_vector = self.sample_from_distribution(distribution)
            if not return_latent:
                return distribution
            return latent_vector, distribution
        else:           
            distribution = None
            latent_vector = distribution
            if not return_latent:
                return distribution
            return latent_vector, distribution


    def allsplit_step(self, split: str, batch, batch_idx):
        # Prepare the generated motion features
        length = batch["length"]
        # Encode the text/decode to a motion
        ret = self.text_to_motion_forward(batch["text"],
                                          batch["length"],
                                          return_latent=True)
        datastruct_from_text, latent_from_text, distribution_from_text, att_token = ret


        # GT data
        datastruct_ref = batch["datastruct"]

        # Motion part
        if self.motion_branch:
            # Encode the first motion
            # Encode the motion/decode to a motion
            ret = self.motion_to_motion_forward(batch["datastruct"],
                                                batch["length"],
                                                return_latent=True)
            datastruct_from_motion, latent_from_motion, distribution_from_motion = ret
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            dist_T = distribution_from_text
            mu_ref = torch.zeros_like(dist_T[0][0].loc)
            scale_ref = torch.ones_like(dist_T[0][0].scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None
        datastruct_gt = batch['datastruct']
        latent_from_text = torch.stack((latent_from_text))

        dis_from_text1, dis_from_text2 = map(list, zip(*distribution_from_text))

        dis_T1_loc = torch.stack([dis_T.loc for dis_T in dis_from_text1])
        dis_T2_loc = torch.stack([dis_T.loc for dis_T in dis_from_text2])

        dis_T1_scale = torch.stack([dis_T.scale for dis_T in dis_from_text1])
        dis_T2_scale = torch.stack([dis_T.scale for dis_T in dis_from_text2])

        dis_from_text1 = torch.distributions.Normal(dis_T1_loc, dis_T1_scale)
        dis_from_text2 = torch.distributions.Normal(dis_T2_loc, dis_T2_scale)

        total_loss, loss_dict = self.losses[split](ds_ref=datastruct_gt,
                                                   ds_text=datastruct_from_text,
                                                   ds_motion=datastruct_from_motion,
                                                   dis_text=dis_from_text1,
                                                   dis_text1=dis_from_text2,
                                                   dis_motion=distribution_from_motion,
                                                   dis_ref=distribution_ref,
                                                   lat_text=latent_from_text[:, 0],
                                                   lat_text1=latent_from_text[:, 1],
                                                   lat_motion=latent_from_motion,
                                                   lengths=length)
 
        if split == 'val':
            self.metrics(datastruct_gt.detach().joints,
                         datastruct_from_text.detach().joints,
                         length)
            
            
        if batch_idx == 0:
            nvids = self.hparams.nvids_to_save
            if nvids is not None and nvids != 0:
                del self.store_examples[split]
                lengths = batch['length'][:nvids]
                keyids = batch['keyid'][:nvids]
                motion_features = batch['datastruct']
                def prepare_pos(x):
                    x = x.detach().joints[:nvids]
                    x = x.cpu().numpy()
                    return remove_padding(x, lengths)
                def prepare_verts(x):
                    x = x.detach().vertices[:nvids]
                    x = x.cpu().numpy()
                    return remove_padding(x, lengths)

                self.store_examples[split] = { "text": batch["text"][:nvids] }
                self.store_examples[split].update({
                    'ref': prepare_pos(datastruct_gt),
                    'ref_features': motion_features.detach(),
                    'keyids': keyids
                })

        self.tracker[split].update(loss_dict)
        return total_loss
