import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from src.model.utils import PositionalEncoding
from src.data.tools import lengths_to_mask_njoints


class ActorAgnosticEncoder(pl.LightningModule):
    def __init__(self, nfeats: int, njoints: int,
                 vae: bool,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", z_len=6, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)
        #                                 # global orientation + z translation + x,y projected velocities
        self.bp_embedding_lst = nn.ModuleList([nn.Linear(9, latent_dim)] +
                                          # main body
                                          [nn.Linear(5*6, latent_dim)] +
                                          # left/right arms left/right foot
                                          [nn.Linear(4*6, latent_dim) for _ in range(4)])
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(z_len, latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(z_len, latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(z_len, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)
        self.z_len = z_len

    def forward(self, features_lst: List[Tensor], lengths: Optional[List[int]] = None) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features_lst]
        # extract body part number
        njoints = len(features_lst)
        part_feats = torch.stack([linear_part(input_part) for input_part, linear_part in zip(features_lst,
                                                                                             self.bp_embedding_lst)],
                                 dim=-2)

        device = features_lst[0].device
        bs, nframes, nfeats = features_lst[0].shape
        lat_len = self.z_len

        joints_mask = lengths_to_mask_njoints(lengths, 6, device)

        # merge/flatten the joints and time
        x = part_feats.flatten(1, 2) # [bs, time*joints, latent_dim]

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [time*joints, bs, latent_dim]

        # Each batch has its own set of tokens
        if self.hparams.vae:
            mu_token = torch.tile(self.mu_token, (bs, 1, 1)).permute(1, 0, 2)
            logvar_token = torch.tile(self.logvar_token, (bs, 1, 1)).permute(1, 0, 2)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token, logvar_token, x), 0)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2*lat_len), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, joints_mask), 1)
        else:
            raise NotImplementedError
 
        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)

        # final [seqlen=time*njoints, bs, latent_dim]
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        if self.hparams.vae:
            mu, logvar = final[:lat_len], final[lat_len:2*lat_len]
            mu = mu.permute(1, 0, 2)
            logvar = logvar.permute(1, 0, 2)

            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return final[0]

