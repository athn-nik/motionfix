import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from sinc.model.utils import PositionalEncoding
from sinc.data.tools import lengths_to_mask_njoints


class ActorAgnosticEncoder(pl.LightningModule):
    def __init__(self, nfeats: int, njoints: int,
                 vae: bool,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(njoints, latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(njoints, latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(njoints, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device
        bs, nframes, njoints, nfeats = features.shape

        joints_mask = lengths_to_mask_njoints(lengths, njoints, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # merge/flatten the joints and time
        x = x.flatten(1, 2) # [bs, time*joints, latent_dim]

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
            token_mask = torch.ones((bs, 2*njoints), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, joints_mask), 1)
        else:
            raise NotImplementedError
 
        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)

        # final [seqlen=time*njoints, bs, latent_dim]
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        if self.hparams.vae:
            mu, logvar = final[:njoints], final[njoints:2*njoints]
            mu = mu.permute(1, 0, 2)
            logvar = logvar.permute(1, 0, 2)

            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            dist = torch.distributions.Normal(mu, std)
            return dist
        else:
            return final[0]

