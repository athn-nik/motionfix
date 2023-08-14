import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from src.model.utils import PositionalEncoding
from src.data.tools import lengths_to_mask


class ActorAgnosticEncoder(pl.LightningModule):
    def __init__(self, nfeats: int, vae: bool,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        input_feats = nfeats
        self.skel_embedding = nn.Linear(input_feats, latent_dim)
        # self.layer_norm = nn.LayerNorm(nfeats)
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout, batch_first=True) # multi-GPU

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation,
                                                             batch_first=True) # multi-gpu

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        # x = self.layer_norm(x)
        x = self.skel_embedding(x)
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        # x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]
        # Each batch has its own set of tokens
        if self.hparams.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token[:, None], logvar_token[:, None], x), 1)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[:, None], x), 1)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # i = 0
        # while True:
        #     sample = normal.sample(x.shape)
        #     assert sample.isfinite().all(), f'{i=}, {sample=}'  # asserts after e.g. 1000 or 40000 iterations if SAMPLE_GPU == True
        #     i += 1
        if self.hparams.vae:
            mu, logvar = final[:, [0]], final[:, [1]]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            try:
                dist = torch.distributions.Normal(mu, std)
                # torch.save(mu, f'mean_tensor.pt')
                # torch.save(std, f'std_tensor.pt')

            except:
                torch.save(mu, 'mean_tensor.pt')
                torch.save(std, 'std_tensor.pt')
                torch.save(xseq, 'xseq_tensor.pt')
                torch.save(final, 'final_tensor.pt')
                torch.save(lengths, 'lengths_tensor.pt')
                import ipdb; ipdb.set_trace()

            return dist
        else:
            return final[0]
