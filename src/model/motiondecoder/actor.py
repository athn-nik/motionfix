import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional
from torch import nn, Tensor

from src.model.utils import PositionalEncoding
from src.data.tools import lengths_to_mask
from einops import rearrange


class ActorAgnosticDecoder(pl.LightningModule):
    def __init__(self, nfeats: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        output_feats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout) # multi GPU

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation) # for multi GPU

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, mask: Tensor, mem_masks=None):
        latent_dim = z.shape[-1]
        bs, nframes = mask.shape
        nfeats = self.hparams.nfeats

        # z = z[:, None]  # sequence of 1 element for the memory 
        # separate latents
        # torch.cat((z0[:, None], z1[:, None]), 1)
        if len(z.shape) > 3: 
            z = rearrange(z, "bs nz z_len latent_dim -> (nz z_len) bs latent_dim")
        else:
            z = rearrange(z, "bs z_len latent_dim -> z_len bs latent_dim")
        
        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        if mem_masks is not None:
            mem_masks = ~mem_masks
        output = self.seqTransDecoder(tgt=time_queries, memory=z,
                                      tgt_key_padding_mask=~mask,
                                      memory_key_padding_mask=mem_masks)
        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats
