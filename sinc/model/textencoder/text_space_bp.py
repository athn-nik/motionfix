from locale import ABMON_10
from .distilbert import DistilbertEncoderBase
import torch

from typing import List, Union, Optional
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from sinc.model.utils import PositionalEncoding
from sinc.data.tools import lengths_to_mask


class TextSpaceBP(DistilbertEncoderBase):
    def __init__(self, modelpath: str,
                 nfeats: int,
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)
        self.save_hyperparameters(logger=False)

        encoded_dim = self.text_encoded_dim
        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, latent_dim))
        self.z_len = 6

        # TransformerVAE adapted from ACTOR
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(self.z_len, latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(self.z_len, latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(self.z_len, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim,
                                                        dropout) # multi-gpu

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation) # multi-gpu

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)
        

    def forward(self, texts: List[str], mapping=None) -> Union[Tensor, Distribution]:
        # text_encoded => [sents, max_wds, 768], 
        # mask => [sents, max_wds]
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)
        
        # [sents, max_wds, 256]
        text_emb = self.projection(text_encoded)
        
        # [(a), (i, j, k), (g, f), ...]       
        bs = text_emb.shape[0]
        text_emb = self.sequence_pos_encoding(text_emb)

        # # nsents * 2 
        # correct the mask to attend properly

  
        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
  
        text_emb = text_emb.permute(1, 0, 2)  # now it is [nwords, bs, latent_dim]
       
        # TEXT-MOTION part
        # VAE only for now
        if self.hparams.vae:

            mu_token = torch.tile(self.mu_token, (bs, 1, 1)).permute(1, 0, 2)
            logvar_token = torch.tile(self.logvar_token, (bs, 1, 1)).permute(1, 0, 2)
            xseq = torch.cat((mu_token, logvar_token, text_emb), 0)
 
            token_mask = torch.ones((bs, 2*self.z_len), dtype=bool,
                                     device=text_emb.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        else:
            raise NotImplementedError

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        if self.hparams.vae:
            mu, logvar = final[:self.z_len], final[self.z_len:2*self.z_len]
            # wds, bs
            mu = mu.permute(1, 0, 2)
            logvar = logvar.permute(1, 0, 2)

            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py / could NaN cause error
            dist = torch.distributions.normal.Normal(mu, std)
            return dist
        else:
            return final[0]
