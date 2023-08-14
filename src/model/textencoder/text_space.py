from locale import ABMON_10
from .distilbert import DistilbertEncoderBase
import torch

from typing import List, Union, Optional
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from src.model.utils import PositionalEncoding
from src.data.tools import lengths_to_mask


class TextSpace(DistilbertEncoderBase):
    def __init__(self, modelpath: str,
                 nfeats: int,
                 hist_frames: int = 1,
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 separate_latents: bool = False ,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)
        self.save_hyperparameters(logger=False)

        encoded_dim = self.text_encoded_dim
        self.separate_latents = separate_latents
        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, latent_dim))
        # TransformerVAE adapted from ACTOR
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        # separate 2 texts
        self.separation_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim,
                                                        dropout,
                                                        batch_first=True) # multi-gpu

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation,
                                                             batch_first=True) # multi-gpu

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

        input_feats = nfeats

    def forward(self, texts: List[str], mapping=None) -> Union[Tensor, Distribution]:

        # [sents, max_wds, 768],[sents, max_wds]
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)
        # [sents, max_wds, 256]
        text_emb = self.projection(text_encoded)
        try:
            if self.separate_latents:
                # [(a), (i, j, k), (g, f), ...]
                bs = len(texts)
                text_emb = self.sequence_pos_encoding(text_emb)
            else:
                # Joint text encoder
                # train with both texts so encode them raw
                # pack them back to the batch
                max_tokens = text_emb.shape[-2]
                from collections import Counter
                max_sim_texts = Counter(mapping).most_common(1)[0][1]
                # extract each embedding for each sentences
                bs = len(set(mapping))
                texts_batch = [[] for _ in range(bs)]
                masks_batch = [[] for _ in range(bs)]
                for i, index in enumerate(mapping):
                    texts_batch[index].append(text_emb[i])
                    masks_batch[index].append(mask[i])

                for i, latent_vector_batch in enumerate(texts_batch):
                    texts_batch[i] = torch.stack(latent_vector_batch)
                    masks_batch[i] = torch.stack(masks_batch[i])
                # [[text[i]_len, max_tokens, 256], [text[i+1]_len, max_tokens, 256], ...]
                # same for batch
                from src.data.tools.collate import collate_tensor_with_padding
                texts_batch = collate_tensor_with_padding(texts_batch)
                masks_batch = collate_tensor_with_padding(masks_batch)
                # [bs, max_texts, max_tokens, 256]
                bs = texts_batch.shape[0]
        except:
            import ipdb; ipdb.set_trace()
        # # nsents * 2
        # correct the mask to attend properly


        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]

        # x = x.permute(1, 0, 2)  # now it is [nwords, bs, latent_dim]

        # TEXT-MOTION part
        # VAE only for now
        if self.hparams.vae:

            if self.separate_latents:
                mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
                logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)
                xseq = torch.cat((mu_token[:, None], logvar_token[:, None], text_emb), 1)
                number_of_extra_tokens = 2

            else:
                mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
                logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)
                sep_token = torch.tile(self.separation_token, (bs,)).reshape(bs, -1)
                xseq = torch.cat((mu_token[:, None], logvar_token[:, None]), 1)
                mask_in = masks_batch[:, 0]
                if max_sim_texts > 1:
                    for i in range(max_sim_texts-1):
                        sep_mask = torch.ones((bs, 1), dtype=bool, device=text_emb.device)

                        mask_in = torch.cat((mask_in, masks_batch[:, i+1], sep_mask), 1)
                        xseq = torch.cat((xseq, texts_batch[:, i], sep_token[:, None]), 1)
                    xseq = torch.cat((xseq, texts_batch[:, i+1]), 1)
                else:
                    # []
                    xseq = torch.cat((xseq, texts_batch[:, 0]), 1)
                number_of_extra_tokens =  2 #+ max_sim_texts - 1
                mask = mask_in
            # should I change the augmented mask differently for separation token?
            token_mask = torch.ones((bs, number_of_extra_tokens), dtype=bool,
                                     device=text_emb.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            raise NotImplementedError

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        if self.hparams.vae:
            # if self.separate_latents:
            #     mu1, logvar1 = final[:, 0], final[:, 1]
            #     mu0, logvar0 = final[:, 2], final[:, 3]
            #     std0 = logvar0.exp().pow(0.5)
            #     std1 = logvar1.exp().pow(0.5)
            #     dist0 = torch.distributions.normal.Normal(mu0, std0)
            #     dist1 = torch.distributions.normal.Normal(mu1, std1)
            #     return dist0, dist1

            # else:
            mu, logvar = final[:, [0]], final[:, [1]]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py / could NaN cause error
            dist = torch.distributions.normal.Normal(mu, std)
            return dist
        else:
            return final[0]
