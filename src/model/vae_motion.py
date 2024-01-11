##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class TIRG(pl.LightningModule):
    """
    The TIRG model.
    Implementation derived (except for BaseModel-inherence) from
    https://github.com/google/tirg (downloaded on July 23th 2020).
    The method is described in Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia
    Li, Li Fei-Fei, James Hays. "Composing Text and Image for Image Retrieval -
    An Empirical Odyssey" CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, input_dim=[512, 512], output_dim=512, out_l2_normalize=False):
        super(TIRG, self).__init__()

        self.input_dim = sum(input_dim)
        self.output_dim = output_dim

        # --- modules
        self.a = nn.Parameter(torch.tensor([1.0, 1.0])) # changed the second coeff from 10.0 to 1.0
        self.gated_feature_composer = nn.Sequential(
                ConCatModule(), nn.BatchNorm1d(self.input_dim), nn.ReLU(),
                nn.Linear(self.input_dim, self.output_dim))
        self.res_info_composer = nn.Sequential(
                ConCatModule(), nn.BatchNorm1d(self.input_dim), nn.ReLU(),
                nn.Linear(self.input_dim, self.input_dim), nn.ReLU(),
                nn.Linear(self.input_dim, self.output_dim))

        if out_l2_normalize:
            self.output_layer = L2Norm() # added to the official TIRG code
        else:
            self.output_layer = nn.Sequential()

    def query_compositional_embedding(self, main_features, modifying_features):
        f1 = self.gated_feature_composer((main_features, modifying_features))
        f2 = self.res_info_composer((main_features, modifying_features))
        f = torch.sigmoid(f1) * main_features * self.a[0] + f2 * self.a[1]
        f = self.output_layer(f)
        return f

class ConCatModule(pl.LightningModule):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return x

class NormalDistDecoder(pl.LightningModule):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))


class CorrectionModule(pl.LightningModule):
    """
    Given a pose A and the modifier m, compute an embedding representing the
    result from the modification of A by m.
    """

    def __init__(self, latentD, text_latentD, mode="tirg"):
        super(CorrectionModule, self).__init__()

        self.mode = mode

        if self.mode == "tirg":
            self.tirg = TIRG(input_dim=[latentD, text_latentD],
                             output_dim=latentD,
                             out_l2_normalize=False)
            self.forward = self.tirg.query_compositional_embedding
        elif self.mode == "concat-mlp":
            self.sequential = nn.Sequential(
                ConCatModule(),
                nn.BatchNorm1d(latentD+text_latentD),
                nn.ReLU(),
                nn.Linear(latentD+text_latentD, 2 * latentD),
                nn.ReLU(),
                nn.Linear(2 * latentD, latentD)
            )
            self.forward = self.process_concat_sequential
        else:
            print(f"Name for the mode of the correction module is unknown (provided {mode}).")
            raise NotImplementedError

    def process_concat_sequential(self, pose_embeddings, modifier_embeddings):
        return self.sequential((pose_embeddings, modifier_embeddings))