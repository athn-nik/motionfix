import torch
from torch.nn.functional import smooth_l1_loss


class ReconsBP:
    def __call__(self, input_motion_feats_lst, output_features_lst):
        recons = [smooth_l1_loss(x.squeeze(), 
                                 y.squeeze(), 
                                 reduction='none') for x,y in zip(input_motion_feats_lst,
                                                                  output_features_lst)]

        recons = torch.stack([bpl.mean((1,2)) for bpl in recons], dim=1)
        return recons 

    def __repr__(self):
        return "ReconsBP()"


