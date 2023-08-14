import torch
from torch.nn.functional import smooth_l1_loss


class Recons:
    def __call__(self, input_motion_feats_lst, output_features_lst):
        # for x,y in zip(input_motion_feats_lst, output_features_lst):
        #     print('----------------------')
        #     print(x.shape, y.shape)
        #     print(smooth_l1_loss(x, y, reduction="mean").shape)
        #     print('----------------------')
        recons = torch.stack([smooth_l1_loss(x.squeeze(), y.squeeze(), reduction="mean") for x,y in zip(input_motion_feats_lst,
                                                                                    output_features_lst)]).mean()
        return recons 

    def __repr__(self):
        return "Recons()"
