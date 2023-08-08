from typing import Optional

import torch
from torch import Tensor, nn
from pathlib import Path
import os
import hydra
import glob
from sinc.utils.file_io import get_typename

class Rots2Rfeats(nn.Module):
    def __init__(self, path: Optional[str] = None,
                 normalization: bool = True,
                 eps: float = 1e-5,
                 **kwargs) -> None:
        if normalization and path is None:
            raise TypeError("You should provide a path if normalization is on.")

        super().__init__()
        self.normalization = normalization
        self.eps = eps
        if normalization:
            # workaround for cluster local/sync
            rel_p = path.split('/')
            # superhacky it is for the datatype ugly stuff change it, copy the main stuff to seperate_pairs dict
            if rel_p[-1] == 'separate_pairs':
                rel_p.remove('separate_pairs') 
            if '+' in rel_p[-1]:
                rel_p = get_typename(rel_p)
            
            # if '+' in rel_p[-1]:
            #     rel_p.remove(rel_p[-1])
            ########################################################
            rel_p = rel_p[rel_p.index('deps'):]
            rel_p = '/'.join(rel_p)
            path = hydra.utils.get_original_cwd() + '/' + rel_p

            mean_path = Path(path) / "rfeats_mean.pt"
            std_path = Path(path) / "rfeats_std.pt"

            self.register_buffer('mean', torch.load(mean_path))
            self.register_buffer('std', torch.load(std_path))

    def normalize(self, features: Tensor) -> Tensor:
        if self.normalization:
            features = (features - self.mean)/(self.std + self.eps)
        return features

    def unnormalize(self, features: Tensor) -> Tensor:

        # Debugging Block
        # if features.isnan().any():
        #     print("Features are buggy")

        #     torch.save(features, f'features_base_nan.pt')

        # if (features*(self.std+self.eps) + self.mean).isnan().any():
        #     torch.save(self.mean, f'mean_base_nan.pt')
        #     torch.save(self.std, f'std_base_nan.pt')

            
        #     torch.save(features, f'features_pt')
        # if ((features > 1000000).any() or (features < -1000000).any()):
        #     print("Features are buggy")
        #     torch.save(features, f'features_base_big.pt')
        # if (features*(self.std + self.eps) + self.mean > 1000000).any() \
        #         or ((features*(self.std + self.eps) + self.mean < -1000000).any()):
        #     torch.save(self.mean, f'mean_base_big.pt')
        #     torch.save(self.std, f'std_base_big.pt')
        if self.normalization:
            features = features * (self.std + self.eps) + self.mean

        return features
