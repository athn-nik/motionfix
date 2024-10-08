import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1,
                 max_len=5000, batch_first=False, negative=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
        self.negative = negative
        
        if negative:
            pe = torch.zeros(2*max_len, d_model)
            position = torch.arange(-max_len, max_len, dtype=torch.float).unsqueeze(1)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)            

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x, hist_frames=0):
        if not self.negative:
            center = 0
            assert hist_frames == 0
            first = 0
        else:
            center = self.max_len
            first = center-hist_frames
        if self.batch_first:
            last = first + x.shape[1]
            x = x + self.pe.permute(1, 0, 2)[:, first:last, :]
        else:
            last = first + x.shape[0]
            x = x + self.pe[first:last, :]
        return self.dropout(x)