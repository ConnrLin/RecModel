'''
Author: Felix
Date: 2023-04-03 21:47:28
LastEditors: Felix
LastEditTime: 2023-04-04 15:20:51
Description: Layer Norm
'''
from torch import nn
import torch

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        """ layer normalization

        Args:
            d_model (int): dimension of the model
            eps (float, optional): eps. Defaults to 1e-6.
        """
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
