'''
Author: Felix
Date: 2023-04-04 14:45:04
LastEditors: Felix
LastEditTime: 2023-04-04 14:45:04
Description: Glue activation 
'''

import torch.nn as nn
import torch
import math


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
