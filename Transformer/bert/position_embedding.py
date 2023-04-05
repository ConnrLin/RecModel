'''
Author: Felix
Date: 2023-04-04 14:59:40
LastEditors: Felix
LastEditTime: 2023-04-04 15:31:36
Description: Positional Embedding
'''

import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, embed_size=512):
        """ PositionalEmbedding

        Args:
            max_len (int): maximum length of input sequence, parts over this length will be depricated
            embed_size (int): size of word embedded
        """
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, embed_size)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
