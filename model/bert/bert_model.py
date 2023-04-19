'''
Author: Felix
Date: 2023-04-11 14:52:07
LastEditors: Felix
LastEditTime: 2023-04-11 19:44:12
Description: Please enter description
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from .bert import BERT


class BERTModel(nn.Module):

    def __init__(self, max_len, n_layers, num_head, n_items, embed_size, dropout) -> None:

        super(BERTModel, self).__init__()

        self.bert = BERT(max_len, n_layers, num_head,
                         n_items, embed_size, dropout)
        self.tran = nn.Linear(embed_size, n_items+1)

    def forward(self, x):
        x = self.bert(x)
        x = self.tran(x)
        return x
