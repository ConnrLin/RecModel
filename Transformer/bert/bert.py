'''
Author: Felix
Date: 2023-04-04 14:57:25
LastEditors: Felix
LastEditTime: 2023-04-04 15:49:05
Description: BERT4rec
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from .encoder import EncoderBlock
from .embedding import Embedding


class BERT(nn.Module):
    
    
    def __init__(self, max_len, n_layers, num_head, n_items, embed_size, dropout) -> None:
        """Initialize BERT model

        Args:
            max_len (int): the maximum length of item sequence, parts over the max_len will be depricated
            n_layers (int): the number of layers of transfromer
            num_head (int): the number of head of multi-head attention
            n_items (int): the number of unqiue items in the dataset
            embed_size (int): the dimension of embedded output
            dropout (float): dropout
        """
        super().__init__()

        self.embedding = Embedding(vocab_size=n_items,embed_size=embed_size,max_len=max_len,dropout=dropout)
        self.transformer_blocks = nn.ModuleList([EncoderBlock(embed_size,num_head,embed_size*4,dropout) for _ in range(n_layers)])
        
    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x

