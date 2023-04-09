'''
Author: Felix
Date: 2023-04-03 21:26:50
LastEditors: Felix
LastEditTime: 2023-04-04 15:28:11
Description: Encoder of transformer
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from .feed_forward_network import FFN
from .norm import Norm
from ..attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    
    def __init__(self, d_model, num_head ,d_hidden, dropout = 0.1):
      """ Encoder block of transformer, containing 1 multi-head attention layer and 1 feed forward network

      Args:
          d_model (int): dimension of the model
          num_head (int): number of head of multi-head attention 
          d_hidden (int): dimensin of hidden layer of feed forward network
          dropout (float, optional): dropout. Defaults to 0.1.
      """
      super(EncoderBlock, self).__init__()

      self.attention = MultiHeadAttention(num_head,d_model,dropout)
      self.norm1 = Norm(d_model)
      self.norm2 = Norm(d_model)
      self.dropout1 = nn.Dropout(dropout)
      self.dropout2 = nn.Dropout(dropout)
      self.FFN = FFN(d_model,d_hidden,dropout)
    

    def forward(self,x,mask=None):
       x1 = self.norm1(x)
       x = x + self.dropout1(self.attention(x1,x1,x1,mask))
       x2 = self.norm2(x)
       x = x + self.dropout2(x2)
       self.FFN(x2)
       return x
       
