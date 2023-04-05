'''
Author: Felix
Date: 2023-04-03 21:30:10
LastEditors: Felix
LastEditTime: 2023-04-04 15:21:08
Description: Feed Forward Network
'''
import torch.nn as nn
import torch.nn.functional as F
from .gelu import GELU

class FFN(nn.Module):
    
    def __init__(self, d_model, d_hidden, dropout = 0.1):
        """ feed forward network, it contains 2 linear layers. 

        Args:
            d_model (int): dimension of the model
            d_hidden (int): dimension of hidden layers
            dropout (float, optional): dropout. Defaults to 0.1.
        """
        self.d = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear_layer1 = nn.Linear(d_model,d_hidden)
        self.linear_layer2 = nn.Linear(d_hidden,d_model)
        self.glue = GELU() 
    
    def forward(self,x):
        x = self.linear_layer1(x)
        x = self.glue(x)
        x = self.dropout(x)
        x = self.linear_layer2(x)
        return x
