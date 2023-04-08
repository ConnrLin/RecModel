'''
Author: Felix
Date: 2023-04-03 20:27:44
LastEditors: Felix
LastEditTime: 2023-04-05 16:30:46
Description: Please enter description
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

"""
Single head attention
"""
class SingleHeadAttention(nn.Module):
    
    '''
    Author: Felix
    description: Calculate attention for given K,Q,V
    param {torch.Tensor} K:Key ,shape:[batch_size,seq_len,d_model]
    param {torch.Tensor} Q:Query shape:[batch_size,seq_len,d_model]
    param {torch.Tensor} V:Value shape:[batch_size,seq_len,d_model]
    param {torch.Tensor} mask
    param {float} dropout
    return {torch.Tensor} Softmax([K*Q.T/(d^0.5)])*V
    '''    
    def forward(K, Q, V,mask=None, dropout=None):
        score = torch.matmul(Q,K.transpose(-2, -1)) / math.sqrt(K.size[-1])
        if mask:
            # if masked, set masked items to a very small value
            score = score.masked_fill(mask,-1e9)
        score = F.softmax(score,dim = -1)
        if dropout:
            score = dropout(score)
        attention = torch.matmul(score,V)
        return attention

        