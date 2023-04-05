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
    param {*} K:Key ,shape:[batch_size,seq_len,d_model]
    param {*} Q:Query shape:[batch_size,seq_len,d_model]
    param {*} V:Value shape:[batch_size,seq_len,d_model]
    param {*} mask
    param {*} dropout
    return {*} Softmax([K*Q.T/(d^0.5)])*V
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

        