'''
Author: Felix
Date: 2023-04-03 20:54:04
LastEditors: Felix
LastEditTime: 2023-04-04 14:30:20
Description: MultiHeadAttention
'''
import torch.nn as nn
from .single_head_attention import SingleHeadAttention




class MultiHeadAttention(nn.Module):
    
    
    '''
    Author: Felix
    description: Intitialize the multi head attention layer
    param {*} num_head : the number of head (attention head)
    param {*} d : dimension of the output
    param {*} dropout : paramater for dropout layer
    '''    
    def __init__(self,num_head,d,dropout=0.1):
    
        
        super().__init__()

        # check if d is an integer multiple of num_head
        assert d%num_head == 0

        # the dimension of each head
        self.d_h = d//num_head
        self.num_head = num_head
        self.d = d

        # Wk,Wq,Wv
        self.linear_layers = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.output_layer = nn.Linear(d,d)
        self.attention = SingleHeadAttention()
        self.dropout = None
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)


    '''
    Author: Felix
    description: Calculate mulit-head attention
    param {*} K shape:[batch_size,seq_len,d_model]
    param {*} Q shape:[batch_size,seq_len,d_model]
    param {*} V shape:[batch_size,seq_len,d_model]
    param {*} masked
    return {*}
    '''    
    def forward(self,K,Q,V,mask=None):
        
        batch_size = K.size(0)

        # apply linear porjection
        k, q, v = [l(x).view(batch_size,-1,self.num_head,self.d_h).transpose(1, 2) for l,x in zip(self.linear_layers,(K,Q,V))]
        
        # calculate attention
        atten = self.attention(k, q, v, mask=mask, dropout=self.dropout)
        # concat multi head attention to shape [batch_size,seq_len,d_model]
        atten = atten.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_h)
        
        return self.output_layer(atten)


            
