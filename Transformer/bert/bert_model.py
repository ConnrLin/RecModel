
import torch.nn as nn
import torch.nn.functional as F
import torch
from .bert import BERT


class BERTModel(nn.Module):
    
    
    def __init__(self, max_len, n_layers, num_head, n_items, embed_size, dropout) -> None:
        
        super(BERTModel, self).__init__()

        self.bert = BERT(max_len, n_layers, num_head, n_items, embed_size, dropout)
        self.tran = nn.Linear(embed_size ,n_items+1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.bert(x)
        x = self.tran(x)
        x = self.softmax(x)
        return x