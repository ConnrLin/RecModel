from .base import BaseModel
from .bert_modules import BERT

import torch.nn as nn


class BERTModel(nn.Module):
    def __init__(self, bert_max_len, num_items, bert_num_blocks, bert_num_heads, bert_hidden_units, bert_dropout):
        super().__init__()
        self.bert = BERT(bert_max_len, num_items, bert_num_blocks,
                         bert_num_heads, bert_hidden_units, bert_dropout)
        self.out = nn.Linear(self.bert.hidden, num_items + 1)

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)
