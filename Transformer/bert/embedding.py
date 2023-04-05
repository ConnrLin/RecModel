'''
Author: Felix
Date: 2023-04-04 15:32:48
LastEditors: Felix
LastEditTime: 2023-04-04 15:41:03
Description: BERT embedding
'''
from torch import nn
from .position_embedding import PositionalEmbedding
from .token_embedding import TokenEmbedding

class Embedding(nn.Module):
    
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1) -> None:
      """ Embedding layer of BERT. Embedding(x) = Embedding_token(x)+Embedding_position(x)

      Args:
          vocab_size (int): the size of vocabulary of corpus
          embed_size (int): size of vertored vocabulary
          max_len (int): maximum length of input sequence
          dropout (float, optional): dropout. Defaults to 0.1.
      """
      super().__init__()
      self.token_embedding = TokenEmbedding(vocab_size,embed_size)
      self.position_embedding = PositionalEmbedding(max_len,embed_size)
      self.dropout = nn.Dropout(dropout)

    def forward(self,x):
       x = self.token_embedding(x)+self.position_embedding(x)
       return self.dropout(x)