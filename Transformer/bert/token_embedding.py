'''
Author: Felix
Date: 2023-04-04 14:58:21
LastEditors: Felix
LastEditTime: 2023-04-04 15:39:42
Description: Token embedding for bert
'''

import torch.nn as nn
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        """ Token embedding of bert

        Args:
            vocab_size (int): the size of vocabulary of corpus
            embed_size (int, optional): the size of embedding output. Defaults to 512.
        """
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, padding_idx=0)
