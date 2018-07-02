import torch
import torch.nn as nn
import data

class Model(nn.Module):
    def __init__(self, word_vocab_size, word_dim, char_vocab_size, char_dim):
        self.word_embedding = nn.Embedding(word_vocab_size, word_dim, padding_idx=data.NULL_ID)
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=data.NULL_ID)


    def forward(self, c, q, ch, qh):
        ch_emb = self.char_embedding(ch)
        qh_emb = self.char_embedding(qh)