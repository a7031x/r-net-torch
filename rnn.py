import torch
import torch.nn as nn
import data
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class EncoderBase(nn.Module):
    def forward(self, src, lengths=None, encoder_state=None):
        raise NotImplementedError


class RNNEncoder(EncoderBase):
    def __init__(self, embeddings, num_layers, hidden_size, bidirectional, dropout, type='lstm', use_bridge=False, batch_first=True):
        super(RNNEncoder, self).__init__()
        self.embeddings = embeddings
        self.vocab_size, self.embedding_dim = embeddings.weight.shape
        hidden_size = hidden_size//2 if bidirectional else hidden_size
        rnn = {
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }
        self.rnn = rnn[type](
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first)


    def forward(self, src, lengths, encoder_state=None):
        emb = self.embeddings(src)
        packed_emb = pack(emb, lengths, batch_first=self.rnn.batch_first)
        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
        memory_bank = unpack(memory_bank, batch_first=self.rnn.batch_first)[0]
        return encoder_final, memory_bank