import torch
import torch.nn as nn
import data
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class EncoderBase(nn.Module):
    def forward(self, src, lengths=None, encoder_state=None):
        raise NotImplementedError


class RNNEncoder(EncoderBase):
    def __init__(self, input_size, num_layers, hidden_size, bidirectional, dropout, type='lstm', use_bridge=False, batch_first=True):
        super(RNNEncoder, self).__init__()
        hidden_size = hidden_size//2 if bidirectional else hidden_size
        rnn = {
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }
        self.type = type
        self.rnn = rnn[type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first)


    def sort_batch(self, data, lengths):
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        sorted_data = data[sorted_idx]
        return sorted_data, sorted_lengths


    def forward(self, src, lengths, encoder_state=None, ordered=False):
        if lengths is not None:
            lengths = lengths + lengths.eq(0).long()
            if ordered:
                packed_emb = pack(src, lengths, batch_first=self.rnn.batch_first)
                memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
                memory_bank = unpack(memory_bank, batch_first=self.rnn.batch_first)[0]
            else:
                sorted_lengths, perm_idx = lengths.sort(descending=True)
                sorted_src = src[perm_idx]
                if encoder_state is not None:
                    encoder_state = encoder_state[perm_idx]
                packed_emb = pack(sorted_src, sorted_lengths, batch_first=self.rnn.batch_first)
                memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
                memory_bank = unpack(memory_bank, batch_first=self.rnn.batch_first)[0]
                _, odx = perm_idx.sort()
                memory_bank = memory_bank[odx]
                encoder_final = [s[:, odx, :] for s in encoder_final] if self.type == 'lstm' else encoder_final[:, odx, :]
        else:
            memory_bank, encoder_final = self.rnn(src, encoder_state)
        return memory_bank, encoder_final


if __name__ == '__main__':
    embeddings = [
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [-0.1, 0.05, -0.2],
        [0.2, -0.1, 0.1],
        [-0.12, -0.2, 0.15]
    ]
    embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings))
    embeddings.padding_idx = 0
    seq0 = [1, 2, 3]
    seq1 = [4, 2, 0]
    seq2 = [3, 0, 0]
    s0 = torch.tensor([seq0, seq1, seq2])
    l0 = torch.tensor([3, 2, 1])
    s1 = torch.tensor([seq1, seq2, seq0])
    l1 = torch.tensor([2, 1, 3])

    for type in ['lstm', 'gru']:
        rnn = RNNEncoder(3, 1, 4, True, 0.0, type=type)
        m0, t0 = rnn(embeddings(s0), l0, ordered=True)
        m1, t1 = rnn(embeddings(s1), l1, ordered=False)
        d0 = m1[2] - m0[0]
        d1 = m1[0] - m0[1]
        d2 = m1[1] - m0[2]
        assert d0.abs().sum().tolist() == 0
        assert d1.abs().sum().tolist() == 0
        assert d2.abs().sum().tolist() == 0
        if not isinstance(t0, tuple):
            t0 = (t0,)
            t1 = (t1,)
        d0 = t1[0][:,2,:] - t0[0][:,0,:]
        d1 = t1[-1][:,0] - t0[-1][:,1]
        d2 = t1[0][:,1] - t0[0][:,2,:]
        assert d0.abs().sum().tolist() == 0
        assert d1.abs().sum().tolist() == 0
        assert d2.abs().sum().tolist() == 0