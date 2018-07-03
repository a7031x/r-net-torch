import torch
import torch.nn as nn
import func
import data
import rnn
import utils
import os

class Model(nn.Module):
    def without_embedding(self, word_vocab_size, word_dim, char_vocab_size, char_dim):
        self.word_embedding = nn.Embedding(word_vocab_size, word_dim, padding_idx=data.NULL_ID)
        self.char_embedding = nn.Embedding(char_vocab_size, char_dim, padding_idx=data.NULL_ID)


    def with_embedding(self, word_mat, char_mat):
        self.word_embedding = nn.Embedding.from_pretrained(word_mat, freeze=True)
        self.char_embedding = nn.Embedding.from_pretrained(char_mat, freeze=False)
        self.char_embedding.padding_idx = data.NULL_ID


    def initialize(self, char_hidden_size, dropout):
        '''
        char_hidden_size: default 200
        '''
        self.char_rnn = rnn.RNNEncoder(
            input_size=self.char_embedding.weight.shape[1],
            num_layers=1,
            hidden_size=char_hidden_size,
            bidirectional=True,
            type='gru',
            dropout=dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, c, q, ch, qh):
        n, pl, _ = ch.shape
        ql = qh.shape[1]
        ch = ch.view(n*pl, -1)
        qh = qh.view(n*ql, -1)
        ch_len = (ch != data.NULL_ID).sum(-1)
        qh_len = (qh != data.NULL_ID).sum(-1)

        ch_emb = self.char_embedding(ch)#[n*pl, cl, dc]
        qh_emb = self.char_embedding(qh)#[n*ql, cl, dc]
        ch_emb = self.dropout(ch_emb)
        qh_emb = self.dropout(qh_emb)

        _, state = self.char_rnn(ch_emb, ch_len)#[num_layers,n*pl,dc]
        ch_emb = torch.cat([state[0], state[1]], -1).view(n, pl, -1)
        _, state = self.char_rnn(qh_emb, qh_len)
        qh_emb = torch.cat([state[0], state[1]], -1).view(n, ql, -1)

        c_emb = self.word_embedding(c)
        q_emb = self.word_embedding(q)
        c_emb = torch.cat([c_emb, ch_emb], -1)#[n, pl, dw+dc=500]
        q_emb = torch.cat([q_emb, qh_emb], -1)#[n, ql, dw+dc=500]


def build_train_model(opt, dataset=None):
    dataset = dataset or data.Dataset(opt)
    model = build_model(opt, dataset)
    feeder = data.TrainFeeder(dataset)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate)
    feeder.prepare('train')
    return model, optimizer, feeder


def build_model(opt, dataset=None):
    dataset = dataset or data.Dataset(opt)
    model = Model()
    model.with_embedding(func.tensor(dataset.word_emb), func.tensor(dataset.char_emb))
    model.initialize(opt.char_hidden_size, opt.dropout)
    if func.gpu_available():
        model = model.cuda()
    return model


def make_loss_compute():
    criterion = torch.nn.NLLLoss(size_average=False)
    if func.gpu_available():
        criterion = criterion.cuda()
    return criterion


def load_or_create_models(opt, train):
    if os.path.isfile(opt.ckpt_path):
        ckpt = torch.load(opt.ckpt_path, map_location=lambda storage, location: storage)
        model_options = ckpt['model_options']
        for k, v in model_options.items():
            setattr(opt, k, v)
    else:
        ckpt = None
    if train:
        model, optimizer, feeder = build_train_model(opt)
    else:
        model = build_model(opt)
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        if train:
            optimizer.load_state_dict(ckpt['optimizer'])
            feeder.load_state(ckpt['feeder'])
    if train:
        return model, optimizer, feeder, ckpt
    else:
        return model, ckpt


def restore(opt, model, optimizer, feeder):
    ckpt = torch.load(opt.ckpt_path, map_location=lambda storage, location: storage)
    if model is not None:
        model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if feeder is not None:
        feeder.load_state(ckpt['feeder'])


def save_models(opt, model, optimizer, feeder):
    model_options = ['num_layers', 'word_vec_size', 'rnn_size', 'bidirectional_encoder', 'attn_type', 'position_encoding',
        'head_count', 'transformer_hidden_size', 'transformer_enc_layers', 'transformer_dec_layers', 'model_type']
    model_options = {k:getattr(opt, k) for k in model_options}
    utils.ensure_folder(opt.ckpt_path)
    torch.save({
        'model':  model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'feeder': feeder.state(),
        'model_options': model_options
        }, opt.ckpt_path)


if __name__ == '__main__':
    model = Model()
    model.init_with_embedding(func.tensor([[1, 2, 3], [4, 5, 6]]).float(), func.tensor([[1, 2], [3, 4]]).float())
    assert not model.word_embedding.weight.requires_grad
    assert model.char_embedding.weight.requires_grad