import torch
import torch.nn as nn
import func

class DotAttention(nn.Module):
    def __init__(self, input_size, memory_size, hidden_size, dropout):
        super(DotAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense_input = nn.Linear(input_size, hidden_size, bias=False)
        self.dense_memory = nn.Linear(memory_size, hidden_size, bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.hidden_size = hidden_size
        self.res_size = input_size + memory_size
        self.dense_res = nn.Linear(self.res_size, self.res_size, bias=False)


    def forward(self, inputs, memory, mask):
        #attention
        d_inputs = self.dropout(inputs)
        d_memory = self.dropout(memory)
        h_inputs = self.dense_input(d_inputs).relu()
        h_memory = self.dense_memory(d_memory).relu()
        outputs = torch.bmm(h_inputs, h_memory.transpose(1, 2)) / (self.hidden_size ** 0.5)#[n,pl,ql]
        masked_outputs = func.softmax_mask(outputs, mask.unsqueeze(1))
        logits = self.sm(masked_outputs)
        outputs = torch.bmm(logits, memory)#[n,pl,dim=450]
        res = torch.cat([inputs, outputs], 2)#[n,pl,dim*2=900]

        #gate
        d_res = self.dropout(res)
        gate = self.dense_res(d_res).sigmoid()
        return res * gate
        

class Summary(nn.Module):
    def __init__(self, memory_size, hidden_size, dropout):
        super(Summary, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense_memory = nn.Linear(memory_size, hidden_size, bias=False)
        self.dense_one = nn.Linear(hidden_size, 1, bias=False)
        self.sm = nn.Softmax(dim=-1)


    def forward(self, memory, mask):
        d_memory = self.dropout(memory)
        s0 = self.dense_memory(d_memory).tanh()
        s = self.dense_one(s0)
        s1 = func.softmax_mask(s.squeeze(2), mask)
        a = self.sm(s1).unsqueeze(2)
        res = (a*memory).sum(1)
        return res