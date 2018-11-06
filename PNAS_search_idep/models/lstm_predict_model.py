from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn




class PreAccLstm(nn.Module):
    def __init__(self, D, layers):
        super(PreAccLstm, self).__init__()
        self.input_embedding = nn.Embedding(7, D)
        self.ops_embedding = nn.Embedding(8, D)
        self.lstm1 = nn.LSTM(
                input_size=4*D,
                hidden_size=4*D,
                num_layers=layers,
                batch_first=True
                )
        self.linear =nn.Sequential(
                nn.Linear(4*D, 1),
                nn.Sigmoid()
                )
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.fill_(1.7)

    def forward(self, x):
        step = x.size(-1)
        step //= 2
        # x_embed = torch.LongTensor([])
        for i in range(step):
            if i == 0 :
                x1 = self.input_embedding(x[:,:,2*i])
                x2 = self.input_embedding(x[:,:,2*i+1])
            else:
                x3 = self.ops_embedding(x[:,:,2*i])
                x4 = self.ops_embedding(x[:,:,2*i+1])
        x_embed = torch.cat([x1, x2, x3, x4], -1)
        out,(h,c) = self.lstm1(x_embed, None)
        out = self.linear(out[:,-1,:])
        return out





