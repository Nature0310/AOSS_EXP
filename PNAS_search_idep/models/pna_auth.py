
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from models.operation import *
from torch.autograd import Variable
import math


from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
# from utils import drop_path
PNASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 0),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 1),
    ('max_pool_3x3', 1),
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 4),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 0),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 1),
    ('max_pool_3x3', 1),
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 4),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('skip_connect', 1),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
)

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.reduction = reduction
        self._op_ = [
            'conv_1x1',
            'sep_conv_3x3',
            'sep_conv_5x5',
            'identity',
            'avg_pool_3x3',
            'max_pool_3x3',
            'sep_conv_7x7',
            'dil_conv_3x3'
        ]
        indices, op_names = self._oneto_num(genotype)
        if reduction_prev is None:
            self.preprocess0 = Identity()
        elif reduction_prev is True:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
            # self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 2, 0)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        # if reduction:
        #     op_names, indices = zip(*genotype.reduce)
        #     concat = genotype.reduce_concat
        # else:
        #     op_names, indices = zip(*genotype.normal)
        #     concat = genotype.normal_concat

        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        # self._concat = concat
        # self.multiplier = len(concat)

        self.multiplier = len(op_names) // 2
        if 1 not in indices or 0 not in indices:
            self.multiplier += 1
            if reduction:
                self.preprocess3 = FactorizedReduce(C, C)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            if reduction_prev is None and index == 0:## hu lue zhe yi hang
                op = OPS[name](C_prev_prev, C, stride, True)
            else:
                op = OPS[name](C, C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            # if self.training and drop_prob > 0.:
            #   if not isinstance(op1, Identity):
            #     h1 = drop_path(h1, drop_prob)
            #   if not isinstance(op2, Identity):
            #     h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        if 0 not in self._indices:
            del states[1]
            if self.reduction:
                states[0] = self.preprocess3(s0)
        elif 1 not in self._indices:
            del states[0]
            if self.reduction:
                states[0] = self.preprocess3(s1)
        else:
            states = states[2:]
        # states = states[2:]
        return torch.cat([hid for hid in states], dim=1)

    def _oneto_num(self, compre_op_lc):
        use_prev = []
        ops = []
        for block in compre_op_lc:
            I1 = int(block[0])
            I2 = int(block[1])
            O1 = int(block[2])
            O2 = int(block[3])
            # result.append([I1, I2, O1, O2])
            use_prev.append(I1)
            use_prev.append(I2)
            ops.append(self._op_[O1])
            ops.append(self._op_[O2])
        return use_prev, ops


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCifar(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype=PNASNet, n=2):
        super(NetworkCifar, self).__init__()
        self.drop_path_prob = 0
        self._layers = layers
        self._auxiliary = auxiliary
        self.C = C
        self.stem_multiplier = 3
        C_curr = self.stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_curr, eps=1e-3)
        )
        # self.stem1 = Cell(genotype, 96, 96, C // 4, True, None)
        # self.stem2 = Cell(genotype, 96, C * self.stem1.multiplier // 4, C // 2, True, True)
        #
        # C_prev_prev, C_prev, C_curr = C * self.stem1.multiplier // 4, C * self.stem2.multiplier // 2, C
        self.n = n
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # self.init_conv = nn.Conv2d(3, self.C, kernel_size=1, stride=1, padding=0, bias=False)

        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [self.n, 2 * self.n]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            # if i == 2 * layers // 3:
            #     C_to_auxiliary = C_prev

        # if auxiliary:
        #     self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.relu = nn.ReLU(inplace=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, input):
        # logits_aux = None
        # s0 = self.conv0(input)
        # s0 = self.conv0_bn(s0)
        # s1 = self.stem1(s0, s0, self.drop_path_prob)
        # s0, s1 = s1, self.stem2(s0, s1, self.drop_path_prob)
        # s0 = self.init_conv(input)
        s0 = s1 = self.stem(input)
        # s0 = torch.zeros_like(s0)
        # s1 = self.init_conv(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            # if i == 2 * self._layers // 3:
            #     if self._auxiliary and self.training:
            #         logits_aux = self.auxiliary_head(s1)
        s1 = self.relu(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
