from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import numpy as np
import copy
import collections
import torch

#
# ops = {
#     0: lambda stride, reduction: edict({'nout': 32 if reduction else 16, 'type': 'conv', 'ks': 1, 'stride': stride, 'padding': 0}),
#     1: lambda stride, reduction: edict({'nout': 32 if reduction else 16, 'type': 'sep_conv', 'ks': 3, 'stride': stride, 'padding': 1}),
#     2: lambda stride, reduction: edict({'nout': 32 if reduction else 16, 'type': 'sep_conv', 'ks': 5, 'stride': stride, 'padding': 2}),
#     3: lambda stride, reduction:
#     edict({'nout': 32 if reduction else 16,
#            'type': 'identity' if stride == 1 else 'conv',
#            'ks': 0 if stride == 1 else 1,
#            'stride': 0 if stride==1 else 1, 'padding': 0}),
#     4: lambda stride, reduction: edict({'nout': 0, 'type': 'avg_pol', 'ks': 0, 'stride': stride, 'padding': 1}),
#     5: lambda stride, reduction: edict({'nout': 0, 'type': 'max_pol', 'ks': 0, 'stride': stride, 'padding': 1}),
#     6: lambda stride, reduction: edict({'nout': 32 if reduction else 16, 'type': 'sep_conv', 'ks': 7, 'stride': stride, 'padding': 3}),
#     7: lambda stride, reduction: edict({'nout': 32 if reduction else 16, 'type': 'dil_conv', 'ks': 3, 'stride': stride, 'padding': 1}),
#
# }

ops = {
    0: lambda channel, stride, reduction: edict(
        {'nout': channel, 'type': 'conv', 'ks': 1, 'stride': stride, 'padding': 0}),
    1: lambda channel, stride, reduction: edict(
        {'nout': channel, 'type': 'sep_conv', 'ks': 3, 'stride': stride, 'padding': 1}),
    2: lambda channel, stride, reduction: edict(
        {'nout': channel, 'type': 'sep_conv', 'ks': 5, 'stride': stride, 'padding': 2}),
    3: lambda channel, stride, reduction:
    edict({'nout': channel,
           'type': 'identity' if stride == 1 else 'conv',
           'ks': 0 if stride == 1 else 1,
           'stride': 0 if stride == 1 else 1, 'padding': 0}),
    4: lambda channel, stride, reduction: edict(
        {'nout': 0, 'type': 'avg_pol', 'ks': 0, 'stride': stride, 'padding': 1}),
    5: lambda channel, stride, reduction: edict(
        {'nout': 0, 'type': 'max_pol', 'ks': 0, 'stride': stride, 'padding': 1}),
    6: lambda channel, stride, reduction: edict(
        {'nout': 32 if reduction else 16, 'type': 'sep_conv', 'ks': 7, 'stride': stride, 'padding': 3}),
    7: lambda channel, stride, reduction: edict(
        {'nout': 32 if reduction else 16, 'type': 'dil_conv', 'ks': 3, 'stride': stride, 'padding': 1}),

}

class Node:
    def __init__(self, name='node', attribute=None):
        self.name = name
        self.in_nodes = []
        self.out_nodes = []
        self.attribute = edict({'type': 'none', 'nout': 3, 'stride': 1, 'ks': 0, 'padding': 0})
        if not attribute is None:
            self.attribute.update(attribute)

    def __call__(self, *args, **kwargs):
        return self.name

    @property
    def real_attribute(self):
        return np.array([self.attribute[key] for key in ['nout', 'stride', 'ks', 'padding']])

    @property
    def in_degree(self):
        return len(self.in_nodes)

    @property
    def out_degree(self):
        return len(self.out_nodes)

    def __setattr__(self, key, value):
        self.__dict__[key] = value


class NetGraph(object):
    def __init__(self):
        self.G = edict()
        self.nodes = edict()
        self.anchors = []
        self.max_degree = np.zeros(3)
        self.degree_reps = None
        self.len_nodes = 0

    def add(self, node):
        self.nodes[node()] = copy.deepcopy(node)

    def add_nodes(self, node_list):
        for node in node_list:
            self.add(node)

    def create(self):
        for node_name, node in self.nodes.items():
            self.len_nodes += 1
            self.G[node_name] = edict({'in_nodes': node.in_nodes, 'out_nodes': node.out_nodes})
            self.max_degree[0] = max(self.max_degree[0], len(node.in_nodes))
            self.max_degree[1] = max(self.max_degree[1], len(node.out_nodes))
            self.max_degree[2] = max(self.max_degree[2], len(node.in_nodes) + len(node.out_nodes))
        self.max_degree = list(self.max_degree.astype(int))

    def set_anchor(self, node_names=None, reset=False):
        if len(self.anchors) == 0 or reset:
            anchors = []
            if node_names is None:
                name = []
                degree = []
                # dtype = [('name', '|S100'), ('degree', int)]
                for node_name, node in self.nodes.items():
                    if len(node.in_nodes) == 0:
                        anchors.append(node_name)
                    elif len(node.out_nodes) == 0:
                        anchors.append(node_name)
                    else:
                        name.append(node_name)
                        degree.append(node.in_degree + node.out_degree)
                        # record.append((node_name, node.in_degree + node.out_degree))
                if len(degree) > 0:
                    idx = np.arange(len(degree))
                    # np.random.shuffle(idx)
                    name = np.array(name)[idx]
                    degree = np.array(degree)[idx]

                    sort_index = np.argsort(np.array(degree))
                    sort_names = list(np.array(name)[sort_index])
                    N = int(np.ceil(np.log2(self.len_nodes) - len(anchors)))
                    if N > 0:
                        for i in range(1, N+1):
                            anchors.append(str(sort_names[-i]))
            else:
                anchors = node_names
            self.anchors = anchors

    def find_k_adjacent(self, node_name, k):
        assert k >= 1
        res_up = [self.G[node_name].in_nodes]
        res_down = [self.G[node_name].out_nodes]
        if k == 1:
            return res_up, res_down
        for i in range(2, k+1):
            k_up = []
            k_down = []
            for name in res_up[-1]:
                tmp_up, _ = self.find_k_adjacent(name, k=1)
                k_up += tmp_up[-1]
            for name in res_down[-1]:
                _, tmp_down = self.find_k_adjacent(name, k=1)
                k_down += tmp_down[-1]
            tmp_res_up = list(np.unique(np.array(k_up)))
            tmp_res_down = list(np.unique(np.array(k_down)))
            res_up.append(tmp_res_up)
            res_down.append(tmp_res_down)
        return res_up, res_down

    def cal_structure_reps(self, k=2, delta=0.1, reset=False):
        if self.degree_reps is None or reset:
            degree_reps = {}
            for node_name in self.G.keys():
                k_degree_in = np.zeros(self.max_degree[0] + 1)
                k_degree_out = np.zeros(self.max_degree[1] + 1)
                k_adjacent_up, k_adjacent_down = self.find_k_adjacent(node_name, k=k)
                assert k == len(k_adjacent_up)
                thresh = 1.0
                for i in range(k):
                    tmp_in = np.zeros(self.max_degree[0] + 1)
                    tmp_out = np.zeros(self.max_degree[1] + 1)
                    k_set = k_adjacent_up[i] + k_adjacent_down[i]
                    if len(k_set) != 0:
                        for k_node_name in k_set:
                            tmp_in[len(self.G[k_node_name].in_nodes)] += 1
                            tmp_out[len(self.G[k_node_name].out_nodes)] += 1
                        # tmp_in[len(self.G[node_name].in_nodes)] += 1
                        # tmp_out[len(self.G[node_name].out_nodes)] += 1
                    k_degree_in += tmp_in * thresh
                    k_degree_out += tmp_out * thresh
                    thresh *= delta
                degree_reps[node_name] = {'indegree': k_degree_in, 'outdegree':k_degree_out}
                self.nodes[node_name].reps_indegree = k_degree_in
                self.nodes[node_name].reps_outdegree = k_degree_out
            self.degree_reps = degree_reps


def cos_dist(a, b):
    return np.sum(a*b)/np.sqrt(np.sum(a * a))/np.sqrt(np.sum(b * b))


def euler_dist(a, b):
    return np.sqrt(np.sum((a - b) * (a - b)))


def calculate_attribute_dist(node1, node2):
    type_dist = 0 if node1.attribute.type == node2.attribute.type else 1
    real_dist = cos_dist(node1.real_attribute, node2.real_attribute)
    # real_dist = euler_dist(node1.real_attribute, node2.real_attribute)
    return type_dist + real_dist


def cal_norm2(a, b):
    Na = len(a)
    Nb = len(b)
    if Na < Nb:
        a_ = np.zeros(Nb)
        a_[0:Na] = a
        return np.sum((a_ - b) * (a_ - b))
    elif Na > Nb:
        b_ = np.zeros(Na)
        b_[0:Nb] = b
        return np.sum((a - b_) * (a - b_))
    else:
        return np.sum((a - b) * (a - b))


def cal_similarity(node1, node2, gamma_in=0.1, gamma_out=0.1, gamma_attr=2):
    in_dist = cal_norm2(node1.reps_indegree, node2.reps_indegree)
    in_dist *= gamma_in

    out_dist = cal_norm2(node1.reps_outdegree, node2.reps_outdegree)
    out_dist *= gamma_out

    attr_dist = calculate_attribute_dist(node1, node2)
    attr_dist *= gamma_attr

    return np.exp(- in_dist - out_dist - attr_dist)


def normlization(Y):
    return Y / np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))


def calculate_graph_dist(G1, G2, k=2, delta=0.1):
    G1.set_anchor()
    G2.set_anchor()
    G1.cal_structure_reps(k=k, delta=delta)
    G2.cal_structure_reps(k=k, delta=delta)
    anchors = [G1.nodes[name] for name in G1.anchors]
    anchors += [G2.nodes[name] for name in G2.anchors]
    p = len(anchors)
    Matrix_Y = np.zeros((G1.len_nodes + G2.len_nodes, p))
    ind = []
    i = -1
    for node_name, node in G1.nodes.items():
        i += 1
        j = -1
        if node_name in G1.anchors:
            ind.append(i)
        for anchor in anchors:
            j += 1
            Matrix_Y[i, j] = cal_similarity(node, anchor)

    for node_name, node in G2.nodes.items():
        i += 1
        j = -1
        if node_name in G2.anchors:
            ind.append(i)
        for anchor in anchors:
            j += 1
            Matrix_Y[i, j] = cal_similarity(node, anchor)

    Matrix_W = Matrix_Y[np.array(ind)]
    inv_W = np.linalg.pinv(Matrix_W)
    u, s, vh = np.linalg.svd(inv_W)

    Matrix_Y = np.dot(Matrix_Y, u)
    Matrix_Y = np.dot(Matrix_Y, np.diag(s) ** 0.5)
    Matrix_Y = normlization(Matrix_Y)
    Y1 = Matrix_Y[0:G1.len_nodes]
    Y2 = Matrix_Y[G1.len_nodes:]
    dis_m = np.sum(Y1 * Y1, axis=1, keepdims=True) \
            + np.transpose(np.sum(Y2 * Y2, axis=1, keepdims=True), (1, 0)) \
            - 2 * np.dot(Y1, np.transpose(Y2, (1, 0)))
    # print(np.exp(-dis_m))
    # print(np.max(dis_m, axis=0))
    dist = np.mean(np.min(dis_m, axis=0)) + np.mean(np.min(dis_m, axis=1))
    # print('distance:%.4f' % dist )
    return Matrix_Y, dist



class CreatePNANetGraph(object):
    def __init__(self, archtecture, channel, cell_n=2):
        num_block = archtecture.size(0)
        self.NODES = collections.OrderedDict()
        self.channel = 3*channel
        self.NODES['input_node'] = Node(name='input_node', attribute=edict({'nout': 3, 'type': 'image'}))
        self.NODES['init_conv_node'] = Node(name='init_conv_node',
                                            attribute=edict({'nout': self.channel, 'type': 'conv', 'ks': 3,
                                                             'stride': 1, 'padding': 1}))

        # define node
        use = []
        for i in range(3*cell_n):
            if i == 0:
                cell_use = [self.NODES['init_conv_node'](), self.NODES['init_conv_node']()]
            else:
                cell_use = [cell_use[1], self.NODES['cell.%d_concate' % (i-1)]()]

            for j, block in enumerate(archtecture):

                reduction = True if i in [cell_n, 2*cell_n + 1] else False
                channel =  2*channel if i in [cell_n, 2*cell_n + 1] else channel
                # block left
                stride = 2 if reduction and int(block[0]) < 2 else 1
                self.NODES['cell%d_block%d_op1' % (i, j)] = Node(name='cell%d_block%d_op1' % (i, j),
                                                                 attribute=ops[int(block[2])](channel,stride, reduction))
                # block right
                stride = 2 if reduction and int(block[1]) < 2 else 1
                self.NODES['cell%d_block%d_op2' % (i, j)] = Node(name='cell%d_block%d_op2' % (i, j),
                                                                 attribute=ops[int(block[3])](channel, stride, reduction))
                # add
                self.NODES['cell%d_block%d_add' % (i, j)] = Node(name='cell%d_block%d_adda' % (i, j),
                                                                 attribute=edict({'type': 'add'}))
                # if i == 2*cell_n/3:
                #     self.NODES['AuxiliaryHead_avg'] = Node(name='Aux_avg', attribute=ops[4]( num_block*(channel), 2, reduction))
                #     self.NODES['AuxiliaryHead_cov1'] = Node(name='Aux_cov1', attribute=ops[0]( num_block*(channel), 1, reduction))
                #     self.NODES['AuxiliaryHead_cov2'] = Node(name='Aux_cov2', attribute=ops[1](128, 1, reduction))
                cell_use.append(self.NODES['cell%d_block%d_add' % (i, j)]())
            self.NODES['cell.%d_concate' % (i)] = Node(name='cell.%d_block%d_concate' % (i, j),
                                                                attribute=edict({'type': 'concate',
                                                                'nout': num_block*(channel)}))
            use.append(cell_use)
        # self.NODES['AuxiliaryHead_avg'] = Node(name='Aux_avg', attribute=ops[4])
        #  define link
        self.NODES['input_node'].in_nodes = []
        self.NODES['input_node'].out_nodes = [self.NODES['init_conv_node']()]
        self.NODES['init_conv_node'].in_nodes = [self.NODES['input_node']()]
        self.NODES['init_conv_node'].out_nodes =[]
        for v, block in enumerate(archtecture):
            if block[0] <= 1:
                self.NODES['init_conv_node'].out_nodes.append(self.NODES['cell%d_block%d_op1' % (0, v)]())
                if block[0] == 0:
                    self.NODES['init_conv_node'].out_nodes.append(self.NODES['cell%d_block%d_op1' % (1, v)]())
            if block[1] <= 1:
                self.NODES['init_conv_node'].out_nodes.append(self.NODES['cell%d_block%d_op2' % (0, v)]())
                if block[1] == 0:
                    self.NODES['init_conv_node'].out_nodes.append(self.NODES['cell%d_block%d_op2' % (1, v)]())
        for i in range(3*cell_n):
            used_input = use[i]
            for j, block in enumerate(archtecture):
                # block left
                self.NODES['cell%d_block%d_op1' % (i, j)].in_nodes = [used_input[int(block[0])]]
                self.NODES['cell%d_block%d_op1' % (i, j)].out_nodes = [self.NODES['cell%d_block%d_add' % (i, j)]()]
                # block right
                self.NODES['cell%d_block%d_op2' % (i, j)].in_nodes = [used_input[int(block[1])]]
                self.NODES['cell%d_block%d_op2' % (i, j)].out_nodes = [self.NODES['cell%d_block%d_add' % (i, j)]()]
                # add
                self.NODES['cell%d_block%d_add' % (i, j)].in_nodes = [self.NODES['cell%d_block%d_op1' % (i, j)](),
                                                                     self.NODES['cell%d_block%d_op2' % (i, j)]()]
                self.NODES['cell%d_block%d_add' % (i, j)].out_nodes = [self.NODES['cell.%d_concate' % (i)]()]
            # concate
            self.NODES['cell.%d_concate' % (i)].in_nodes = [self.NODES['cell%d_block%d_add' % (i, h)]()
                                                            for h in range(num_block)]
            a = []
            b = []
            for v, block in enumerate(archtecture):
                if i < 3*cell_n-2:
                    if block[0] == 1:
                        a.append(self.NODES['cell%d_block%d_op1' % (i+1, v)]())
                    elif block[0] == 0:
                        b.append(self.NODES['cell%d_block%d_op1' % (i + 2, v)]())

                    if block[1] == 1:
                        a.append(self.NODES['cell%d_block%d_op2' % (i + 1, v)]())
                    elif block[1] == 0:
                        b.append(self.NODES['cell%d_block%d_op2' % (i + 2, v)]())
                elif i == 3*cell_n-2:
                    if block[0] == 1:
                        a.append(self.NODES['cell%d_block%d_op1' % (i+1, v)]())
                    elif block[1] == 1:
                        a.append(self.NODES['cell%d_block%d_op2' % (i + 1, v)]())
            self.NODES['cell.%d_concate' % (i)].out_nodes = a
            for p in b:
                self.NODES['cell.%d_concate' % (i)].out_nodes.append(p)
            # if i == 2 * cell_n / 3:
            #     self.NODES['AuxiliaryHead_avg'].in_nodes = [self.NODES['cell.%d_concate' % (i)]()]
            #     self.NODES['AuxiliaryHead_avg'].out_nodes = [self.NODES['AuxiliaryHead_cov1']()]
            #     self.NODES['AuxiliaryHead_cov1'].in_nodes = [self.NODES['AuxiliaryHead_avg']()]
            #     self.NODES['AuxiliaryHead_cov1'].out_nodes = [self.NODES['AuxiliaryHead_cov2']()]
            #     self.NODES['AuxiliaryHead_cov2'].in_nodes = [self.NODES['AuxiliaryHead_cov1']()]
            #     self.NODES['AuxiliaryHead_cov2'].out_nodes = []
    def create(self):
        Graph1 = NetGraph()
        Graph1.add_nodes([self.NODES[name] for name in self.NODES.keys()])
        Graph1.create()
        return Graph1




def create_pnanet_graph(arch, F, n):
    G1 = CreatePNANetGraph(archtecture=arch, channel=F ,cell_n=n)
    return G1.create()




def train_setting(input_size, ops_size=8):
    elements = crea_double(input_size=input_size, ops_size=ops_size)
    S1 = torch.LongTensor([])
    for i in range(len(elements)):
        I1 = elements[i][0]
        O1 = elements[i][1]
        for j in range(i, len(elements)):
            I2 = elements[j][0]
            O2 = elements[j][1]
            # S1.append([[I1, I2, O1, O2]])
            s1 = torch.LongTensor([[[I1, I2, O1, O2]]])
            S1 = torch.cat([S1, s1], 0)
    return S1

def crea_double(input_size, ops_size):
    s1 = []
    for i in range(input_size):
        for j in range(ops_size):
            s1.append([i, j])
    return s1
def _oneto_str(compre_op_lc):
    use_prev = ''
    ops = ''
    for block in compre_op_lc:
        I1 = str(int(block[0]))
        I2 = str(int(block[1]))
        O1 = str(int(block[2]))
        O2 = str(int(block[3]))
        # result.append([I1, I2, O1, O2])
        use_prev += I1+I2
        ops += O1+O2
    return use_prev, ops
# if __name__ == '__main__':
#     S1 =  train_setting(2)
#     print (S1[5])
#     F=24
#     G1 = create_pnanet_graph(S1[0], F, n=2)
#     for s in S1:
#         G2 = create_pnanet_graph(s, F, n=2)
#         Y, dist = calculate_graph_dist(G1, G2)
#         a = _oneto_str(s)
#         print(a, dist)
def str_to_arch(w, b):
    arch = torch.LongTensor([])
    print(w)
    for idx in range(b):
        I1 = int(w[idx*2])
        I2 = int(w[idx*2+1])
        O1 = int(w[b*2+1+idx*2])
        O2 = int(w[b*2+2+idx*2])
        oneblock = torch.LongTensor([[[I1, I2, O1, O2]]])
        arch = torch.cat([arch, oneblock], 1)
    return arch
if __name__ == '__main__':
    # name = load_data('/home/lmy/Neural Archtecture Search/PNAS_pytorch/9_20_topk/lstm64_1/block_3.txt')
    S1 =  train_setting(2)
    c = S1[0]
    # print (name[0])
    a = str_to_arch('1102031101 2341245323 ', b=5)
    G1 = create_pnanet_graph(S1[23], F=24, n=2)
    d_max = 0.636
    # for s in S1:
    #     # s = str_to_arch(s, b=3)[0]
    #     G2 = create_pnanet_graph(s, n=2)
    #     Y, dist = calculate_graph_dist(G1, G2)
    #     if d_max < dist:
    #         d_max = dist
    print(d_max)
    for s in S1:
        # s = str_to_arch(s, b=3)[0]
        G2 = create_pnanet_graph(s, n=2)
        Y, dist = calculate_graph_dist(G1, G2)
        a = _oneto_str(s)
        print(a, dist/d_max)