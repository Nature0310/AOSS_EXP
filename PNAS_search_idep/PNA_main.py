from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import argparse
import os
import time
import math
import sys
import numpy as np
from models.pna_auth import NetworkCifar
import cfar_10_data as cfar
from models.lstm_predict_model import PreAccLstm
import torch.utils.data as Data
from diverse import create_pnanet_graph, calculate_graph_dist
import multiprocessing

parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
# parser.add_argument('data', metavar='DIR', default='cifar10',
#                     help='path to dataset')
parser.add_argument('--model', default='PNAlstm', type=str, metavar='M',
                    help='model to train the dataset')
parser.add_argument('-F', default=24, type=int,
                    metavar='N', help='num of fliters')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('-k-num', default=5, type=int,
#                     metavar='N', help='num of choose arch')
parser.add_argument('--lr', '--learning-rate', default=0.025, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep'])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--distance', default=0.015, type=float,
                    metavar='D', help='diverse delta (default: 0.015)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--diverse', dest='diverse', action='store_true',
                    help='use pre-trained model (default: false)')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='use pre-trained model (default: false)')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='manual seed (default: 0)')
parser.add_argument('--block-start', default=1, type=int, metavar='N',
                    help='block start (default: 1)')
parser.add_argument('--id', default=0, type=int, metavar='N',
                    help='id (default: 0)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default='0',
                    help='gpu available')
parser.add_argument('--hidden', default=100, type=int)
parser.add_argument('--layers', default=1, type=int)
parser.add_argument('--knum', default=256, type=int)

parser.add_argument('--dicdir', type=str, metavar='PATH', default='dictionary/',
                    help='path to save model validation accuracy')
parser.add_argument('--savedir', type=str, metavar='PATH', default='results_PNA/savedir',
                    help='path to save result and checkpoint (default: results/savedir)')
# parser.add_argument('--resume', action='store_true',
#                     help='use latest checkpoint if have any (default: none)')

parser.add_argument('--celln', type=int, metavar='N', default=2,
                    help='num times to unroll cell')
parser.add_argument('--dropout-rate', default=0, type=float,
                    help='drop out (default: 0)')
parser.add_argument('--start', default=0, type=float,
                    help='drop out (default: 0)')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.data = "cifar10"
if args.data == 'cifar10':
    args.num_classes = 10
else:
    sys.exit(0)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random as rd
import pickle as p
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

def read_result(paths):
    if os.path.exists(paths):
        f = open(paths, 'rb')
        result = p.load(f)
        f.close()
    else:
        result = {}
    return result
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

def newTrainSet(last_choose, new_connect):
    new_sample = torch.LongTensor([])
    hidden_num = last_choose.size(1) + 1
    for h0 in last_choose:
        for h1 in new_connect:
            h = torch.cat([h0, h1], 0)
            h = h.resize(1, hidden_num, 4)
            new_sample = torch.cat([new_sample, h], 0)
    return new_sample
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def CNN_train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    learned_module_list = []

    ### Switch to train mode
    model.train()
    running_lr = None

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # progress = float(epoch * len(train_loader) + i) / \
        #            (args.epochs * len(train_loader))
        # args.progress = progress
        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        ### Measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        ### Compute output
        # output = model(input_var, progress)
        optimizer.zero_grad()
        output = model(input_var)
        # print (output.data)
        loss = criterion(output, target_var)

        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        ### Compute gradient and do SGD step
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}\t'  # ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f}\t'  # ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f}\t'  # ({top5.avg:.3f})'
                  'lr {lr: .4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
    return 100. - top1.avg, 100. - top5.avg, losses.avg, running_lr

def CNN_validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ### Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        ### Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

####xiugai
def CNN_save_checkpoint(args, result, filename):
    print(args)
    result_filename = os.path.join(args.savedir, filename)
    os.makedirs(args.savedir, exist_ok=True)
    print("=> saving result '{}'".format(result_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    return

def LSTM_train(model, train, loss_func, optim, epoch):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    model.train()
    # print("debug2", train)
    rd.shuffle(train)# shuffle train sequence
    print ("show train dataloader:", train)
    for train_loader in train:
        print ("<========= some block data =========>")
        for step,(batch_x, batch_y) in enumerate(train_loader):
            # lr = adjust_learning_rate
            data_time.update(time.time() - end)

            batch_y = batch_y.cuda(async=True)
            batch_x = torch.autograd.Variable(batch_x)
            batch_y = torch.autograd.Variable(batch_y)

            ###compute output
            optim.zero_grad()
            output = model(batch_x)
            loss = loss_func(output, batch_y)

            #### Measure accuracy and record loss
            losses.update(loss.data[0], batch_x.size(0))

            loss.backward()
            optim.step()


            ### Measure elapsed time
            batch_time.update(time.time() - end)

            if step % 3 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                      'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                      'Loss {loss.val:.4f}\t'.format(
                    epoch, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
    return losses.avg

def LSTM_prediction(model, data_loader, args, archtecture, num=args.knum, diverse=False):

    Y = torch.FloatTensor([]).cuda()
    topk = []
    prek = []
    for i, inputs in enumerate(data_loader):
        input_var = torch.autograd.Variable(inputs, volatile=True)
        y = 0
        for md in model:
            md.eval()
            y += md(input_var)
            # print(y.data)
        y = y / len(model)
        y = y.resize(input_var.size(0))
        # print(i)
        Y = torch.cat([Y, y.data], 0)
    if diverse:
        k_num, predic= Diverse(Y, archtecture, num)
        topk, prek = top_k(Y, archtecture, num)
    else:
        k_num, predic = top_k(Y, archtecture, num)

        topk, prek = Diverse(Y, archtecture, num)
    return k_num, predic, topk, prek

def target_fn(arg):
    G2 = create_pnanet_graph(arg['G2'], F=args.F, n=2)
    _, dist = calculate_graph_dist(arg['G1'], G2)
    result = dict(id=arg['id'], dist=dist)
    return result
def parallel_exc(func, args, num_pool=20):
    pool = multiprocessing.Pool(processes=num_pool)
    res = []
    for arg in args:
        res.append(pool.apply_async(func=func, args=(arg, )))
    print('waiting multi process ...')
    pool.close()
    pool.join()
    return res
def Diverse(Y, archtecture, num, deta=args.distance):
    Y = Y.cpu().numpy()
    index = np.argsort(-Y)
    Y = Y[index]
    archtecture = archtecture[index, :, :]
    _mask = np.ones_like(Y)
    diverse_k = []
    end = time.time()
    d_max = 1
    print ('debug4', d_max, time.time()-end)
    _m_num = 0
    end = time.time()
    for i in range(num):
        idx = np.where(_mask==1)[0]
        if len(idx) == 0:
            break
        diverse_k.append(idx[0])
        G1 = create_pnanet_graph(archtecture[idx[0]], F=args.F, n=2)
        dele = _m_num
        graphs = []
        for j in range(num*100-dele):
            graph = dict(id=j, G1=G1, G2=archtecture[idx[j]])
            graphs.append(graph)
        res = parallel_exc(func=target_fn, args=graphs)
        for r in res:
            distance = r.get()
            if distance['dist']/d_max < deta:
                _mask[idx[distance['id']]] = 0
                _m_num+=1

    idx = np.where(_mask==1)[0]
    print('debug5', idx[0], time.time()-end)
    prediction = Y[diverse_k]
    arch = archtecture[diverse_k, :, :]
    return arch, prediction



def top_k(Y, archtecture, num):
    Y = Y.cpu()
    Y = Y.numpy()
    topk = np.argsort(-Y)[:num]
    arch = archtecture[topk, :, :]
    prediction = Y[topk]
    return arch, prediction

def load_checkpoint(args):
    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state

def train_lstm(x, y, training, lr=0.003):
    Epochs = 120
    policy_model = PreAccLstm(D=args.hidden, layers=args.layers)
    print(policy_model)
    checkpoint = load_checkpoint(args)
    policy_model = torch.nn.DataParallel(policy_model).cuda()
    loss_func = nn.L1Loss()
    optim = torch.optim.Adam(policy_model.parameters(), lr=lr)
    if checkpoint is not None:
        policy_model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])

    torch_trainset = Data.TensorDataset(x, y)
    # print (torch_trainset)
    train_loader = Data.DataLoader(
        dataset=torch_trainset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    train = [train_loader]
    # print ("debug 0 ", training)
    for tr in training:
        # print ("debug 4", tr)
        idx = np.random.choice(tr[0].size(0), int(tr[0].size(0) * 0.95))
        # print(idx)
        # print(tr[1][idx])
        torch_trainset = Data.TensorDataset(tr[0][idx], tr[1][idx])
        train_loader = Data.DataLoader(
            dataset=torch_trainset,
            batch_size=2,
            shuffle=True,
            num_workers=2
        )
        train.append(train_loader)
    # print("debug1", train)
    for epoch in range(Epochs):
        ### Train for one epoch
        loss = LSTM_train(policy_model, train, loss_func, optim, epoch)
    return policy_model, loss







def load_data(arch, change=False):
    if change:
        acc = []
        for s in arch:
            use, ops = _oneto_str(s)
            filename = ("PNAnet%s__%s.txt" %
                        (use, ops))
            with open("/home/bigeye/lmy/PNAS_pytorch/results_test/savedir/"+filename, "r") as fp:
                lines =fp.readlines()
                w = lines[19].split(" ")[2]
                w = float(w) / 100
                acc.append(w)
        length = len(acc)
        acc = torch.Tensor(acc).resize(length, 1)
        print('block1', acc)
        return acc
    # with open("/home/lmy/Neural Archtecture Search/PNAS_pytorch/results_test/savedir/PNAS_0_block.txt", "r") as fp:

    fp = open("./PNAS_1_block.txt", "r")
    lines = fp.readlines()
    acc = []
    print("reading one block accuracy")
    for line in lines:
        w = line.split(":")[1][:-9]
        w = float(w) / 100
        acc.append(w)
    length = len(acc)
    acc = torch.Tensor(acc).resize(length, 1)
    print('block1', acc)
    return acc

def load_model_list(b):
    path = ('model_block%d.txt' % b)
    file_path = os.path.join(args.savedir, path)

    fp = open(file_path, "r")
    lines = fp.readlines()
    topk = torch.LongTensor([])
    k_num = torch.LongTensor([])
    predic = []
    prek = []
    for line in lines:
        w1 = line.split(":")[0]
        w2 =line.split(":")[2]
        k_num = torch.cat([k_num, str_to_arch(w1, b)], 0)
        topk = torch.cat([topk, str_to_arch(w2, b)], 0)
        predic.append(float(line.split(":")[1][:7]))
        prek.append(float(line.split(":")[3][:7]))
    return k_num, predic, topk, prek
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

def main():
    global args
    print (args)
    args.savedir = os.path.join(args.savedir, '%d' % args.id)
    # a = torch.FloatTensor([2,3,1.2,4.533])
    # a = a.numpy()
    # print(a[3])
    last_k = train_setting(input_size=2)
    # for g in last_k:
        # print(last_k)
    x = last_k
    y = load_data(last_k, False)
    train = []
    # result_dict = read_result('./model.pkl')
    # print (result_dict)



    for b in range(2, 6):
        new_connect = train_setting(input_size=b+1)### new child connection
        new_data = newTrainSet(last_k, new_connect)#### generate new archtecture for prediction
        ############# trainning on LSTM
        loss = []
        ensemble_model = []
        #if resume use last data
        modelpath = os.path.join(args.savedir, 'model_block%d.txt' % b)
        if os.path.exists(modelpath):
            k_num, predic, topk, prek = load_model_list(b)
        else:
            #
            for i in range(5):
                idx = np.random.choice(x.size(0), int(x.size(0) * 0.85))
                models, losses = train_lstm(x[idx], y[idx], train, lr=0.01 if b == 2 else 0.003)
                ensemble_model.append(models)
                loss.append(losses)
            result_lstm = os.path.join(args.savedir, "lstm_ensemble.txt")
            os.makedirs(args.savedir, exist_ok=True)
            with open(result_lstm, 'a') as fout:
                for ls in loss:
                    fout.write("%.4f  " % ls)
                fout.write("\n")
            #############   prediction for accuracy
            data_loader = Data.DataLoader(
                dataset=new_data,
                batch_size=64,
                shuffle=False,
                num_workers=4
            )
            k_num, predic, topk, prek = LSTM_prediction(ensemble_model, data_loader, args, new_data, diverse=args.diverse)####ensemble method
            print("+========================+")
            model_file = ('model_block%d.txt' % b)
            model_file_path = os.path.join(args.savedir, model_file)
            print ('saving: ' + model_file)
            for k1, GG1, k2, GG2 in zip(k_num, predic, topk, prek):
                with open(model_file_path, 'a') as fp:
                    a1, a2 = _oneto_str(k1)
                    print('debug1', a1, a2)
                    a3, a4 = _oneto_str(k2)
                    fp.write('%s %s : %.4f :%s %s : %.4f\n' % (a1, a2, GG1, a3, a4, GG2))
                # print(_oneto_str(k), GG)
            print("+========================+")
        val_acc = []
        ##############   train CNN given archtecture
        for j, s in enumerate(k_num):
            os.makedirs(args.dicdir, exist_ok=True)
            readpath = os.path.join(args.dicdir, './model.pkl')
            result_dict = read_result(readpath)
            if j < args.start:
                continue
            model = NetworkCifar(24, 10, 6, False, s, n=args.celln)
            print(model)
            use, ops = _oneto_str(s)
            name = use + ' ' + ops + ' '
            u, o = _oneto_str(topk[j, :, :])
            if result_dict.get(name) == None:

                # if args.diverse:
                filename = ("PNAnet%s__%s.txt" %     #####yaoxiugai
                            (use, ops))
                model = torch.nn.DataParallel(model).cuda()
                criterion = nn.CrossEntropyLoss().cuda()
                optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay,
                                            nesterov=True)
                assert args.data == "cifar10"
                normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                                 std=[0.2471, 0.2435, 0.2616])
                train_set = cfar.CIFAR10('./data', train=True, num_valid=5000, valid=False,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize,
                                         ]))
                test_set = datasets.CIFAR10('./data', train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize,
                                            ]))
                val_set = cfar.CIFAR10('./data', train=True, num_valid=5000, valid=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)

                val_loader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
                test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
                for epoch in range(args.epochs):
                    ##Train for one epoch
                    tr_prec1, tr_prec5, loss, lr = CNN_train(train_loader, model, criterion, optimizer, epoch)
                    ## Evaluate
                    val_prec1, val_prec5 = CNN_validate(val_loader, model, criterion)
                    test_prec1, test_prec5 = CNN_validate(test_loader, model, criterion)
                    CNN_save_checkpoint(
                        args,
                        "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" %
                        (test_prec1, test_prec5, val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr),
                        filename
                    )
                val_acc.append(val_prec1)
                fn = ("block_%d.txt" % b)
                result_dict = read_result(readpath)
                result_filename = os.path.join(args.savedir, fn)
                os.makedirs(args.savedir, exist_ok=True)

                ####
                test = test_prec1.cpu().numpy()
                valid = val_prec1.cpu().numpy()
                ####

                result_dict[name] = {'test': [test]}
                result_dict[name]['valid'] = [valid]
                save_dict(result_dict, readpath)
                with open(result_filename, 'a') as fout:
                    fout.write("%s %s : %.4f %.4f %.4f   %s %s : %.4f\n"
                               % (use, ops, val_prec1, test_prec1, predic[j], u, o, prek[j]))
                # else:
                #     with open(result_filename, 'a') as fout:
                #         fout.write("%s %s : %.4f %.4f \n"
                #                     % (use, ops, val_prec1, predic[j]))
            else:
                val_prec1 = average(result_dict[name]['valid'])
                test_prec1 = average(result_dict[name]['test'])

                #####
                val_prec1 = torch.from_numpy(np.array(val_prec1,dtype=np.float32)).cuda()
                test_prec1 = torch.from_numpy(np.array(test_prec1,dtype=np.float32)).cuda()
                ######

                val_acc.append(val_prec1)
                fn = ("block_%d.txt" % b)
                result_filename = os.path.join(args.savedir, fn)
                os.makedirs(args.savedir, exist_ok=True)
                # if args.diverse:
                is_writting = True
                with open(result_filename, 'a') as fou:
                    with open(result_filename, 'r') as fout:
                        lines =fout.readlines()
                        for line in lines:
                            if ("%s %s" % (use, ops)) in line.split(':')[0]:
                                is_writting = False
                        if is_writting:
                            fou.write("%s %s : %.4f %.4f %.4f   %s %s : %.4f\n"
                                        % (use, ops, val_prec1, test_prec1, predic[j], u, o, prek[j]))
        # print("debug1", len(val_acc))
        val = torch.Tensor(val_acc).resize(len(val_acc), 1) / 100
        # print (val)
        # y = torch.cat([y, val], 0)
        # x = torch.cat([x, k_num], 0)
        train.append([k_num, val])
        last_k = k_num
def average(list):
    a = 0
    for ele in list:
        a += ele
    a /= len(list)
    return a

def save_dict(result, path):
    f = open(path, 'wb')
    p.dump(result, f)
    f.close()

if __name__ == "__main__":
    main()
