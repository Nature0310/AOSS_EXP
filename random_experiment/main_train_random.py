from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import argparse
import os
import shutil
import time
import math
import warnings
import models
import random
import sys
from models.macro_model import NetworkCifar
import cfar_10_data as cfar
parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
# parser.add_argument('data', metavar='DIR', default='cifar10',
#                     help='path to dataset')
parser.add_argument('--model', default='condensenet', type=str, metavar='M',
                    help='model to train the dataset')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--normal-name', default='1111311000 6521153125 ', type=str,
                    help='model_name')
parser.add_argument('--range', default='1-5', type=str,
                    help='range_num')
parser.add_argument('--reduction-name', default=None, type=str,
                    help='model_name(None）')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-F', default=48, type=int,
                    metavar='N', help='num of fliters')
parser.add_argument('--lr', '--learning-rate', default=0.025, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep'])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip-gradient', '--cg', default=5, type=float,
                    metavar='W', help='clip gradient (default: 5)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model (default: false)')
parser.add_argument('--no-save-model', dest='no_save_model', action='store_true',
                    help='only save best model (default: false)')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='manual seed (default: 0)')
# parser.add_argument('--clip-gradient', default=5, type=float, metavar='N',
#                     help='clip-gradient (default: 5)')
parser.add_argument('--gpu', default='5',
                    help='gpu available')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', action='store_true',
                    help='use latest checkpoint if have any (default: none)')

parser.add_argument('--random-NF', action='store_true',
                    help='random celln and F (default: none)')
parser.add_argument('--random', action='store_true',
                    help='use latest checkpoint if have any (default: none)')
parser.add_argument('--savedir', type=str, metavar='PATH', default='randomtest/savedir',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--celln', type=int, metavar='N', default=3,
                    help='num times to unroll cell')
parser.add_argument('--drop-path-prob', default=0.4, type=float,
                    help='drop out (default: 0.4)')
parser.add_argument('--cutout', default=1, type=int,
                    help='cutout (default: 1)')
parser.add_argument('--cutout-length', default=16, type=int,
                    help='cutout-length (default: 16)')

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
import numpy as np

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
best_prec1 = 0
global_step = 0
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform
def load_data(filepath ,topk=10,change=True):
    fp = open(filepath, "r")
    lines = fp.readlines()
    acc = []
    archs= []
    print("reading one block accuracy")
    for line in lines:
        arch = line.split(":")[0]
        w = line.split(":")[1][9:16]
        w = float(w) / 100
        acc.append(w)
        archs.append(arch)
    # length = len(acc)
    acc = np.array(acc)
    index = np.argsort(acc)[-10:]
    arch=[]
    for i in index:
        arch.append(archs[i])
    return arch

    # acc = torch.Tensor(acc).resize(length, 1)
    # print('block1',acc)
    # return acc,
def str_to_arch(w):
    if w == None:
        return w
    b = len(w.split(' ')[0]) // 2
    arch = torch.LongTensor([])
    print(w)
    for idx in range(b):
        I1 = int(w[idx*2])
        I2 = int(w[idx*2+1])
        O1 = int(w[b*2+1+idx*2])
        O2 = int(w[b*2+2+idx*2])
        oneblock = torch.LongTensor([[[I1, I2, O1, O2]]])
        arch = torch.cat([arch, oneblock], 1)
    arch = arch.resize(arch.size(1), arch.size(2))
    return arch
def random_model(blocks=5):
    inputs = ['0', '1']
    ops = ['0', '1', '2', '3', '4', '5', '6', '7']
    con = ''
    op = ''
    for i in range(blocks):
        # a = random.sample(inputs, 1)
        con += random.sample(inputs, 1)[0]
        con += random.sample(inputs, 1)[0]
        op += random.sample(ops, 1)[0]
        op += random.sample(ops, 1)[0]
        inputs.append(str(i+2))
    arch = con + ' ' + op + ' '
    return arch
def add_model_s(index):
    global args
    os.makedirs(args.savedir, exist_ok=True)
    names = os.listdir(args.savedir)
    exist = False
    s_finish = False
    finish = False
    arch = None
    filename =None
    for name in names:
        if name[-3:] == 'txt'and name.split('_')[0] == 's' + str(index):
            exist = True
            filename = name
            elements = filename.split('_')
            arch = elements[1] + ' ' + elements[2] + ' '
            args.celln = int(elements[3])
            args.F = int(elements[4][:-4])
            break
    if exist:
        filepath = os.path.join(args.savedir, filename)
        fp = open(filepath, "r")
        lines = fp.readlines()
        if len(lines)==50:
            s_finish = True
            filename = filename[1:]
            filepath = os.path.join(args.savedir, filename)
            if os.path.exists(filepath):
                fp = open(filepath, "r")
                lines = fp.readlines()
                if len(lines)==600:
                    finish = True

    return exist, s_finish, arch, finish, filename



def main(index):
    global args
    global best_prec1
    global global_step
    if args.random_NF:
        args.celln = random.randint(2, 5)
        args.F = random.randint(30, 50)
    s2 = None
    cudnn.benchmark = True
    exist, S_finish, s, Finish, args.filename = add_model_s(index)
    print (s)
    s = str_to_arch(s)
    if not exist:
        s = random_model()
        s = str_to_arch(s)
        use, ops = _oneto_str(s)
        args.filename = ("s%d_%s_%s_%d_%d.txt" %
                         (index, use, ops, args.celln, args.F))
    if not Finish:
        model = NetworkCifar(args.F, 10, args.celln * 3, True, s, n=args.celln, genotype_reduction=s2)
    else:
        return
    # if not S_finish:
    #     model = S_NetworkCifar(args.F, 10, args.celln * 3, True, s, n=args.celln, genotype_reduction=s2)

    # model = NetworkCifar(args.F, 10, args.celln*3, True, s, n=args.celln, genotype_reduction=s2)
    if not Finish and S_finish:
        args.epochs = 600
        args.drop_path_prob = 0.4
        args.clip_gradient = 5
        args.start_epoch = 0
        global_step = args.epochs * math.ceil(50000 / args.batch_size)
        best_prec1 = 0
        print('<<<<<<just training macro model>>>>>>>')
        do_it(model)
    if not S_finish:
        args.epochs = 50
        args.drop_path_prob = 0
        args.clip_gradient = 30
        best_prec1 = 0
        global_step = args.epochs * math.ceil(50000 / args.batch_size)
        print('<<<<<<training the same models>>>>>>>')
        for i in range(2):
            do_it(model)
            model = NetworkCifar(args.F, 10, args.celln * 3, True, s, n=args.celln, genotype_reduction=s2)
            args.epochs = 600
            args.start_epoch = 0
            global_step = args.epochs * math.ceil(50000 / args.batch_size)
            args.drop_path_prob = 0.4
            args.clip_gradient = 5
            args.filename = args.filename[1:]
            best_prec1 = 0


def do_it(model):
    global best_prec1
    global args
    print(model)
    params = model.named_parameters()
    k = 0
    for name, param in params:
        l = 1
        for j in param.size():
            l *= j
        if 'aux' not in name:
            k = k + l
    print("The sum of parameters：" + str(k / 1e6))
    a = np.sum(np.prod(v.size()) for v in model.parameters())
    print(a / 1e6)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    loss_function = criterion
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    assert args.data == "cifar10"
    transforms_train, transforms_valid = _data_transforms_cifar10(args)
    train_set = cfar.CIFAR10('./data', train=True, num_valid=5000, valid=False,
                             transform=transforms_train
                             )
    test_set = datasets.CIFAR10('./data', train=False,
                                transform=transforms_valid
                                )
    val_set = cfar.CIFAR10('./data', train=True, num_valid=5000, valid=True,
                           transform=transforms_valid
                           )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    for epoch in range(args.start_epoch, args.epochs):
        ##Train for one epoch
        tr_prec1, tr_prec5, loss, lr = train(train_loader, model, loss_function, optimizer, epoch)
        ## Evaluate
        val_prec1, val_prec5 = validate(val_loader, model, loss_function)
        test_prec1, test_prec5 = validate(test_loader, model, criterion)
        is_best = val_prec1 > best_prec1
        print(is_best, val_prec1, best_prec1)
        best_prec1 = max(val_prec1, best_prec1)
        model_filename = 'lastest.pth.tar'
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" %
                                          (test_prec1, test_prec5, val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr))

def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.savedir, args.filename)
    model_dir = os.path.join(args.savedir, args.filename[:-4])
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if args.no_save_model:
        shutil.move(model_filename, best_filename)
    elif is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

def crea_arc_b1(in_one_hot, ops_one_hot):
    elements = []
    for i in range(2):
        i1 = in_one_hot[i]
        for j in range(8):
            o1 = ops_one_hot[j]
            elements.append([i1,o1])
    return elements

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

def train(train_loader, model, criterion, optimizer, epoch, training=True):
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

        #### Computer drop_path_prob
        current_step = ((epoch)/args.epochs) * global_step + i
        current_radio = min(1, current_step / global_step)
        drop_prob = current_radio * args.drop_path_prob
        ####

        ### Compute output
        # output = model(input_var, progress)
        optimizer.zero_grad()
        output, output2 = model(input_var, training, drop_prob)
        # print (output.data)
        loss1 = criterion(output, target_var)
        loss2 = criterion(output2, target_var)
        loss = loss1 + 0.4 * loss2
        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        ### Compute gradient and do SGD step
        # optimizer.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
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
    return top1.avg, top5.avg, losses.avg, running_lr

def validate(val_loader, model, criterion, training=False):
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
        output = model(input_var, training)
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
# def save_checkpoint(args, result, filename):
#     print(args)
#     result_filename = os.path.join(args.savedir, filename)
#     os.makedirs(args.savedir, exist_ok=True)
#     print("=> saving result '{}'".format(result_filename))
#     with open(result_filename, 'a') as fout:
#         fout.write(result)
#     return

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
def load_checkpoint(args):
    model_dir = os.path.join(args.savedir, args.filename[:-4])
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

if __name__ == '__main__':
    # num = 1 + args.range.split('-')[1] - args.range.split('-')[0]
    begin = int(args.range.split('-')[0])
    end = int(args.range.split('-')[1])+1

    for index in range(begin, end):
        main(index)