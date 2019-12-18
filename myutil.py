import numpy as np
import torch
from torch.autograd import Variable
import pdb
from torch import nn


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


def teig(x):
    e, v = torch.symeig(x, eigenvectors=True)
    return e, v
    try:
        v, e, _ = torch.svd(x)
    except:
        __import__("pdb").set_trace()
    return e, v


def calc_Q(e):
    dim = e.size(0)
    s = e.view(dim, 1)
    ones = torch.ones(1, dim)
    s = s @ ones
    k = 1 / (s - s.t())
    k[torch.eye(dim) > 0] = 0
    k[k == float("Inf")] = 0
    return k


def dij(xy, zy):
    n = xy.size(0)
    m = zy.size(0)
    return xy.view(n, 1).expand(n, m).eq(zy.view(m, 1).t().expand(n, m)).float()


def pair(a, b, dij=None):
    na = a.size(0)
    nb = b.size(0)
    dim = a.size(1)
    a = a.view(na, -1).unsqueeze(1).expand(na, nb, dim)
    b = b.view(nb, -1).unsqueeze(0).expand(na, nb, dim)
    result = torch.cat((a, b), dim=2).view(na * nb, -1)
    if dij is None:
        return result
    index = dij.byte().view(-1, 1).expand(na * nb, dim * 2)
    return result[index].view(-1, dim * 2)


def group_list(x, y, group_size):
    for i in range(0, len(x), group_size):
        yield x[i : i + group_size], y[i : i + group_size]


def check_gpu():
    return torch.cuda.is_available()


def one_hot(size, index):
    mask = torch.LongTensor(index.size(0), size).fill_(0).to(index.device)
    ret = mask.scatter_(1, index.view(-1, 1), 1)
    return ret


class bcolors:

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    def str(s, color):
        return color + s + bcolors.ENDC
