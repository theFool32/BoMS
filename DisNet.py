import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from LogEig import LogEigFunction
from myutil import pair, check_gpu
import pdb


class JBLD_Distance(Function):
    @staticmethod
    def logdet(a):
        # return 2 * torch.potrf(a).diag().log().sum().item()
        return 2 * np.sum(np.log(np.diag(np.linalg.cholesky(a))))

    @staticmethod
    def calc_dis(a, b):
        a = a.numpy()
        b = b.numpy()
        return (
            JBLD_Distance.logdet(0.5 * (a + b))
            - 0.5 * JBLD_Distance.logdet(a)
            - 0.5 * JBLD_Distance.logdet(b)
        )

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        a = x[:, : x.size(1) // 2].contiguous()
        b = x[:, x.size(1) // 2 :].contiguous()
        shape = int(np.sqrt(a.size(1)))
        a = a.view(a.size(0), shape, shape)
        b = b.view(b.size(0), shape, shape)
        output = torch.zeros(a.size()[0], 1)
        for i in range(a.size()[0]):
            output[i] = JBLD_Distance.calc_dis(a[i], b[i])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        a = x[:, : x.size(1) // 2].contiguous()
        b = x[:, x.size(1) // 2 :].contiguous()
        shape = int(np.sqrt(a.size(1)))
        a = a.view(a.size(0), shape, shape)
        b = b.view(b.size(0), shape, shape)
        grada = torch.zeros(a.size())
        gradb = torch.zeros(b.size())
        for i in range(a.size()[0]):
            inv = (a[i] + b[i]).inverse()
            grada[i] = inv - a[i].inverse() / 2
            gradb[i] = inv - b[i].inverse() / 2
            grada[i] *= grad_output.data[i]
            gradb[i] *= grad_output.data[i]
        grada = grada.view(a.size(0), shape ** 2)
        gradb = gradb.view(b.size(0), shape ** 2)
        return Variable(torch.cat((grada, gradb), 1))


class Log_DisNet(nn.Module):
    def __init__(self, input_dim):
        super(Log_DisNet, self).__init__()

    def forward(self, x, y, dij=None):
        x = LogEigFunction.apply(x)
        x = x.view(x.size(0), -1)
        y = LogEigFunction.apply(y)
        y = y.view(y.size(0), -1)
        fe = pair(y, x, dij)
        if check_gpu():
            fe = fe.cuda()

        x1 = fe[:, : x.size(1)].contiguous()
        x2 = fe[:, x.size(1) :].contiguous()
        d = torch.norm(x1 - x2 + 1e-16, dim=1)
        return d


class Log_w_DisNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Log_w_DisNet, self).__init__()
        self.dimLinear = nn.Linear(input_dim, output_dim, bias=None)
        if check_gpu():
            self.dimLinear = self.dimLinear.cuda()
        # nn.init.kaiming_uniform_(self.dimLinear.weight)
        nn.init.eye_(self.dimLinear.weight)

    def forward(self, x, y, dij=None):
        x = LogEigFunction.apply(x)
        x = x.view(x.size(0), -1)
        y = LogEigFunction.apply(y)
        y = y.view(y.size(0), -1)
        fe = pair(y, x, dij)
        if check_gpu():
            fe = fe.cuda()

        x1 = fe[:, : x.size(1)].contiguous()
        x1 = self.dimLinear(x1)
        x2 = fe[:, x.size(1) :].contiguous()
        x2 = self.dimLinear(x2)
        d = torch.norm(x1 - x2 + 1e-16, dim=1)
        return d


class JBLD_DisNet(nn.Module):
    def __init__(self):
        super(JBLD_DisNet, self).__init__()

    def forward(self, x, y, dij=None):
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        fe = pair(y, x, dij)
        return JBLD_Distance.apply(fe)
