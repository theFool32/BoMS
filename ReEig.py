import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch import nn
from myutil import calc_Q, teig

_ep = 1e-6


class ReEigFunction(Function):

    @staticmethod
    def forward(ctx, input, ep):
        # output = torch.ones(input.shape, device=input.device)
        output = torch.ones(input.shape)
        ctx.__e = []
        ctx.__v = []
        ctx.__ep = ep
        for i in range(input.shape[0]):
            e, v = teig(input[i])
            ctx.__e.append(e.clone())
            ctx.__v.append(v)
            e[e < ep] = ep
            output[i] = v @ e.diag() @ v.t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ep = ctx.__ep
        grad_input = None

        if ctx.needs_input_grad[0]:
            # grad_input = torch.ones(
            #     grad_output.size(), device=grad_output.device)
            grad_input = torch.ones(grad_output.size())
            for i in range(grad_output.size()[0]):
                e = ctx.__e[i]
                v = ctx.__v[i]
                t = e.clone()
                t[t <= ep] = ep
                grad_u = ((grad_output.data[i] + grad_output.data[i].t())
                          @ v @ torch.diag(t))

                q = torch.diag(e > ep).float()
                grad_e = q @ v.t() @ (
                    grad_output.data[i]+grad_output.data[i].t())/2 @ v

                P = calc_Q(e)
                s = v.t() @ grad_u
                s = P.t() * s
                s = (s+s.t())/2
                d = torch.diag(torch.diag(grad_e))
                grad_input[i] = v @ (s+d) @ v.t()
        return Variable(grad_input), None


class ReEig(nn.Module):
    def __init__(self, ep):
        super(ReEig, self).__init__()
        self.ep = ep

    def forward(self, input):
        return ReEigFunction.apply(input, self.ep)
