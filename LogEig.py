import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch import nn
from myutil import calc_Q, teig

_ep = 1e-6


class LogEigFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # output = torch.ones(input.shape, device=input.device)
        output = torch.ones(input.size())
        ctx.__e = []
        ctx.__v = []
        for i in range(input.shape[0]):
            e, v = teig(input[i])
            ctx.__e.append(e.clone())
            ctx.__v.append(v)
            e = torch.log(e)
            output[i] = v @ torch.diag(e) @ v.t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None

        if ctx.needs_input_grad[0]:
            # grad_input = torch.ones(
            #     grad_output.size(), device=grad_output.device)
            grad_input = torch.ones(grad_output.size())
            for i in range(grad_output.size(0)):
                e = ctx.__e[i]
                v = ctx.__v[i]
                grad_u = ((grad_output.data[i] + grad_output.data[i].t()) @ v
                          @ torch.diag(e.log()))
                grad_e = (torch.diag(e).inverse() @ v.t() @ (
                    grad_output.data[i]+grad_output.data[i].t())/2 @ v)
                P = calc_Q(e)
                s = v.t() @ grad_u
                s = P.t() * s
                s = (s+s.t())/2
                d = torch.diag(torch.diag(grad_e))
                grad_input[i] = v @ (s + d) @ v.t()
        return Variable(grad_input)


class LogEig(nn.Module):
    def __init__(self):
        super(LogEig, self).__init__()

    def forward(self, input):
        return LogEigFunction.apply(input)
