import torch
from torch.autograd import Function
from torch import nn


class BiMapFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = (weight @ input) @ weight.t()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_variables
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = weight.t() @ grad_output @ weight

        if ctx.needs_input_grad[1]:
            e_grad = 2 * grad_output @ weight @ input
            e_grad = e_grad.sum(0)
            grad_weight = e_grad - e_grad @ weight.t() @ weight
        return grad_input, grad_weight


class BiMap(nn.Module):
    def __init__(self, input_features, output_features):
        super(BiMap, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(
            output_features, input_features))
        a = torch.rand(input_features, input_features)
        u, s, v = torch.svd(a@a.t())
        self.weight.data = u[:, :output_features].t()

    def forward(self, input):
        return BiMapFunction.apply(input, self.weight)
