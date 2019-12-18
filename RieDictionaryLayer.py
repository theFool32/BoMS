import torch
from torch.autograd import Variable
from torch import nn
from myutil import dij


class RieDictionaryLayer(nn.Module):
    def __init__(self, n_dictionary, n_dim, margin):
        super(RieDictionaryLayer, self).__init__()
        self.n_dictionary = n_dictionary
        self.n_dim = n_dim
        self.margin = margin

        self.dictionaries = nn.Parameter(torch.Tensor(n_dictionary, n_dim, n_dim))
        self.labels = nn.Parameter(torch.Tensor(n_dictionary), requires_grad=False)

    def allTripletLoss(self, dists, target):
        d = dij(target, self.labels)
        n = target.size(0)
        m = self.labels.size(0)
        all_dists = dists.unsqueeze(2).expand(n, m, m)
        all_dists = all_dists - all_dists.transpose(2, 1)
        d1 = d.unsqueeze(2).expand(n, m, m)
        d2 = d1.transpose(2, 1)
        d = d1 - d2
        # d[d < 0] = 0
        d = nn.functional.relu(d, True)
        loss = (all_dists + self.margin) * d
        # loss[loss < 0] = 0
        loss = nn.functional.relu(loss, True)
        loss = loss.sum() / (d.sum() + 1e-6)
        return loss

    def forward(self, dists, target):
        dl = self.allTripletLoss(dists, target)
        return dl
