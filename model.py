import torch
import torch.nn as nn
from BiMap import BiMap
from ReEig import ReEig
from RieDictionaryLayer import RieDictionaryLayer
from DisNet import Log_DisNet, JBLD_DisNet, Log_w_DisNet
from myutil import dij, check_gpu
from torch.autograd import Variable
import time
from myutil import one_hot


class Timer:
    def __init__(self):
        self.total_time = 0
        self.current_time = 0

    def tic(self):
        self.current_time = time.time()

    def toc(self):
        self.total_time += time.time() - self.current_time


class BMS_Net(nn.Module):
    def __init__(self, args):
        super(BMS_Net, self).__init__()
        self.ep = args.ep
        dims = [int(i) for i in args.dims.split(",")]
        self.feature = []
        for i in range(len(dims) - 2):
            self.feature.append(BiMap(dims[i], dims[i + 1]))
            self.feature.append(ReEig(self.ep))
        self.feature.append(BiMap(dims[-2], dims[-1]))
        self.feature = nn.Sequential(*self.feature)
        self.dictLayer = RieDictionaryLayer(args.n_atom, dims[-1], args.margin1)
        if args.metric_method == "log":
            self.distFun = Log_DisNet(dims[-1] ** 2)
        elif args.metric_method == "log_w":
            self.distFun = Log_w_DisNet(dims[-1] ** 2, args.log_dim)
        elif args.metric_method == "jbld":
            self.distFun = JBLD_DisNet()
        else:
            raise NotImplementedError
        self.margin2 = args.margin2
        self.n_class = args.n_class
        self.use_intra_loss = args.lambda2 != 0
        self.use_triplet_loss = args.lambda1 != 0

        classifier = []
        if args.n_fc == 1:
            classifier.append(nn.Linear(args.n_atom, args.n_class, bias=None))
        else:
            classifier.append(nn.Linear(args.n_atom, args.n_fc_node, bias=None))
            for i in range(args.n_fc - 2):
                classifier.append(nn.ReLU(True))
                classifier.append(nn.Linear(args.n_fc_node, args.n_fc_node, bias=None))
            classifier.append(nn.ReLU(True))
            classifier.append(nn.Linear(args.n_fc_node, args.n_class, bias=None))

        self.classifier = nn.Sequential(*classifier)
        if check_gpu():
            self.classifier = self.classifier.cuda()

    def encoding(self, x):
        v = self.feature(x)
        return v

    def calc_distance_between_codebooks(self, v, dij=None):
        d = self.dictLayer.dictionaries.transpose(2, 1) @ self.dictLayer.dictionaries
        dists = self.distFun(d, v, dij).cpu()
        if dij is None:
            dists = dists.view(v.size(0), self.dictLayer.labels.size(0))
        return dists

    def calc_triplet_loss(self, dists, y):
        return self.dictLayer(dists, y)

    def triplet_loss(self, dist, y):
        if self.use_triplet_loss:
            triplet_loss = self.calc_triplet_loss(dist, y)
        else:
            triplet_loss = Variable(torch.Tensor([0]))
        return triplet_loss

    def intra_loss(self):
        if self.use_intra_loss:
            d = (
                self.dictLayer.dictionaries.transpose(2, 1)
                @ self.dictLayer.dictionaries
            )
            _dij = dij(self.dictLayer.labels, self.dictLayer.labels)
            dic_dists = self.calc_distance_between_codebooks(d, _dij)
            # dic_dists[dic_dists > self.margin2] = self.margin2
            dic_dists = self.margin2 - dic_dists
            dic_dists = nn.functional.relu(dic_dists, True)
            dic_dists = self.margin2 - dic_dists
            intra_loss = dic_dists.mean()
        else:
            intra_loss = Variable(torch.Tensor([0]))
        return intra_loss

    def forward(self, x):
        feature = self.encoding(x)
        dist = self.calc_distance_between_codebooks(feature)
        if check_gpu():
            dist = dist.cuda()
        # classifier_output = self.classifier(dist).cpu()

        dist2 = dist * -5
        dist2 = nn.functional.softmax(dist2, dim=1)
        one_hot_labels = one_hot(self.n_class, self.dictLayer.labels).float().cuda()
        classifier_output = dist2 @ one_hot_labels
        return classifier_output.cpu(), dist.cpu()
