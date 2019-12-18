import torch
import torch.optim as optim
import numpy as np
import time
from myutil import AverageMeter
import math
from tqdm import tqdm
from model import BMS_Net
import argparse
from pprint import pprint
from myutil import group_list
from myutil import bcolors
from torch import nn
from BiMap import BiMap
from ReEig import ReEigFunction
from LogEig import LogEigFunction

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="data/afew_dataset.npz", help="path to dataset"
)
parser.add_argument("--ep", type=float, default=1e-4, help="epsilon for ReEig layer")
parser.add_argument("--batch_size", type=int, default=30, help="batch size")
parser.add_argument("--n_atom", type=int, default=28, help="number of dictionary atom")
parser.add_argument("--margin1", type=float, default=1, help="margin for triplet loss")
parser.add_argument("--margin2", type=float, default=1, help="margin for intra loss")
parser.add_argument(
    "--dims",
    type=str,
    default="400,200,100,50",
    help="dimensionality for extracting feature",
)
parser.add_argument("--n_class", type=int, default=7, help="number of class")
parser.add_argument(
    "--lambda1", type=float, default=1.2, help="trade-off coefficient for triplet loss"
)
parser.add_argument(
    "--lambda2", type=float, default=0.7, help="trade-off coefficient for intra loss"
)
parser.add_argument(
    "--save_folder",
    type=str,
    default="./models/afew_logw.pkl",
    help="path to save model",
)
parser.add_argument(
    "--use_tensorboard", type=bool, default=False, help="whether to use tensorboard"
)
parser.add_argument(
    "--metric_method",
    type=str,
    default="log_w",
    help="method for feature metric, log, log_w or jbld",
)
parser.add_argument(
    "--log_dim", type=int, default=20, help="dimensionality for log metric"
)
parser.add_argument("--n_fc", type=int, default=1, help="number of fc layers")
parser.add_argument(
    "--n_fc_node", type=int, default=4096, help="number of nodes in fc layer"
)

args = parser.parse_args()
pprint(args)

best_correct = 0
best_epoch = 0

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)

dataset = np.load(args.dataset)
train_x = dataset["trX"]
train_y = dataset["trY"].astype(np.long)
test_x = dataset["teX"]
test_y = dataset["teY"].astype(np.long)


class SPDNet(nn.Module):

    """Docstring for SPDNet. """

    def __init__(self):
        super(SPDNet, self).__init__()
        self.bimap1 = BiMap(400, 200)
        self.bimap2 = BiMap(200, 100)
        self.bimap3 = BiMap(100, 50)
        self.linear = nn.Linear(2500, 7)

    def forward(self, x):
        x = self.bimap1(x)
        x = ReEigFunction.apply(x, 1e-4)
        x = self.bimap2(x)
        x = ReEigFunction.apply(x, 1e-4)
        x = self.bimap3(x)
        x = LogEigFunction.apply(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


model = SPDNet()

optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=5e-4, momentum=0.0)

loss_function = torch.nn.CrossEntropyLoss()
n_trained_batch = 0


def train(epoch, optimizer):
    global n_trained_batch
    global train_x
    global train_y

    meter_acc = AverageMeter()
    meter_loss_total = AverageMeter()
    meter_loss_cls = AverageMeter() meter_loss_triplet = AverageMeter() meter_loss_intra = AverageMeter()

    model.train()
    batch_idx = 0
    index = np.random.permutation(len(train_x))
    train_x = train_x[index]
    train_y = train_y[index]
    for (data, target) in tqdm(
        group_list(train_x, train_y, args.batch_size),
        total=len(train_x) // args.batch_size,
    ):
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        n_data = data.size(0)

        output = model(data)

        triplet_loss = 0
        intra_loss = 0
        classifier_loss = loss_function(output, target)

        total_loss = (
            classifier_loss + triplet_loss * args.lambda1 - intra_loss * args.lambda2
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        for layer in model.named_children():
            if layer[0].startswith("bimap"):
                q, r = torch.qr(layer[1].weight.data.t())
                layer[1].weight.data = (
                    q @ torch.diag(torch.sign(torch.sign(torch.diag(r)) + 0.5))
                ).t()

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()

        total_loss = total_loss.item()
        classifier_loss = classifier_loss.item()

        meter_acc.update(correct / n_data, n_data)
        meter_loss_total.update(total_loss, n_data)
        meter_loss_cls.update(classifier_loss, n_data)
        meter_loss_triplet.update(triplet_loss, n_data)
        meter_loss_intra.update(intra_loss, n_data)

        if batch_idx % (len(train_x) // args.batch_size // 5) == 0:
            pstr = (
                f"Epoch:{epoch:2} Batch_idx:{batch_idx:2} "
                f"Loss:{bcolors.OKGREEN}{total_loss:.2f}{bcolors.ENDC}"
                f"({classifier_loss:.2f}/{triplet_loss:.2f}/{intra_loss:.2f})\t"
                f"Acc:{bcolors.OKGREEN}{correct/n_data*100:.2f}{bcolors.ENDC}\n"
                f"Average: Loss:{meter_loss_total.avg:.2f}"
                f"({meter_loss_cls.avg:.2f}/{meter_loss_triplet.avg:.2f}/{meter_loss_intra.avg:.2f})"
                f"\tTime:{time.ctime()}"
                f"\tAcc:{bcolors.OKGREEN}{meter_acc.avg*100:.2f}{bcolors.ENDC}"
            )
            tqdm.write(pstr)
        batch_idx += 1
        n_trained_batch += 1


def test(epoch):
    global best_correct
    global best_epoch
    global test_x
    global test_y

    meter_acc = AverageMeter()
    meter_loss_total = AverageMeter()
    meter_loss_cls = AverageMeter()
    meter_loss_triplet = AverageMeter()
    meter_loss_intra = AverageMeter()

    model.eval()
    correct = 0
    for data, target in tqdm(
        group_list(test_x, test_y, args.batch_size),
        total=len(test_x) // args.batch_size,
    ):
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        n_data = data.size(0)

        output = model(data)

        triplet_loss = 0
        intra_loss = 0
        classifier_loss = loss_function(output, target)

        total_loss = (
            classifier_loss + triplet_loss * args.lambda1 - intra_loss * args.lambda2
        )

        total_loss = total_loss.item()
        classifier_loss = classifier_loss.item()

        pred = output.data.max(1, keepdim=True)[1]
        current_correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()
        correct += current_correct

        meter_acc.update(current_correct / n_data, n_data)
        meter_loss_total.update(total_loss, n_data)
        meter_loss_cls.update(classifier_loss, n_data)
        meter_loss_triplet.update(triplet_loss, n_data)
        meter_loss_intra.update(intra_loss, n_data)

    if correct > best_correct:
        best_correct = correct
        best_epoch = epoch
        state = {
            "acc": correct / test_x.shape[0],
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "args": args,
        }
        # torch.save(state, args.save_folder)
    print(
        f"Epoch:{epoch:2} "
        f"Loss:{meter_loss_total.avg:.2f}"
        f"({meter_loss_cls.avg:.2f}/{meter_loss_triplet.avg:.2f}/{meter_loss_intra.avg:.2f})"
        f"\tAcc:{meter_acc.avg*100:.2f}\tTime:{time.ctime()}"
    )
    print(f"Best epoch:{best_epoch} Accuracy:{best_correct/len(test_x)*100:.2f}\n")
    print("=" * 20)
    return correct


if __name__ == "__main__":
    for epoch in range(1, 601):
        train(epoch, optimizer)
        with torch.no_grad():
            correct = test(epoch)
