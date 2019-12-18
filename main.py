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
from tensorboardX import SummaryWriter
from ReEig import ReEigFunction

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="data/afew_dataset.npz", help="path to dataset"
)
parser.add_argument("--ep", type=float, default=1e-4, help="epsilon for ReEig layer")
parser.add_argument("--batch_size", type=int, default=30, help="batch size")
parser.add_argument("--n_atom", type=int, default=28, help="number of dictionary atom")
parser.add_argument("--margin1", type=float, default=2, help="margin for triplet loss")
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
    default="log",
    help="method for feature metric, log, log_w or jbld",
)
parser.add_argument(
    "--log_dim", type=int, default=30, help="dimensionality for log metric"
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

writer = SummaryWriter()


def initialize(y, num):
    n = y.size()[0]
    t, _ = np.unique(y.numpy(), return_inverse=True)
    t = len(t)
    ratio = float(num) / n
    res = None
    for i in range(t):
        indices = torch.nonzero(torch.eq(y, i))
        if len(indices.size()) == 0:
            continue
        c = math.ceil(ratio * len(indices))
        indices = indices[torch.randperm(indices.size()[0])]
        if res is None:
            res = indices[:c]
        else:
            res = torch.cat((res, indices[:c]), 0)
    res = res[torch.randperm(res.size()[0])]
    return res.view(res.size(0))


# initialize dictionary
y = torch.from_numpy(train_y)
indices = initialize(y, args.n_atom)
args.n_atom = indices.size(0)
y = y[indices]
model = BMS_Net(args)
model.dictLayer.labels.data = y

# fine-tune from the spdnet
# weights = torch.load("./afew.pkl")
# model.feature[0].weight.data = weights["bimap0.weight"].float()
# model.feature[2].weight.data = weights["bimap1.weight"].float()
# model.feature[4].weight.data = weights["bimap2.weight"].float()

for i in range(indices.shape[0]):
    mats = train_x[indices[i]]
    mats = torch.from_numpy(mats).unsqueeze(0)
    mats = model.encoding(mats)
    mats = torch.from_numpy(np.linalg.cholesky(mats.data[0].numpy()).transpose())
    model.dictLayer.dictionaries.data[i] = mats
print(model)

optimizer = optim.SGD(
    [
        {"params": model.feature.parameters(), "lr": 1e-3},
        {"params": model.distFun.parameters(), "lr": 1e-3},
        {
            "params": filter(lambda p: p.requires_grad, model.dictLayer.parameters()),
            "lr": 1e-3,
        },
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ],
    lr=1e-3,
    weight_decay=5e-4,
    momentum=0.9,
)

loss_function = torch.nn.CrossEntropyLoss()
n_trained_batch = 0


def train(epoch, optimizer):
    global n_trained_batch
    global train_x
    global train_y

    meter_acc = AverageMeter()
    meter_loss_total = AverageMeter()
    meter_loss_cls = AverageMeter()
    meter_loss_triplet = AverageMeter()
    meter_loss_intra = AverageMeter()

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

        output, dist = model(data)

        triplet_loss = model.triplet_loss(dist, target)
        intra_loss = model.intra_loss()
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
        with torch.no_grad():
            for i in range(model.dictLayer.dictionaries.size(0)):
                model.dictLayer.dictionaries.data[i].triu_()
            m = ReEigFunction.apply(
                model.dictLayer.dictionaries.data.transpose(2, 1)
                @ model.dictLayer.dictionaries.data,
                args.ep,
            )
            for i in range(model.dictLayer.dictionaries.size(0)):
                model.dictLayer.dictionaries.data[i] = torch.from_numpy(
                    np.linalg.cholesky(m[i].numpy()).transpose()
                )

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()

        total_loss = total_loss.item()
        classifier_loss = classifier_loss.item()
        triplet_loss = triplet_loss.item()
        intra_loss = intra_loss.item()

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
            # tqdm.write(pstr)
        batch_idx += 1
        n_trained_batch += 1
    print(
        f"Epoch:{epoch:2} "
        f"Loss:{meter_loss_total.avg:.2f}"
        f"({meter_loss_cls.avg:.2f}/{meter_loss_triplet.avg:.2f}/{meter_loss_intra.avg:.2f})"
        f"\tAcc:{meter_acc.avg*100:.2f}\tTime:{time.ctime()}"
    )
    writer.add_scalar("train/Loss", meter_loss_total.avg, epoch)
    writer.add_scalar("train/Acc", meter_acc.avg * 100, epoch)
    writer.add_scalar("train/Cls_Loss", meter_loss_cls.avg, epoch)
    writer.add_scalar("train/Triplet_Loss", meter_loss_triplet.avg, epoch)
    writer.add_scalar("train/Intra_Loss", meter_loss_intra.avg, epoch)
    writer.file_writer.flush()


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

        output, dist = model(data)

        triplet_loss = model.triplet_loss(dist, target)
        intra_loss = model.intra_loss()
        classifier_loss = loss_function(output, target)

        total_loss = (
            classifier_loss + triplet_loss * args.lambda1 - intra_loss * args.lambda2
        )

        total_loss = total_loss.item()
        classifier_loss = classifier_loss.item()
        triplet_loss = triplet_loss.item()
        intra_loss = intra_loss.item()

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
    writer.add_scalar("test/Loss", meter_loss_total.avg, epoch)
    writer.add_scalar("test/Acc", meter_acc.avg * 100, epoch)
    writer.add_scalar("test/Cls_Loss", meter_loss_cls.avg, epoch)
    writer.add_scalar("test/Triplet_Loss", meter_loss_triplet.avg, epoch)
    writer.add_scalar("test/Intra_Loss", meter_loss_intra.avg, epoch)
    writer.file_writer.flush()
    return correct


if __name__ == "__main__":
    for epoch in range(1, 601):
        train(epoch, optimizer)
        with torch.no_grad():
            correct = test(epoch)
    print(f"Best epoch:{best_epoch} Accuracy:{best_correct/len(test_x)*100:.2f}\n")
