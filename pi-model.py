from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import os
import argparse
from datasets.cifar10 import load_cifar_dataset
from datasets.mnist import load_mnist_dataset
from utils import accuracy, _disable_tracking_bn_stats
from models import SmallNet, LargeNet
import torch.nn.functional as F
import math


def cal_consistency_weight(epoch, init_ep=0, end_ep=300, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep) / float(end_ep - init_ep)
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w  # exp
    return weight_cl


def pi_smallnet_forward(net, input, device):
    input = input.contiguous().view(input.shape[0], -1)
    if net.training:
        delta = torch.randn_like(input) * 0.01
    else:
        delta = torch.zeros_like(input)
    input = input + delta
    for i in range(3):
        input = net.linear_layers[i](input)
        input = net.bn_layers[i](input)
        input = net.act_layers[i](input)
    # add dropout
    input = F.dropout(input=input, p=0.5, training=net.training)
    for i in range(3, 5):
        input = net.linear_layers[i](input)
        input = net.bn_layers[i](input)
        input = net.act_layers[i](input)
    return input


def train_epoch(model, labeled_train_loader, unlabeled_train_loader, optimizer, device, epoch, max_epoch):  # Training Process
    model.train()
    tot_loss, tot_accuracy = 0.0, 0.0
    times = 0
    unlabeled_iter = iter(unlabeled_train_loader) if unlabeled_train_loader is not None else None
    len_iter = len(iter(labeled_train_loader))
    for i, (input, target) in enumerate(labeled_train_loader):
        input = input.to(device)
        input_shape = input.shape
        target = target.to(device)
        optimizer.zero_grad()
        ul_input, _ = next(unlabeled_iter)
        ul_input = ul_input.to(device)
        input = torch.cat([input, ul_input], dim=0)
        logit = pi_smallnet_forward(model, input, device)
        with _disable_tracking_bn_stats(model):
            with torch.no_grad():
                logit1 = pi_smallnet_forward(model, input, device)
        diff = F.mse_loss(logit, logit1, reduction='mean') / 10
        weight = cal_consistency_weight(epoch * len_iter + i, end_ep=(max_epoch // 2) * len_iter, end_w=1.0)
        logit = logit[:input_shape[0]]
        loss = F.cross_entropy(logit, target)
        loss = loss + diff * weight * 10.0
        acc = accuracy(logit, target)
        loss.backward()
        optimizer.step()
        times += 1
        tot_loss += loss.cpu().data.numpy()
        tot_accuracy += acc.cpu().data.numpy()
    tot_loss /= times
    tot_accuracy /= times
    return tot_loss, tot_accuracy


@torch.no_grad()
def valid_epoch(model, valid_loader, device):  # Valid Process
    model.eval()
    tot_loss, tot_accuracy = 0.0, 0.0
    times = 0
    for i, (input, target) in enumerate(valid_loader):
        input = input.to(device)
        target = target.to(device)
        logit = pi_smallnet_forward(model, input, device)
        loss = F.cross_entropy(logit, target)
        acc = accuracy(logit, target)
        times += 1
        tot_loss += loss.cpu().data.numpy()
        tot_accuracy += acc.cpu().data.numpy()
    tot_loss /= times
    tot_accuracy /= times
    return tot_loss, tot_accuracy


@torch.no_grad()
def test_epoch(model, test_loader, device):  # Valid Process
    model.eval()
    tot_loss, tot_accuracy = 0.0, 0.0
    times = 0
    for i, (input, target) in enumerate(test_loader):
        input = input.to(device)
        target = target.to(device)
        logit = pi_smallnet_forward(model, input, device)
        loss = F.cross_entropy(logit, target)
        acc = accuracy(logit, target)
        times += 1
        tot_loss += loss.cpu().data.numpy()
        tot_accuracy += acc.cpu().data.numpy()
    tot_loss /= times
    tot_accuracy /= times
    return tot_loss, tot_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--label_num', default=1000, type=int)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    args = parser.parse_args()
    config = 'pi_batch-{}_epochs-{}_dataset-{}_labelnum-{}'.format(args.batch_size, args.epochs, args.dataset, args.label_num)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    device = torch.device('cuda')
    tb_writer = SummaryWriter(args.log_dir)
    if args.dataset == 'mnist':
        labeled_train_loader, unlabeled_train_loader, test_loader, valid_loader, labeled_len = load_mnist_dataset(args)
        model = SmallNet()
    elif args.dataset == 'cifar10':
        labeled_train_loader, unlabeled_train_loader, test_loader, valid_loader, labeled_len = load_cifar_dataset(args)
        model = LargeNet()
    else:
        print("Unsupported dataset.")
        exit(0)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    if args.do_train:
        best_val_acc = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch}")
            train_loss, train_acc = train_epoch(model, labeled_train_loader, unlabeled_train_loader, optimizer, device, epoch, args.epochs)
            val_loss, val_acc = valid_epoch(model, valid_loader, device)
            print(f"training loss: {train_loss}")
            print(f"training accuracy: {train_acc}")
            print(f"validation loss: {val_loss}")
            print(f"validation accuracy: {val_acc}")
            tb_writer.add_scalar("training loss", train_loss, global_step=epoch)
            tb_writer.add_scalar("training accuracy", train_acc, global_step=epoch)
            tb_writer.add_scalar("validation loss", val_loss, global_step=epoch)
            tb_writer.add_scalar("validation accuracy", val_acc, global_step=epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best.pth'))
            print(f"best epoch: {best_epoch}")
            print(f"best validation accuracy: {best_val_acc}")
    else:
        model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'best.pth')))
        test_loss, test_acc = test_epoch(model, test_loader, device)
        print(f"testing loss: {test_loss}")
        print(f"testing accuracy: {test_acc}")
