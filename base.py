from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import os
import argparse
from datasets.cifar10 import load_cifar_dataset
from datasets.mnist import load_mnist_dataset
from utils import accuracy
from models import SmallNet, LargeNet
import torch.nn.functional as F


def train_epoch(model, labeled_train_loader, optimizer, device):  # Training Process
    model.train()
    tot_loss, tot_accuracy = 0.0, 0.0
    times = 0
    for i, (input, target) in enumerate(labeled_train_loader):
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        logit = model(input)
        loss = F.cross_entropy(logit, target)
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
        logit = model(input)
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
        logit = model(input)
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
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--label_num', default=0, type=int)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    args = parser.parse_args()
    config = 'base_batch-{}_epochs-{}_dataset-{}_labelnum-{}'.format(args.batch_size, args.epochs, args.dataset, args.label_num)
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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.do_train:
        best_val_acc = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch}")
            train_loss, train_acc = train_epoch(model, labeled_train_loader, optimizer, device)
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
