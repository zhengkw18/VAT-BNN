from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import os
import argparse
from datasets.cifar10_ssl import load_cifar_dataset
from datasets.mnist_ssl import load_mnist_dataset
from utils import accuracy
from models import SmallNet, LargeNet
import torch.nn.functional as F
from scalablebdl.mean_field import to_bayesian, PsiSGD
from utils import mi_adversarial_loss
from scalablebdl.bnn_utils import unfreeze


def train_ssl_epoch(model, labeled_train_loader, unlabeled_train_loader, mi_optimizer, psi_optimizer, device, epsilon, adv_target=False):
    model.train()
    tot_loss, tot_accuracy, tot_mi = 0.0, 0.0, 0.0
    times = 0
    labeled_iter = iter(labeled_train_loader)
    for i, (ul_input, _) in enumerate(unlabeled_train_loader):
        try:
            (input, target) = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_train_loader)
            (input, target) = next(labeled_iter)
        input_shape = input.shape
        input = torch.cat([input, ul_input], dim=0)
        input = input.to(device)
        target = target.to(device)
        mi_optimizer.zero_grad()
        psi_optimizer.zero_grad()
        logit = model(input)
        mi_loss = mi_adversarial_loss(model, input, epsilon, adv_target)
        logit = logit[:input_shape[0]]
        loss = F.cross_entropy(logit, target)
        acc = accuracy(logit, target)
        loss += mi_loss
        loss.backward()
        mi_optimizer.step()
        psi_optimizer.step()

        times += 1
        tot_loss += loss.cpu().data.numpy()
        tot_accuracy += acc.cpu().data.numpy()
        tot_mi += mi_loss.cpu().data.numpy()
    tot_loss /= times
    tot_accuracy /= times
    tot_mi /= times
    return tot_loss, tot_accuracy, tot_mi


def train_epoch(model, labeled_train_loader, mi_optimizer, psi_optimizer, device, epsilon, adv_target=False):  # Training Process
    model.train()
    tot_loss, tot_accuracy, tot_mi = 0.0, 0.0, 0.0
    times = 0
    for i, (input, target) in enumerate(labeled_train_loader):
        input = input.to(device)
        target = target.to(device)
        mi_optimizer.zero_grad()
        psi_optimizer.zero_grad()
        logit = model(input)

        mi_loss = mi_adversarial_loss(model, input, epsilon, adv_target)

        loss = F.cross_entropy(logit, target)
        acc = accuracy(logit, target)
        loss += mi_loss
        loss.backward()
        mi_optimizer.step()
        psi_optimizer.step()

        times += 1
        tot_loss += loss.cpu().data.numpy()
        tot_accuracy += acc.cpu().data.numpy()
        tot_mi += mi_loss.cpu().data.numpy()
    tot_loss /= times
    tot_accuracy /= times
    tot_mi /= times
    return tot_loss, tot_accuracy, tot_mi


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
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--strategy', type=str, choices=["miadv_train", "mipred"], default="mipred")
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--label_num', default=0, type=int)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--epsilon', type=float, default=2.0)
    args = parser.parse_args()
    config = f"{args.strategy}_batch-{args.batch_size}_epochs-{args.epochs}_dataset-{args.dataset}_labelnum-{args.label_num}_epsilon-{args.epsilon}"
    if args.pretrained:
        config = config + "_pretrained"
    base_config = 'basebnn_batch-{}_epochs-{}_dataset-{}_labelnum-{}'.format(args.batch_size, args.epochs, args.dataset, args.label_num)
    base_ckpt_dir = os.path.join(args.ckpt_dir, base_config)
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
    model = to_bayesian(model)
    unfreeze(model)
    model.to(device)

    if args.pretrained:
        model.load_state_dict(torch.load(os.path.join(base_ckpt_dir, 'best.pth')))
        test_loss, test_acc = test_epoch(model, test_loader, device)
        print(f"pretrained model testing loss: {test_loss}")
        print(f"pretrained model testing accuracy: {test_acc}")

    mus, psis = [], []
    for name, param in model.named_parameters():
        if 'psi' in name:
            psis.append(param)
        else:
            mus.append(param)
    mu_optimizer = optim.Adam(mus, lr=args.learning_rate)
    psi_optimizer = PsiSGD(psis, lr=args.learning_rate, momentum=0.9,
                           weight_decay=2e-4, nesterov=True,
                           num_data=len(labeled_train_loader))

    if args.do_train:
        best_val_acc = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            print(f"Epoch: {epoch}")
            if args.label_num == 0:
                train_loss, train_acc, train_mi = train_epoch(model, labeled_train_loader, mu_optimizer, psi_optimizer, device, args.epsilon, (args.strategy == "miadv_train"))
            else:
                train_loss, train_acc, train_mi = train_ssl_epoch(model, labeled_train_loader, unlabeled_train_loader, mu_optimizer,
                                                                  psi_optimizer, device, args.epsilon, (args.strategy == "miadv_train"))
            val_loss, val_acc = valid_epoch(model, valid_loader, device)
            print(f"training loss: {train_loss}")
            print(f"training accuracy: {train_acc}")
            print(f"training mutual information: {train_mi}")
            print(f"validation loss: {val_loss}")
            print(f"validation accuracy: {val_acc}")
            tb_writer.add_scalar("training loss", train_loss, global_step=epoch)
            tb_writer.add_scalar("training accuracy", train_acc, global_step=epoch)
            tb_writer.add_scalar("training mutual information", train_mi, global_step=epoch)
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
