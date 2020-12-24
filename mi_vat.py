from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
import os
import argparse
from utils import accuracy, _disable_tracking_bn_stats, get_normalized_vector, kl_divergence_with_logit
from models import SmallNet, LargeNet
import torch.nn.functional as F
from scalablebdl.mean_field import to_bayesian, PsiSGD
from scalablebdl.bnn_utils import freeze, unfreeze
from utils import mutual_information, generate_mi_adv_target


def mi_adversarial_loss(model, input, logit, epsilon, adv_target=False):
    kl_loss, mi_loss = 0.0, 0.0
    if adv_target:
        r_adv = generate_mi_adv_target(model, input, epsilon)
    else:
        r_adv = torch.zeros_like(input)
    with _disable_tracking_bn_stats(model):
        logit_q = model(input + r_adv)
        if adv_target:
            logit_p = model(input + r_adv)
            mi_loss = mutual_information(logit_p, logit_q)
            kl_loss = kl_divergence_with_logit(logit, logit_q)
        else:
            mi_loss = mutual_information(logit, logit_q)
            kl_loss = torch.zeros_like(mi_loss)
    return mi_loss, kl_loss


def train_ssl_epoch(model, labeled_train_loader, unlabeled_trainset, mi_optimizer, psi_optimizer, device, epsilon, alpha, adv_target=False):
    model.train()
    tot_loss, tot_accuracy, tot_mi, tot_kl = 0.0, 0.0, 0.0, 0.0
    times = 0
    unlabeled_iter = unlabeled_trainingset["iter"]
    for i, (input, target) in enumerate(labeled_train_loader):
        try:
            (ul_input, _) = next(unlabeled_iter)
        except StopIteration:
            unlabeled_train_loader = unlabeled_trainingset["dataloader"]
            unlabeled_iter = iter(unlabeled_train_loader)
            unlabeled_trainingset["iter"] = unlabeled_iter
            (ul_input, _) = next(unlabeled_iter)
        input_shape = input.shape
        input = torch.cat([input, ul_input], dim=0)
        input = input.to(device)
        target = target.to(device)
        mi_optimizer.zero_grad()
        psi_optimizer.zero_grad()
        logit = model(input)
        mi_loss, kl_loss = mi_adversarial_loss(model, input, logit, epsilon, adv_target)
        loss = F.cross_entropy(logit[:input_shape[0]], target)

        tot_loss += loss.cpu().data.numpy()

        acc = accuracy(logit[:input_shape[0]], target)
        loss += mi_loss * alpha + kl_loss
        loss.backward()
        mi_optimizer.step()
        psi_optimizer.step()

        times += 1
        tot_kl += kl_loss.cpu().data.numpy()
        tot_accuracy += acc.cpu().data.numpy()
        tot_mi += mi_loss.cpu().data.numpy()
    tot_loss /= times
    tot_accuracy /= times
    tot_mi /= times
    tot_kl /= times
    return tot_loss, tot_accuracy, tot_mi, tot_kl


def train_epoch(model, labeled_train_loader, mu_optimizer, psi_optimizer, device, epsilon, alpha=1, adv_target=False):  # Training Process
    model.train()
    tot_loss, tot_accuracy, tot_mi = 0.0, 0.0, 0.0
    times = 0
    for i, (input, target) in enumerate(labeled_train_loader):
        input = input.to(device)
        target = target.to(device)
        mu_optimizer.zero_grad()
        psi_optimizer.zero_grad()
        logit = model(input)
        mi_loss, kl_loss = mi_adversarial_loss(model, input, logit, epsilon, adv_target)
        loss = F.cross_entropy(logit, target)

        tot_loss += loss.cpu().data.numpy()

        acc = accuracy(logit, target)
        loss += mi_loss * alpha + kl_loss
        loss.backward()
        mu_optimizer.step()
        psi_optimizer.step()

        times += 1
        tot_kl += kl_loss.cpu().data.numpy()
        tot_accuracy += acc.cpu().data.numpy()
        tot_mi += mi_loss.cpu().data.numpy()
    tot_loss /= times
    tot_accuracy /= times
    tot_mi /= times
    tot_kl /= times
    return tot_loss, tot_accuracy, tot_mi, tot_kl


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


def weight_scheme(epoch, max_epoch, max_alpha, mult):
    return max_alpha * np.exp(mult * (1 - float(epoch) / max_epoch)) + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--strategy', type=str, choices=["mivat_train", "mipred"], default="mipred")
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--label_num', default=0, type=int)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--epsilon', type=float, default=2.0)
    parser.add_argument('--max_alpha', type=float, default=2.0)
    parser.add_argument('--mult', type=float, default=-2.0)
    parser.add_argument('--pretrained_config', default='', type=str)
    args = parser.parse_args()
    print(args)
    config = f"{args.strategy}_batch-{args.batch_size}_epochs-{args.epochs}_dataset-{args.dataset}_labelnum-{args.label_num}_epsilon-{args.epsilon}_max_alpha-{args.max_alpha}_mult-{args.mult}"
    if args.pretrained:
        config = config + "_pretrained"
    base_config = 'base_batch-{}_epochs-{}_dataset-{}_labelnum-{}'.format(args.batch_size, args.epochs, args.dataset, args.label_num)
    if args.pretrained_config != '':
        print("have direct pretrained configure")
        base_config = args.pretrained_config

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
    model.to(device)
    if args.pretrained:
        model.load_state_dict(torch.load(os.path.join(base_ckpt_dir, 'best.pth')))
        test_loss, test_acc = test_epoch(model, test_loader, device)
        print(f"pretrained model testing loss: {test_loss}")
        print(f"pretrained model testing accuracy: {test_acc}")
    model = to_bayesian(model)
    unfreeze(model)
    mus, psis = [], []
    for name, param in model.named_parameters():
        if 'psi' in name:
            psis.append(param)
        else:
            mus.append(param)
    mu_optimizer = optim.Adam(mus, lr=args.learning_rate)
    psi_optimizer = PsiSGD(psis, lr=args.learning_rate, momentum=0.9, weight_decay=2e-4, nesterov=True, num_data=len(labeled_train_loader))

    if args.do_train:
        best_val_acc = 0
        best_epoch = 0
        patience = 10 if args.strategy == "mipred" else 15

        mu_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mu_optimizer, mode='min', factor=0.9, patience=patience, verbose=True)
        psi_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(psi_optimizer, mode='min', factor=0.9, patience=patience, verbose=True)
        print(f"Total train data size: {len(labeled_train_loader)}")
        if args.label_num != 0:
            print(f"Total unlabeled train data size: {len(unlabeled_train_loader)}")
            unlabeled_iter = iter(unlabeled_train_loader)
            unlabeled_trainingset = {"iter": unlabeled_iter, "dataloader": unlabeled_train_loader}
        for epoch in range(args.epochs):
            alpha = weight_scheme(epoch, args.epochs, args.max_alpha, args.mult)
            print(f"Epoch: {epoch}, corresponding alpha: {alpha}")
            if args.label_num == 0:
                train_loss, train_acc, train_mi, train_kl = train_epoch(model, labeled_train_loader, mu_optimizer, psi_optimizer, device, args.epsilon, alpha, (args.strategy == "mivat_train"))
            else:
                train_loss, train_acc, train_mi, train_kl = train_ssl_epoch(model, labeled_train_loader, unlabeled_trainingset, mu_optimizer,
                                                                            psi_optimizer, device, args.epsilon, alpha, (args.strategy == "mivat_train"))
            val_loss, val_acc = valid_epoch(model, valid_loader, device)
            mu_scheduler.step(train_mi)
            psi_scheduler.step(train_mi)
            print(f"training loss: {train_loss}")
            print(f"training accuracy: {train_acc}")
            print(f"training mutual information: {train_mi}")
            print(f"training kl divergence {train_kl}")
            print(f"validation loss: {val_loss}")
            print(f"validation accuracy: {val_acc}")
            tb_writer.add_scalar("training loss", train_loss, global_step=epoch)
            tb_writer.add_scalar("training accuracy", train_acc, global_step=epoch)
            tb_writer.add_scalar("training mutual information", train_mi, global_step=epoch)
            tb_writer.add_scalar("training kl divergence", train_kl, global_step=epoch)
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
