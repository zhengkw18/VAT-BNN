import torch
import os
import argparse
from utils import accuracy
from models import SmallNet, LargeNet
import torchvision
from utils import generate_adversarial_perturbation, generate_virtual_adversarial_perturbation, generate_mi_adv_target
import numpy as np
from bnn_utils import to_bayesian, freeze, unfreeze
from data_loader import fetch_dataloaders_MNIST, fetch_dataloaders_CIFAR10

eps_plot = [0.1, 1, 5, 8, 20, 50, 100]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--pretrain_config', default='base_batch-100_epochs-300_dataset-mnist_labelnum-0', type=str)
    parser.add_argument('--bayes', default=False, type=bool)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--ul_batch_size', default=256, type=int)
    parser.add_argument('--method', default='vat', type=str, choices=["at", "vat", "mi"])
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--label_num', default=0, type=int)
    args = parser.parse_args()
    # args.ckpt_dir = os.path.join(args.ckpt_dir, args.pretrain_config)
    # args.log_dir = os.path.join(args.log_dir, args.pretrain_config)
    if args.dataset == 'mnist':
        dataloaders = fetch_dataloaders_MNIST(args)
        model = SmallNet()
    elif args.dataset == 'cifar10':
        dataloaders = fetch_dataloaders_CIFAR10(args)
        model = LargeNet()
    else:
        print("Unsupported dataset.")
        exit(0)
    model.cuda()
    if args.bayes:
        model = to_bayesian(model)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, os.path.join(args.pretrain_config, 'best.pth'))))
    model.eval()
    if args.bayes and args.method != 'mi':
        freeze(model)
    elif not args.bayes and args.method == 'mi':
        model = to_bayesian(model)
    if args.plot:
        images = []
        for i, (input, target) in enumerate(dataloaders['test']):
            input = input.cuda()[:10]
            target = target.cuda()[:10]
            for j in range(7):
                eps = eps_plot[j]
                if args.method == 'at':
                    r_adv = generate_adversarial_perturbation(input, target, model, eps)
                elif args.method == 'vat':
                    logit = model(input)
                    r_adv = generate_virtual_adversarial_perturbation(input, logit, model, eps)
                else:
                    r_adv = generate_mi_adv_target(model, input, eps)
                images.append(input + r_adv)
            break
        images = torch.cat(images, dim=0)
        images = torchvision.utils.make_grid(images, nrow=10, padding=2, pad_value=255)
        torchvision.utils.save_image(images.cpu(), 'noise.png')

    else:
        accs = []
        for eps in np.arange(0, 5, 0.1):
            tot_accuracy = 0.0
            times = 0
            for i, (input, target) in enumerate(dataloaders['test']):
                input = input.cuda()
                target = target.cuda()
                if args.method == 'at':
                    r_adv = generate_adversarial_perturbation(input, target, model, eps)
                elif args.method == 'vat':
                    logit = model(input)
                    r_adv = generate_virtual_adversarial_perturbation(input, logit, model, eps)
                else:
                    r_adv = generate_mi_adv_target(model, input, eps)
                freeze(model)
                with torch.no_grad():
                    logit = model(input + r_adv)
                if args.method == 'mi':
                    unfreeze(model)
                acc = accuracy(logit, target)
                times += 1
                tot_accuracy += acc.cpu().data.numpy()
            tot_accuracy /= times
            print(f'eps: {eps}, acc: {tot_accuracy}')
            accs.append(tot_accuracy)
        mis_acc = [accs[0] - accs[i] for i in range(1, len(accs))]
        print(mis_acc)
