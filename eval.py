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
import torchvision
from utils import generate_adversarial_perturbation, generate_virtual_adversarial_perturbation, generate_mi_adv_target
from scalablebdl.mean_field import to_bayesian
from scalablebdl.bnn_utils import unfreeze, freeze

eps_plot = [0.1, 1, 5, 8, 20, 50, 100]
eps_curve = [10**(-1.0), 10**(-0.5), 10**0.0, 10**0.5, 10**1.0, 10**1.5, 10**2.0, 10**2.5, 10**3.0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--pretrain_config', default='base_batch-100_epochs-300_dataset-mnist_labelnum-0', type=str)
    parser.add_argument('--bayes', default=False, type=bool)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--method', default='vat', type=str, choices=["at", "vat", "mi"])
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--label_num', default=0, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()
    # args.ckpt_dir = os.path.join(args.ckpt_dir, args.pretrain_config)
    # args.log_dir = os.path.join(args.log_dir, args.pretrain_config)
    device = torch.device('cuda')
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
    if args.bayes:
        model = to_bayesian(model)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, os.path.join(args.pretrain_config, 'best.pth'))))
    model.eval()

    if args.plot:
        images = []
        for i, (input, target) in enumerate(test_loader):
            input = input.to(device)[:10]
            target = target.to(device)[:10]
            for j in range(7):
                eps = eps_plot[j]
                if args.method == 'at':
                    r_adv = generate_adversarial_perturbation(input, target, model, eps)
                elif args.method == 'vat':
                    with torch.no_grad():
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
        pass
