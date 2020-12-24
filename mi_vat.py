from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import os
import argparse
from utils import accuracy
from models import SmallNet, LargeNet
import torch.nn.functional as F
from scalablebdl.mean_field import to_bayesian, PsiSGD
from scalablebdl.bnn_utils import freeze, unfreeze
from utils import mi_adversarial_loss
from tqdm import tqdm
from data_loader import fetch_dataloaders_MNIST, fetch_dataloaders_CIFAR10


def train_single_iter(model, dl_label, dl_unlabel, mu_optimizer, psi_optimizer, epsilon, alpha, adv_target=False):
    model.train()

    label_X, label_y = dl_label.__iter__().next()
    unlabel_X, _ = dl_unlabel.__iter__().next()
    label_X, label_y = label_X.cuda(), label_y.cuda()
    unlabel_X = unlabel_X.cuda()

    label_logit = model(label_X)
    unlabel_logit = model(unlabel_X)
    ce = F.cross_entropy(label_logit, label_y)
    mi_loss, kl_loss = mi_adversarial_loss(model, unlabel_X, unlabel_logit, epsilon, adv_target)
    mi_loss *= alpha
    loss = ce + mi_loss + kl_loss
    mu_optimizer.zero_grad()
    psi_optimizer.zero_grad()
    loss.backward()
    mu_optimizer.step()
    psi_optimizer.step()

    return ce.item(), mi_loss.item(), kl_loss.item()


@torch.no_grad()
def eval_epoch(model, data_loader):  # Valid Process
    model.eval()
    freeze(model)
    tot_loss, tot_accuracy = 0.0, 0.0
    times = 0
    for i, (input, target) in enumerate(data_loader):
        input = input.cuda()
        target = target.cuda()
        logit = model(input)
        loss = F.cross_entropy(logit, target)
        acc = accuracy(logit, target)
        times += input.size(0)
        tot_loss += loss.cpu().data.numpy() * input.size(0)
        tot_accuracy += acc.cpu().data.numpy() * input.size(0)
    tot_loss /= times
    tot_accuracy /= times
    unfreeze(model)
    return tot_loss, tot_accuracy


def train_and_evaluate(args, model, mu_optimizer, psi_optimizer, dataloaders):
    dl_label = dataloaders['label']
    dl_unlabel = dataloaders['unlabel']
    dl_val = dataloaders['val']

    # training steps
    best_val_acc = 0.0
    best_step = 0
    for step in tqdm(range(args.steps)):
        ce, mi_loss, kl_loss = train_single_iter(model, dl_label, dl_unlabel, mu_optimizer, psi_optimizer, args.epsilon, args.alpha, (args.strategy == "mivat_train"))
        if (step + 1) % (args.steps / 1000) == 0:
            tb_writer.add_scalar("training ce loss", ce, global_step=step)
            tb_writer.add_scalar("training mi loss", mi_loss, global_step=step)
            tb_writer.add_scalar("training kl loss", kl_loss, global_step=step)
        if (step + 1) % (args.steps / 100) == 0:
            val_loss, val_acc = eval_epoch(model, dl_val)
            tb_writer.add_scalar("validation loss", val_loss, global_step=step)
            tb_writer.add_scalar("validation accuracy", val_acc, global_step=step)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_step = step
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best.pth'))
            print(f"training ce loss: {ce}")
            print(f"training mi loss: {mi_loss}")
            print(f"training kl loss: {kl_loss}")
            print(f"validation accuracy: {val_acc}")
            print(f"best step: {best_step}")
            print(f"best validation accuracy: {best_val_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--strategy', type=str, choices=["mivat_train", "mipred"], default="mivat_train")
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--ul_batch_size', default=256, type=int)
    parser.add_argument('--steps', default=100000, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--label_num', default=0, type=int)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--epsilon', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--pretrained_config', default='', type=str)
    args = parser.parse_args()
    config = f"{args.strategy}_batch-{args.batch_size}_dataset-{args.dataset}_labelnum-{args.label_num}_epsilon-{args.epsilon}_alpha-{args.alpha}"
    if args.pretrained_config != '':
        config = config + "_pretrained"
        print("have direct pretrained configure")
        base_config = args.pretrained_config
        base_ckpt_dir = os.path.join(args.ckpt_dir, base_config)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    tb_writer = SummaryWriter(args.log_dir)
    if args.dataset == 'mnist':
        dataloaders = fetch_dataloaders_MNIST(args)
        model = SmallNet()
    elif args.dataset == 'cifar10':
        dataloaders = fetch_dataloaders_CIFAR10(args)
        model = LargeNet()
    else:
        print("Unsupported dataset.")
        exit(0)
    model = model.cuda()
    if args.pretrained_config != '':
        model.load_state_dict(torch.load(os.path.join(base_ckpt_dir, 'best.pth')))
        test_loss, test_acc = eval_epoch(model, dataloaders['test'])
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
    # psi_optimizer = optim.Adam(mus, lr=args.learning_rate)
    psi_optimizer = PsiSGD(psis, lr=args.learning_rate, momentum=0.9, weight_decay=2e-4, nesterov=True, num_data=len(dataloaders['unlabel']))

    if args.do_train:
        train_and_evaluate(args, model, mu_optimizer, psi_optimizer, dataloaders)
    else:
        model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'best.pth')))
        test_loss, test_acc = eval_epoch(model, dataloaders['test'])
        print(f"testing loss: {test_loss}")
        print(f"testing accuracy: {test_acc}")
