import torch
import os
import argparse
from trainer import Trainer,covert_to_partial_bayesian
from tensorboardX import SummaryWriter
from datasets.cifar10 import load_cifar_dataset
from datasets.mnist import load_mnist_dataset
from models import SmallNet, LargeNet
from scalablebdl.mean_field import to_bayesian
from scalablebdl.bnn_utils import Bayes_ensemble

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--last_layer', action='store_true')
    parser.add_argument('--strategy', type=str, choices=["miadv_train", "bnn", "mipred"], default="bnn")
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bnn_epochs', default=30, type=int)
    parser.add_argument('--mc_step', type=int, default=2)
    parser.add_argument('--epsilon', type=float, default=2.0)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--label_num', default=0, type=int)
    parser.add_argument('--delta', default=0.000001, type=float)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--pretrain_config', default='base_batch-100_epochs-300_dataset-mnist_labelnum-0', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 30],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                        help='LR for psi is multiplied by gamma on schedule')
    args = parser.parse_args()

    if args.strategy == "bnn":
        config = f"{args.strategy}_epochs-{args.epochs}_dataset-{args.dataset}_labeled_num-{args.label_num}_ll-{args.last_layer}"
    elif args.strategy == "mipred":
        config = f"{args.strategy}_epochs-{args.epochs}_dataset-{args.dataset}_labeled_num-{args.label_num}_ll-{args.last_layer}_mc-{args.mc_step}"
    else:
        config = '{}_epochs-{}_bepochs-{}_dataset-{}_labeled_num-{}_epsilon-{}_ll-{}_mc-{}'.format(args.strategy, args.epochs, args.bnn_epochs, args.dataset, args.label_num, args.epsilon, args.last_layer, args.mv_step)
    print(args)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    device = torch.device('cuda')
    tb_writer = SummaryWriter(args.log_dir)
    if args.dataset == 'mnist':
        labeled_train_loader, unlabeled_train_loader, test_loader, valid_loader, labeled_len = load_mnist_dataset(args)
        net = SmallNet()
    elif args.dataset == 'cifar10':
        labeled_train_loader, unlabeled_train_loader, test_loader, valid_loader, labeled_len = load_cifar_dataset(args)
        net = LargeNet()
    else:
        print("Unsupported dataset.")
        exit(0)
    net = net.to(device)
    net.load_state_dict(torch.load(os.path.join(os.path.join('./ckpt', args.pretrain_config), 'best.pth')))
    print(f"load model successfully from {args.pretrain_config}")
    eval_loss, eval_acc = Bayes_ensemble(test_loader, net, num_mc_samples=1)
    print('Results of deterministic pre-training, eval loss {}, eval acc {}'.format(eval_loss, eval_acc))

    if args.do_train:
        trainer = Trainer(device, net, tb_writer, num_label_data=labeled_len, args=args)
        trainer.train(args.epochs, train_dataloader=labeled_train_loader, test_dataloader=test_loader, unlabeled_train_loader=unlabeled_train_loader, valid_dataloader=valid_loader)
    else:
        if args.last_layer:
            net = covert_to_partial_bayesian(net, args.dataset)
        else:
            net = to_bayesian(net)
        net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'best.pth')))
        eval_loss, eval_acc = Bayes_ensemble(test_loader, net, num_mc_samples=4)
        print('Results of mutual information bayesian training, eval loss {}, eval acc {}'.format(eval_loss, eval_acc))
    # restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
    # netG.restore(restore_ckpt_path)
