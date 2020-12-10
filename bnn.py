import torch
import os
import argparse
from trainer import Trainer
from tensorboardX import SummaryWriter
from datasets.cifar10 import load_cifar_dataset
from datasets.mnist import load_mnist_dataset
from models import SmallNet, LargeNet
from scalablebdl.mean_field import to_bayesian
from scalablebdl.bnn_utils import Bayes_ensemble

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--mc_step', type=int, default=2)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=2.0)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--label_num', default=0, type=int)
    parser.add_argument('--delta', default=0.000001, type=float)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--pretrain_config', default='base_batch-100_epochs-100_dataset-mnist_labelnum-0', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 30],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                        help='LR for psi is multiplied by gamma on schedule')
    args = parser.parse_args()

    config = 'mivat_batch-{}_epochs-{}_dataset-{}_labeled_num-{}_epsilon-{}'.format(args.batch_size, args.epochs, args.dataset, args.label_num, args.epsilon)
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
        trainer.train(args.epochs, args.logging_steps, train_dataloader=labeled_train_loader, test_dataloader=test_loader, unlabeled_train_loader=unlabeled_train_loader, valid_dataloader=valid_loader)
    else:
        net = to_bayesian(net)
        net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'best.pth')))
        eval_loss, eval_acc = Bayes_ensemble(test_loader, net, num_mc_samples=4)
        print('Results of mutual information bayesian training, eval loss {}, eval acc {}'.format(eval_loss, eval_acc))
    # restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
    # netG.restore(restore_ckpt_path)
