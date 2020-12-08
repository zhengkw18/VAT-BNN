from trainer import Trainer
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--mc_step', type=int, default=2)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--portion', default=0.1, type=float)
    parser.add_argument('--delta', default=0.00001, type=float)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./ckpt', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)

    args = parser.parse_args()
