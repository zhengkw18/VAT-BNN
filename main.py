from trainer import Trainer
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import os
import argparse
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout, Bayes_ensemble
from scalablebdl.mean_field import PsiSGD, to_bayesian, to_deterministic
from datasets.cifar10 import load_cifar_dataset
from datasets.minist import load_minist_dataset
from models.wrn import wrn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--cutout', default=True, type=bool)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--mc_step', type=int, default=2)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--portion', default=0.1, type=float)
    parser.add_argument('--delta', default=0.00001, type=float)
    parser.add_argument('--data_path', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='./results', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./runs', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 30],
                    help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                    help='LR for psi is multiplied by gamma on schedule')
    args = parser.parse_args()

    config = 'batch-{}_epochs-{}'.format(args.batch_size, args.epochs)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)


    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    tb_writer = SummaryWriter(args.log_dir)
    labeled_train_loader, unlabeled_train_loader, test_loader, valid_loader, labeled_len = load_minist_dataset(args)
    net = wrn(pretrained=True, depth=28, width=10).to(device)
    print("load model successfully")
    # eval_loss, eval_acc = Bayes_ensemble(test_loader, net, num_mc_samples=1)
    # print('Results of deterministic pre-training, eval loss {}, eval acc {}'.format(eval_loss, eval_acc))

    
    if args.do_train:
        trainer = Trainer(device, net, tb_writer, num_label_data=labeled_len, args=args)
        trainer.train(args.epochs, args.logging_steps, train_dataloader=labeled_train_loader, test_dataloader=test_loader, unlabeled_train_loader=unlabeled_train_loader)

    # restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
    # netG.restore(restore_ckpt_path)

    # num_samples = 3000
    # real_imgs = None
    # real_dl = iter(dataset.training_loader)
    # while real_imgs is None or real_imgs.size(0) < num_samples:
    #     imgs = next(real_dl)
    #     if real_imgs is None:
    #         real_imgs = imgs[0]
    #     else:
    #         real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    # real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5

    # with torch.no_grad():
    #     samples = None
    #     while samples is None or samples.size(0) < num_samples:
    #         imgs = netG.forward(torch.randn(args.batch_size, netG.latent_dim, 1, 1, device=device))
    #         if samples is None:
    #             samples = imgs
    #         else:
    #             samples = torch.cat((samples, imgs), 0)
    # samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    # samples = samples.cpu()

    # fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
    # tb_writer.add_scalar('fid', fid)
    # print("FID score: {:.3f}".format(fid), flush=True)