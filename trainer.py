import torch
import torch.nn as nn
import os
import numpy as np
from scalablebdl.mean_field import PsiSGD, to_bayesian
from scalablebdl.bnn_utils import Bayes_ensemble
from utils import get_normalized_vector, _disable_tracking_bn_stats, ce_loss
import torch.distributions as dist

def adjust_learning_rate(mu_optimizer, psi_optimizer, epoch, args):
    lr = args.learning_rate
    slr = args.learning_rate
    assert len(args.gammas) == len(args.schedule), \
        "length of gammas and schedule should be equal"
    for (gamma, step) in zip(args.gammas, args.schedule):
        if (epoch >= step):
            slr = slr * gamma
        else:
            break
    lr = lr * np.prod(args.gammas)
    for param_group in mu_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in psi_optimizer.param_groups:
        param_group['lr'] = slr
    return lr, slr


class Trainer(object):
    def __init__(self, device, model, tb_writer, num_label_data, args):
        self.device = device
        self.model = model
        self.ckpt_dir = args.ckpt_dir
        self.tb_writer = tb_writer
        os.makedirs(args.ckpt_dir, exist_ok=True)
        # self.model.restore(ckpt_dir)
        self.optimizer = None
        self.mu_optim = None
        self.psi_optim = None
        self.num_label_data = num_label_data
        self.delta = args.delta
        self.epsilon = args.epsilon
        self.MC_Step = args.mc_step
        self.lr = args.learning_rate
        self.args = args

        # for mi debug
        # self.total_time = 0
        # self.succ_time = 0
        # self.temp = 0

    def disable_model_grad(self):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def enable_model_grad(self):
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

    def mutual_information(self, input: list):
        '''
        input shape : [(batch_size, num_class)] for n MC sample probabilty
        output:       MI of input
        '''
        ensemble_prob = 0
        ensemble_entropy = 0
        for prob in input:
            ensemble_prob = ensemble_prob + prob
            entropy = -1 * torch.sum(torch.log(prob) * prob, dim=1)
            ensemble_entropy = ensemble_entropy + entropy

        ensemble_entropy = ensemble_entropy / len(input)
        ensemble_prob = ensemble_prob / len(input)
        ground_entropy = -1 * torch.sum(torch.log(ensemble_prob) * ensemble_prob, dim=1)
        return torch.mean(ground_entropy - ensemble_entropy)

    def mc_calulate_mi(self, input):
        outputs = []
        for _ in range(self.MC_Step):
            output = self.model(input)
            outputs.append(output)
        outputs = [torch.softmax(output, dim=1) for output in outputs]
        return self.mutual_information(outputs)

    def generate_mi_adversarial_perturbation(self, input):
        d = torch.zeros_like(input)
        d.requires_grad_(True)
        with _disable_tracking_bn_stats(self.model):
            mi_pre_perturb = self.mc_calulate_mi(input+d)
            grad = torch.autograd.grad(mi_pre_perturb, [d])[0]
        r_vadv = self.epsilon * get_normalized_vector(grad)
        return r_vadv.detach()


    def fine_tune_step(self, input, target, ul_input):
        '''
            1: calculate the MI of unlabeled input, then find a perturb to maximize the MI
            2: recalculate the Mi with the unlabeled input perturbed
            3: calculate the loss of labeled input data
            4: backward
        '''
        self.model.train()
        if self.args.adv_train:
            r_adv = self.generate_mi_adversarial_perturbation(ul_input)
        else: 
            r_adv = torch.zeros_like(ul_input)
        self.mu_optim.zero_grad()
        self.psi_optim.zero_grad()
        output = self.model(input)
        loss = ce_loss(output, target)
        mi_aft_perturb = self.mc_calulate_mi(ul_input + r_adv)
        total_loss = loss + mi_aft_perturb
        total_loss.backward()
        self.mu_optim.step()
        self.psi_optim.step()

        return loss, mi_aft_perturb, total_loss

    def convert_to_bayesian(self):
        '''
            after the pretrain stage, convert the model to bayesian model
            follow the example given in https://github.com/thudzj/ScalableBDL
        '''
        self.model = to_bayesian(self.model)
        mus, psis = [], []
        for name, param in self.model.named_parameters():
            if 'psi' in name:
                psis.append(param)
            else:
                mus.append(param)
        self.mu_optim = torch.optim.SGD(mus, lr=self.lr, momentum=0.9,
                                        weight_decay=2e-4, nesterov=True)
        self.psi_optim = PsiSGD(psis, lr=self.lr, momentum=0.9,
                                weight_decay=2e-4, nesterov=True,
                                num_data=self.num_label_data)

    def train(self, epochs, logging_steps, train_dataloader, test_dataloader, valid_dataloader, unlabeled_train_loader):
        best_valid_acc = 0
        best_epoch = 0
        self.convert_to_bayesian()
        for epoch in range(epochs):
            unlabeled_iter = iter(unlabeled_train_loader) if unlabeled_train_loader is not None else None
            cur_lr, cur_slr = adjust_learning_rate(self.mu_optim, self.psi_optim, epoch, self.args)
            print(f"current epoch: {epoch}, current lr: {cur_lr}, current slr: {cur_slr}")
            train_loss = 0
            train_total_loss = 0
            train_mi = 0
            for i, (input, target) in enumerate(train_dataloader):
                input = input.to(self.device)
                target = target.to(self.device)
                if unlabeled_iter is not None:
                    unlabeled_input, _ = next(unlabeled_iter)
                    if len(unlabeled_input) > 128 :
                        indice = torch.multinomial(torch.ones(len(unlabeled_input)), num_samples=128, replacement=False)
                        unlabeled_input = unlabeled_input[indice]
                    unlabeled_input = unlabeled_input.to(self.device)
                    unlabeled_input = torch.cat((input, unlabeled_input), dim=0)
                else:
                    unlabeled_input = input
                loss, mi_aft_perturb, total_loss = self.fine_tune_step(input, target, unlabeled_input)
                train_loss = train_loss + loss.item()
                train_total_loss = train_total_loss + total_loss.item()
                train_mi = train_mi + mi_aft_perturb.item()

            train_mi = train_mi/len(train_dataloader)
            train_loss = train_loss/len(train_dataloader)
            train_total_loss = train_total_loss/len(train_dataloader)
            valid_loss, valid_acc = Bayes_ensemble(valid_dataloader, self.model)
            self.tb_writer.add_scalar("train classification loss ", train_loss, global_step=epoch)
            self.tb_writer.add_scalar("train mutual information", train_mi, global_step=epoch)
            self.tb_writer.add_scalar("train total loss", train_total_loss, global_step=epoch)
            self.tb_writer.add_scalar("validation loss", valid_loss, global_step=epoch)
            self.tb_writer.add_scalar("validation accuracy", valid_acc, global_step=epoch)
            print(f"epoch: {epoch}\n validation loss: {valid_loss}\n validation accuracy: {valid_acc}\n current best validation accuracy: {best_valid_acc}\n train loss: {train_loss}\n train mi: {train_mi}")
            if valid_acc > best_valid_acc:
                print("now test on test dataset:")
                best_epoch = epoch
                best_valid_acc = valid_acc
                test_loss, test_acc = Bayes_ensemble(test_dataloader, self.model)
                print(f" test loss: {test_loss}\n test accuracy: {test_acc}\n best epoch: {best_epoch}")
                os.makedirs(self.args.ckpt_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.args.ckpt_dir, 'best.pth'))
