import torch
import torch.nn as nn
import os
import numpy as np
from scalablebdl.mean_field import PsiSGD, to_bayesian
from scalablebdl.bnn_utils import Bayes_ensemble
from torch.autograd import Variable
from utils import get_normalized_vector
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
        self.MC_Step = args.mc_step
        self.lr = args.learning_rate
        self.args = args

        # for mi debug
        self.total_time = 0
        self.succ_time = 0
        self.temp = 0

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
        ensemble_prob = None
        ensemble_entropy = None
        for prob in input:
            ensemble_prob = prob if ensemble_prob is None else ensemble_prob + prob
            entropy = -1 * torch.sum(torch.log(prob) * prob, dim=1)
            ensemble_entropy = entropy if ensemble_entropy is None else ensemble_entropy + entropy

        ensemble_entropy = ensemble_entropy / len(input)
        ensemble_prob = ensemble_prob / len(input)
        ground_entropy = -1 * torch.sum(torch.log(ensemble_prob) * ensemble_prob, dim=1)
        return torch.mean(ground_entropy - ensemble_entropy)

    def fine_tune_step(self, input, target, ul_input, loss_func):
        '''
            1: calculate the MI of unlabeled input, then find a perturb to maximize the MI
            2: recalculate the Mi with the unlabeled input perturbed
            3: calculate the loss of labeled input data
            4: backward
        '''
        # print(f"the input is {input.sum()}")
        # print(f"the input shape is {input.shape}")
        # print(f"the unlabeled input is {ul_input.sum()}")
        # print(f"the unlabeled input shape is {ul_input.shape}")
        self.model.train()
        d = torch.randn_like(ul_input)*self.delta
        d = Variable(d, requires_grad=True)
        self.total_time = self.total_time + 1

        self.disable_model_grad()
        outputs = []
        for _ in range(self.MC_Step):
            output = self.model(ul_input + d)
            outputs.append(output)
        outputs = [torch.softmax(output, dim=1) for output in outputs]
        # print(f"outouts are {outputs}")
        mi_pre_perturb = self.mutual_information(outputs)
        self.temp = mi_pre_perturb
        # print(f"mi_pre_perturb: {mi_pre_perturb}")
        mi_pre_perturb.backward()
        grad = d.grad
        r_adv = get_normalized_vector(grad)*self.delta
        r_adv = r_adv.detach()

        self.enable_model_grad()
        self.mu_optim.zero_grad()
        self.psi_optim.zero_grad()
        outputs = []
        for _ in range(self.MC_Step):
            output = self.model(ul_input + r_adv)
            outputs.append(output)
        outputs = [torch.softmax(output, dim=1) for output in outputs]
        mi_aft_perturb = self.mutual_information(outputs)
        # print(f"mi_aft_perturb: {mi_aft_perturb}")
        self.succ_time = self.succ_time + int(mi_aft_perturb > self.temp)
        # print(f"total time: {self.total_time}, success: {self.succ_time}")

        output = self.model(input)
        loss = loss_func(output, target)

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
        loss_func = nn.CrossEntropyLoss()
        self.convert_to_bayesian()
        for epoch in range(epochs):
            unlabeled_iter = iter(unlabeled_train_loader) if unlabeled_train_loader is not None else None
            cur_lr, cur_slr = adjust_learning_rate(self.mu_optim, self.psi_optim, epoch, self.args)
            print(f"current epoch: {epoch}, current lr: {cur_lr}, current slr: {cur_slr}")
            for i, (input, target) in enumerate(train_dataloader):
                input = input.to(self.device)
                target = target.to(self.device)

                if unlabeled_iter is not None:
                    unlabeled_input, _ = next(unlabeled_iter)
                    unlabeled_input = unlabeled_input.to(self.device)
                    unlabeled_input = torch.cat((input, unlabeled_input), dim=0)
                else:
                    unlabeled_input = input

                print(input.shape)
                print(unlabeled_input.shape)

                loss, mi_aft_perturb, total_loss = self.fine_tune_step(input, target, unlabeled_input, loss_func)
                if (i + 1) % logging_steps == 0:
                    print(f"classification loss: {loss}")
                    print(f"mutual information: {mi_aft_perturb}")
                    print(f"total loss: {total_loss}")
                    self.tb_writer.add_scalar("classification loss ", loss, global_step=i)
                    self.tb_writer.add_scalar("mutual information", mi_aft_perturb, global_step=i)
                    self.tb_writer.add_scalar("total loss", total_loss, global_step=i)
            valid_loss, valid_acc = Bayes_ensemble(valid_dataloader, self.model)
            print(f"epoch: {epoch}, validation loss: {valid_loss}, validation accuracy: {valid_acc}, current best validation accuracy: {best_valid_acc}")
            if valid_acc > best_valid_acc:
                print("now test on test dataset")
                best_valid_acc = valid_acc
                test_loss, test_acc = Bayes_ensemble(test_dataloader, self.model)
                print(f"test loss: {test_loss}, test accuracy: {test_acc}")
