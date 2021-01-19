import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from bnn_optim import PsiSGD
from bnn_utils import freeze, unfreeze, to_bayesian
from utils import virtual_adversarial_loss, entropy, mi_adversarial_loss
sns.set_style('white')

matplotlib.rcParams['figure.figsize'] = (5, 5)
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 1.0

np.random.seed(1)
size = 800
training_range = (-10, 10)
test_range = (-15, 15)
X, Y = datasets.make_blobs(n_samples=size, centers=4, cluster_std=1.8, center_box=training_range, random_state=32)


torch.manual_seed(99999)

m, n = X.shape
# m = 800, n = 2
h = 20  # num. hidden units
k = 4  # num. classes


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.feature_extr = nn.Sequential(
            nn.Linear(n, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU()
        )

        self.clf = nn.Linear(h, k, bias=False)

    def forward(self, x):
        x = self.feature_extr(x)
        return self.clf(x)

# in : 800 * 2
# out : 800 * 4


X_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(Y).long()


def bayesian_model():
    model = Model()
    model = to_bayesian(model)
    mus, psis = [], []
    for name, param in model.named_parameters():
        if 'psi' in name:
            psis.append(param)
        else:
            mus.append(param)
    mu_optimizer = optim.Adam(mus, lr=2e-4)
    psi_optimizer = PsiSGD(psis, lr=2e-4, momentum=0.9, weight_decay=2e-4, nesterov=True, num_data=size)
    for it in range(1200):
        if it % 200 == 0:
            print("iterate time: ", it)
        y = model(X_train)  # y:800 * 4
        loss = F.cross_entropy(y, y_train)
        loss.backward()
        mu_optimizer.step()
        psi_optimizer.step()
        mu_optimizer.zero_grad()
        psi_optimizer.zero_grad()

    print(f'Loss: {loss.item():.3f}')
    return model


def MI(logits):
    probs = [logit.softmax(dim=1) for logit in logits]
    ents = [entropy(logit) for logit in logits]
    prob = sum(probs) / len(probs)
    ent = sum(ents) / len(ents)
    return entropy(torch.log(prob)) - ent


def vat_model(epsilon, vat=False, train_range=1600):
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    for i in range(train_range):
        if i % 100 == 0:
            print("current iterate time", i)
        y = model(X_train)
        loss = F.cross_entropy(y, y_train)
        if vat:
            vat_loss = virtual_adversarial_loss(X_train, y, model, epsilon)
        else:
            vat_loss = torch.zeros_like(loss)
        loss += vat_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Loss: {loss.item():.3f}')
    return model


def mi_model(adv=False, train_range=1600):
    model = vat_model(1.5, False, train_range=1000)
    model = to_bayesian(model)
    mus, psis = [], []
    for name, param in model.named_parameters():
        if 'psi' in name:
            psis.append(param)
        else:
            mus.append(param)
    mu_optimizer = optim.Adam(mus, lr=2e-4)
    psi_optimizer = PsiSGD(psis, lr=1e-4, momentum=0.9, weight_decay=1e-4, nesterov=True, num_data=size)
    mi_avg, loss_avg, kl_avg = 0.0, 0.0, 0.0
    for it in range(train_range):
        if it % 100 == 0:
            mi_avg /= 100
            loss_avg /= 100
            kl_avg /= 100
            print("iterate time: ", it, "mi_avg: ", mi_avg, "kl_avg", kl_avg, "loss_avg", loss_avg)
            mi_avg = 0.0
            loss_avg = 0.0
            kl_avg = 0.0
        y = model(X_train)  # y:800 * 4
        mi_loss, kl_loss = mi_adversarial_loss(model, X_train, y, 1.5, adv)
        loss = F.cross_entropy(y, y_train)
        mi_avg += mi_loss.detach().numpy()
        kl_avg += kl_loss.detach().numpy()
        loss_avg += loss.detach().numpy()
        loss += 2 * mi_loss + kl_loss
        loss.backward()
        mu_optimizer.step()
        psi_optimizer.step()
        mu_optimizer.zero_grad()
        psi_optimizer.zero_grad()

    print(f'Loss: {loss.item():.3f}')
    return model


# model = mi_model(True, train_range=600)
model = mi_model(False, 400)
# model = vat_model(1.5, True)


def plot(X, Y, X1_test, X2_test, Z, test_range):
    cmap = 'Blues'
    plt.figure(figsize=(6, 5))

    im = plt.contourf(X1_test, X2_test, Z, alpha=0.7, cmap=cmap, levels=np.arange(np.min(Z) * 0.9, 1.15 * np.max(Z), (1.15 * np.max(Z) - np.min(Z) * 0.9) / 10))
    plt.colorbar(im)

    plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1],
                c='coral', edgecolors='k', linewidths=0.3)
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1],
                c='yellow', edgecolors='k', linewidths=0.3)
    plt.scatter(X[Y == 2][:, 0], X[Y == 2][:, 1],
                c='yellowgreen', edgecolors='k', linewidths=0.3)
    plt.scatter(X[Y == 3][:, 0], X[Y == 2][:, 1],
                c='violet', edgecolors='k', linewidths=0.3)

    plt.xlim(test_range)
    plt.ylim(test_range)
    plt.xticks([])
    plt.yticks([])

    plt.show()


size = 200
test_range = (-15, 15)
test_rng = np.linspace(*test_range, size)  # 在test_range里面生成50个随机数
# test_rng = np.linspace(-2, 2, num=5)
X1_test, X2_test = np.meshgrid(test_rng, test_rng)  # 生成网格坐标点，共2500个
X_test = np.stack([X1_test.ravel(), X2_test.ravel()]
                  ).T  # 把横纵坐标分别展平，即得到2500个点的横、纵坐标
X_test = torch.from_numpy(X_test).float()
# X_test: 2500 * 2

model.eval()
# model(X_test): 2000 * 4(可以理解为4分类问题得到的分类向量)


def print_conf():
    with torch.no_grad():
        py_map = F.softmax(model(X_test), 1).squeeze().numpy()
        conf = py_map.max(1)
        plot(X, Y, X1_test, X2_test, conf.reshape(size, size), test_range)


def print_entropy():
    with torch.no_grad():
        py_map = F.softmax(model(X_test), 1)
        py_map = -1 * torch.sum(py_map * torch.log(py_map), dim=1)
        conf = py_map.squeeze()
        conf = conf.numpy()
        plot(X, Y, X1_test, X2_test, conf.reshape(size, size), test_range)


def print_mi(total_calcu):
    with torch.no_grad():
        logits = []
        for _ in range(total_calcu):
            logit = model(X_test)
            # print(logit.shape)
            logits.append(logit)
        mi = MI(logits).squeeze().numpy()
        plot(X, Y, X1_test, X2_test, mi.reshape(size, size), test_range)


freeze(model)
print_conf()
print_entropy()
unfreeze(model)
print_mi(10)
