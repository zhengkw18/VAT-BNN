import torch.nn as nn
# from scalablebdl.mean_field import to_bayesian


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class SmallNet(nn.Module):
    def __init__(self, layer_sizes=[784, 1200, 600, 300, 150, 10]):
        super(SmallNet, self).__init__()
        self.linear_layers = []
        self.bn_layers = []
        self.act_layers = []
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.linear_layers.append(nn.Linear(in_features=m, out_features=n))
            self.bn_layers.append(nn.BatchNorm1d(num_features=n))
        for i in range(len(self.linear_layers) - 1):
            self.act_layers.append(nn.ReLU())
        self.act_layers.append(EmptyLayer())
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.bn_layers = nn.ModuleList(self.bn_layers)
        self.act_layers = nn.ModuleList(self.act_layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for l, bn, act in zip(self.linear_layers, self.bn_layers, self.act_layers):
            x = l(x)
            x = bn(x)
            x = act(x)
        return x


class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1)
            
        )
        self.last_feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.dense = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.feature_layers(x)
        x = self.last_feature_layers(x)
        x = x.mean(dim=3).mean(dim=2)
        x = self.dense(x)
        return x
