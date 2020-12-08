import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def load_cifar_dataset(args):
    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=1, length=16))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_data = dataset(args.data_path, train=True,
                             transform=train_transform, download=True)
        test_data = dataset(args.data_path, train=False,
                            transform=test_transform, download=True)

        train_data, valid_data = torch.utils.data.random_split(train_data, lengths=[round(0.1*len(train_data)), len(train_data) - round(0.1*len(train_data))])

        total_len = len(train_data)
        unlabeled_len = round(total_len*args.portion)
        labeled_len = total_len - unlabeled_len
        unlabeled_indice = []
        labeled_indice = []
        targets = np.array(train_data.targets)
        unlabeled_len_per_class = round(unlabeled_len/10)
        for i in range(10):
            indices = np.where(targets == i)[0]
            unlabeled_indice = unlabeled_indice + indices[0:unlabeled_len_per_class].tolist()
            labeled_indice = labeled_indice + indices[unlabeled_len_per_class:].tolist()
        unlabeled_train_data = torch.utils.data.Subset(train_data, unlabeled_indice)
        labeled_train_data = torch.utils.data.Subset(train_data, labeled_indice)

        labeled_train_sampler = DistributedSampler(labeled_train_data) if args.distributed else None
        unlabeled_train_sampler = DistributedSampler(unlabeled_train_data) if args.distributed else None
        test_sampler = DistributedSampler(test_data) if args.distributed else None
        valid_sampler = DistributedSampler(valid_data) if args.distributed else None

        unlabeled_batch_size = round(unlabeled_len * args.batch_size / labeled_len) -1

        unlabeled_train_loader = torch.utils.data.DataLoader(unlabeled_train_data,
            batch_size=unlabeled_batch_size, shuffle=(unlabeled_train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=unlabeled_train_sampler) if unlabeled_batch_size > 0 else None
        labeled_train_loader = torch.utils.data.DataLoader(labeled_train_data,
            batch_size=args.batch_size, shuffle=(labeled_train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=labeled_train_sampler)
        test_loader = torch.utils.data.DataLoader(test_data,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler)
        valid_loader = torch.utils.data.DataLoader(valid_data,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=valid_sampler)
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    return labeled_train_loader, unlabeled_train_loader,  test_loader, valid_loader, labeled_len
