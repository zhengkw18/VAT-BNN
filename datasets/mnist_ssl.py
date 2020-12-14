import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


def load_mnist_dataset(args):

    dataset = dset.MNIST
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = dataset(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dataset(args.data_path, train=False, transform=test_transform, download=True)
    train_data, valid_data = torch.utils.data.random_split(train_data, lengths=[len(train_data) - round(0.1 * len(train_data)), round(0.1 * len(train_data))])

    print("Dataset: MNIST")
    print(f"Training set length: {len(train_data)}")
    print(f"Validation set length: {len(valid_data)}")
    print(f"Testing set length: {len(test_data)}")
    total_len = len(train_data)
    if args.label_num == 0:
        unlabeled_len = 0
        labeled_len = total_len
    else:
        unlabeled_len = total_len - args.label_num
        labeled_len = args.label_num
    unlabeled_indice = []
    labeled_indice = []
    targets = []
    for i in range(0, len(train_data)):
        targets.append(train_data[i][1])
    targets = np.array(targets)
    labeled_len_per_class = round(labeled_len / 10)
    for i in range(10):
        indices = np.where(targets == i)[0]
        labeled_indice = labeled_indice + indices[0: labeled_len_per_class].tolist()
        unlabeled_indice = unlabeled_indice + indices[labeled_len_per_class:].tolist()
    unlabeled_train_data = torch.utils.data.Subset(train_data, unlabeled_indice)
    labeled_train_data = torch.utils.data.Subset(train_data, labeled_indice)

    labeled_train_sampler = DistributedSampler(labeled_train_data) if args.distributed else None
    unlabeled_train_sampler = DistributedSampler(unlabeled_train_data) if args.distributed else None
    test_sampler = DistributedSampler(test_data) if args.distributed else None
    valid_sampler = DistributedSampler(valid_data) if args.distributed else None

    unlabeled_train_loader = torch.utils.data.DataLoader(unlabeled_train_data,
                                                         batch_size=args.batch_size, shuffle=(unlabeled_train_sampler is None),
                                                         num_workers=args.workers, pin_memory=True, sampler=unlabeled_train_sampler)
    labeled_train_loader = torch.utils.data.DataLoader(labeled_train_data,
                                                       batch_size=args.batch_size, shuffle=(labeled_train_sampler is None),
                                                       num_workers=args.workers, pin_memory=True, sampler=labeled_train_sampler)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, sampler=test_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True, sampler=valid_sampler)

    return labeled_train_loader, unlabeled_train_loader, test_loader, valid_loader, labeled_len
