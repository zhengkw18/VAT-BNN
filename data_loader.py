import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def split_datasets(train_dataset, n_labels, n_val):
    """
    Split train dataset into labeled one, unlabeled one, and validation set.
    """
    n_classes = 10
    n_labels_per_class = n_labels / n_classes
    n_val_per_class = n_val / n_classes
    labels_indices = {c: [] for c in range(n_classes)}
    val_indices = {c: [] for c in range(n_classes)}

    rand_indices = [i for i in range(len(train_dataset))]
    np.random.seed(1)
    np.random.shuffle(rand_indices)
    for idx in rand_indices:
        target = int(train_dataset[idx][1])
        if len(labels_indices[target]) < n_labels_per_class:
            labels_indices[target].append(idx)
        elif len(val_indices[target]) < n_val_per_class:
            val_indices[target].append(idx)
        else:
            continue

    labels_set, val_set = [], []
    for indices in labels_indices.values():
        labels_set.extend(indices)
    for indices in val_indices.values():
        val_set.extend(indices)
    assert len(labels_set) == n_labels
    assert len(val_set) == n_val
    unlabels_set = list(set(range(len(train_dataset))) - set(val_set))
    if n_labels == 0:
        labels_set = unlabels_set
    return labels_set, unlabels_set, val_set


def fetch_dataloaders_MNIST(args):
    """
    Fetches the DataLoader objects for MNIST.
    """

    train_dataset = torchvision.datasets.MNIST(args.data_path, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(args.data_path, train=False, transform=transforms.ToTensor(), download=True)
    labels_set, unlabels_set, val_set = split_datasets(train_dataset, args.label_num, 1000)

    print("Dataset: MNIST")
    print(f"Label set length: {len(labels_set)}")
    print(f"Unlabel set length: {len(unlabels_set)}")
    print(f"Validation set length: {len(val_set)}")
    print(f"Testing set length: {len(test_dataset)}")

    labeled_dataset = torch.utils.data.Subset(train_dataset, labels_set)
    unlabeled_dataset = torch.utils.data.Subset(train_dataset, unlabels_set)
    val_dataset = torch.utils.data.Subset(train_dataset, val_set)

    dataloaders = {}
    dataloaders['label'] = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dataloaders['unlabel'] = DataLoader(unlabeled_dataset, batch_size=args.ul_batch_size, shuffle=True, num_workers=args.workers)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=args.ul_batch_size, shuffle=False, num_workers=args.workers)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=args.ul_batch_size, shuffle=False, num_workers=args.workers)

    return dataloaders


def fetch_dataloaders_CIFAR10(args):
    """
    Fetches the DataLoader objects for CIFAR10.
    """

    train_dataset = torchvision.datasets.CIFAR10(args.data_path, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(args.data_path, train=False, transform=transforms.ToTensor(), download=True)
    labels_set, unlabels_set, val_set = split_datasets(train_dataset, args.label_num, 10000)

    print("Dataset: CIFAR10")
    print(f"Label set length: {len(labels_set)}")
    print(f"Unlabel set length: {len(unlabels_set)}")
    print(f"Validation set length: {len(val_set)}")
    print(f"Testing set length: {len(test_dataset)}")

    labeled_dataset = torch.utils.data.Subset(train_dataset, labels_set)
    unlabeled_dataset = torch.utils.data.Subset(train_dataset, unlabels_set)
    val_dataset = torch.utils.data.Subset(train_dataset, val_set)

    dataloaders = {}
    dataloaders['label'] = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    dataloaders['unlabel'] = DataLoader(unlabeled_dataset, batch_size=args.ul_batch_size, shuffle=True, num_workers=args.workers)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=args.ul_batch_size, shuffle=False, num_workers=args.workers)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=args.ul_batch_size, shuffle=False, num_workers=args.workers)

    return dataloaders
