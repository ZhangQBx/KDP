import torch
from torchvision import datasets, transforms

def cifar10(data_path):
    cifar10_trainset = datasets.CIFAR10(data_path, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
                                        ]))
    print("cifar10 training dataset prepared...")

    cifar10_testset = datasets.CIFAR10(data_path, train=False, download=True,
                                       transform=transforms.Compose([
                                           # transforms.RandomCrop(32, padding=4),
                                           # transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
                                       ]))
    print("cifar10 testing dataset prepared...")
    print("------------------------")
    return cifar10_trainset, cifar10_testset

def cifar100(data_path):
    cifar100_trainset = datasets.CIFAR100(data_path, train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
                                    ]))
    print("cifar100 training dataset prepared...")

    cifar100_testset = datasets.CIFAR100(data_path, train=False, download=True,
                                         transform=transforms.Compose([
                                             # transforms.RandomCrop(32, padding=4),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
                                         ]))

    print("cifar100 testing dataset prepared...")
    print("------------------------")
    return cifar100_trainset, cifar100_testset