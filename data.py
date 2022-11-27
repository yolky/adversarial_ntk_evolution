import torch
import numpy as np
import torchvision

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

def get_loader(dataset_name, train = True, batch_size = 128, shuffle = True):
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_train)
        
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform_train)
        
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    
    
def get_data_and_labels(dataset_name):
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        
    return torch.tensor(np.transpose(dataset.data, [0, 3, 1, 2]) / 255), torch.tensor(dataset.targets)

def get_n_classes(dataset_name):
    if dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'cifar100':
        return 100