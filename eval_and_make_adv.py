import jax
import haiku as hk
import jax.numpy as jnp
from jax.example_libraries import optimizers
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import neural_tangents as nt
import functools
import operator
import optax
import copy
import models
import pickle
from utils import bind, _add, _sub
import os
from test_functions import do_perturbation_step_l_inf, do_perturbation_step_l_2, perturb, test, loss_fn
import numpy as np
import argparse
import time
import data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default = '', help = 'where the model is saved')
    parser.add_argument('--save_name', type=str, default = 'test_results', help = 'extra specifier to include when saving results')
    parser.add_argument('--linear', action='store_true', help = 'whether the loaded model is in linearized dynamics')
    parser.add_argument('--centering', action='store_true', help = 'whether the loaded model is in centered dynamics')
    parser.add_argument('--test_path', type=str, default = '', help = 'where the test images are. Set it to empty to use standard cifar10/cifar100 images. This is used for adversarial transferability testing')
    parser.add_argument('--bonus_dir', type=str, default = '.', help = 'include an extra dir the save path for more specific save locations')
    parser.add_argument('--no_adv', action='store_true', help = 'whether to skip making adversarial examples')
    parser.add_argument('--save_examples', action='store_true', help = 'whether to save the adversarial examples')
    parser.add_argument('--short', action='store_true', help = 'dont set this. basically just for debugging')
    parser.add_argument('--random_seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--dataset', type=str, default = 'cifar10', help = 'cifar10/cifar100')
    parser.add_argument('--model', type=str, default = 'resnet18', help = 'model')
    parser.add_argument('--eps', type=float, default = 4.00, help = 'eps value for l-inf adversarial attacks. scaled by 1/255')
    args = parser.parse_args()



    transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])


    class TensorDataset(Dataset):
        def __init__(self, *tensors, transform=None):
            assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
            self.tensors = tensors
            self.transform = transform

        def __getitem__(self, index):
            im, targ = tuple(tensor[index] for tensor in self.tensors)
            if self.transform:
                real_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    self.transform
                ])
                im = real_transform(im)
            return im, targ

        def __len__(self):
            return self.tensors[0].size(0)
            

    rng = jax.random.PRNGKey(args.random_seed)
    net_forward_init, net_forward_apply = models.get_model(args.model, data.get_n_classes(args.dataset))

    checkpoint = pickle.load(open('./{}'.format(args.model_path), 'rb'))
    params = checkpoint['params']

    lin_params = checkpoint['lin_params']
    net_state = checkpoint['net_state']

    if len(args.test_path) > 0:
        test_stuff = pickle.load(open('./{}'.format(args.test_path), 'rb'))
        test_data = torch.tensor(test_stuff['images']).cpu()
        print(test_data.shape)

        test_labels = torch.tensor(test_stuff['labels']).cpu()
        print(test_labels)
        
        test_dataset = TensorDataset(test_data, test_labels, transform=transform_test)
    elif args.dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_labels = np.array(test_dataset.targets)
        print(test_labels)
    elif args.dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_labels = np.array(test_dataset.targets)
        print(test_labels)
        
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)


    acc_clean, acc_dirty, adv_examples, predictions_clean, predictions_dirty, components_clean, components_dirty = test(params, lin_params, net_state, net_forward_apply, rng, test_loader, make_adv_examples = not args.no_adv, linear = args.linear, centering = args.centering, attack = 'linf', return_examples = True, short = args.short, return_components = True, adv_eps = args.eps)


    base_path = os.path.dirname(args.model_path)

    results_dict = {
        'acc_clean': acc_clean,
        'acc_dirty': acc_dirty,
        'predictions_clean': predictions_clean,
        'predictions_dirty': predictions_dirty,
        'components_clean': components_clean,
        'components_dirty': components_dirty,
    }


    print('clean: {:.2f} dirty: {:.2f}'.format(100 * acc_clean, 100 * acc_dirty))

    if not os.path.isdir('./{}/{}/'.format(base_path, args.bonus_dir)):
        os.mkdir('./{}/{}/'.format(base_path, args.bonus_dir))

    pickle.dump(results_dict, open('./{}/{}/test_results_{}.pkl'.format(base_path, args.bonus_dir, args.save_name),'wb'))


    if args. save_examples:
        pickle.dump({'images': np.transpose(adv_examples, [0, 3, 1, 2]), 'labels': test_labels[:adv_examples.shape[0]]}, open('./{}/{}/adv_examples_{}.pkl'.format(base_path, args.bonus_dir, args.save_name),'wb'))

if __name__ == '__main__':
    main()
    