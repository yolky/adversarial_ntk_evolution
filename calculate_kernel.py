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
from utils import bind, _add, _sub, get_class_indices
import os
from test_functions import do_perturbation_step_l_inf, do_perturbation_step_l_2, perturb, test, loss_fn
import numpy as np
import argparse
import time
import data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default = '', help = 'path of the saved model')
    parser.add_argument('--dataset_size', type=int, default = 500, help = 'number of images to estimate the kernel with')
    parser.add_argument('--save_name', type=str, default = 'saved_kernel', help = 'what to name the saved model')
    parser.add_argument('--class_index', type=int, default = -1, help = 'which class to make for the kernel. Default is -1 which means we basically average out every classes kernel')
    parser.add_argument('--show_progress', action='store_true', help = 'for when youre impatient and want to see every time a kernel sub block is made')
    parser.add_argument('--model', type=str, default = 'resnet18', help = 'model')
    parser.add_argument('--bonus_dir', type=str, default = '.', help = 'extra directory for more specific save locations')
    parser.add_argument('--random_seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--use_linear_params', action='store_true', help = '')
    parser.add_argument('--dataset', type=str, default = 'cifar10')
    args = parser.parse_args()


    train_data, train_labels = data.get_data_and_labels(args.dataset)

    x_train = np.transpose(train_data.cpu().numpy(), [0,2,3,1])

    rng = jax.random.PRNGKey(args.random_seed)
    net_forward_init, net_forward_apply = models.get_model(args.model, data.get_n_classes(args.dataset))


    train_subindices = get_class_indices(train_labels, int((args.dataset_size)/10), args.random_seed, n_classes = 10)

    x_train = x_train[train_subindices]

    checkpoint = pickle.load(open('./{}'.format(args.model_path), 'rb'))
    params = checkpoint['params']

    lin_params = checkpoint['lin_params']

    if args.use_linear_params:
        params = lin_params
    net_state = checkpoint['net_state']


    if args.class_index == -1:
        print("Calculating Combined Kernel")
        net_forward_binded = lambda a, b: bind(net_forward_apply, ..., net_state, rng, is_training = False)(a,b)[0]
    else:
        print("Calculating Kernel for class {}".format(args.class_index))
        net_forward_binded = lambda a, b: bind(net_forward_apply, ..., net_state, rng, is_training = False)(a,b)[0][:, args.class_index : args.class_index + 1]


    kernel = np.zeros([x_train.shape[0], x_train.shape[0]])

    kernel_fn = nt.empirical_kernel_fn(net_forward_binded, implementation = 2)
    batch_size = 4
    kernel_fn = jax.jit(nt.batch(kernel_fn, batch_size=batch_size), static_argnums = (2))

    for a in range(int(args.dataset_size/batch_size)):
        for b in range(a, int(args.dataset_size/batch_size)):
            start = time.time()
            if args.show_progress:
                print(a, b)
                
            kernel[a * batch_size : (a+1) * batch_size, b * batch_size : (b+1) * batch_size]  = kernel_fn(x_train[a * batch_size : (a+1) * batch_size],  x_train[b * batch_size : (b+1) * batch_size], 'ntk', params)
            
            
    kernel = np.triu(kernel) + np.triu(kernel, k = 1).T
        
    base_path = os.path.dirname(args.model_path)



    if not os.path.isdir('./{}/{}/'.format(base_path, args.bonus_dir)):
        os.mkdir('./{}/{}/'.format(base_path, args.bonus_dir))
        
    if args.class_index == -1:
        pickle.dump({'kernel': kernel, 'labels': train_labels[train_subindices].numpy()}, open('./{}/{}/{}_{}.pkl'.format(base_path, args.bonus_dir, args.save_name, args.dataset_size),'wb'))
    else:
        pickle.dump({'kernel': kernel, 'labels': train_labels[train_subindices].numpy()}, open('./{}/{}/{}_class_{}_{}.pkl'.format(base_path, args.bonus_dir, args.save_name, args.class_index, args.dataset_size),'wb'))

if __name__ == '__main__':
    main()
    