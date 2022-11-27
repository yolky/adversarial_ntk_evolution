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
from utils import bind, _add, _sub, _multiply
import os
from test_functions import do_perturbation_step_l_inf, do_perturbation_step_l_2, perturb, test, loss_fn
import numpy as np
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default = '', help = 'saved model path')
    parser.add_argument('--dataset_size', type=int, default = 500, help = 'size of the saved kernel')
    parser.add_argument('--save_name', type=str, default = 'ntk_eig_images', help = 'what to name the saved images')
    parser.add_argument('--class_index', type=int, default = -1, help = 'what class to use for the ntk')
    parser.add_argument('--kernel_path', type=str, default = '', help = 'where the kernel is saved')
    parser.add_argument('--n_images', type=int, default = 3, help = 'number of visualization images to make')
    args = parser.parse_args()

    train_size = args.dataset_size
    selected_class = args.class_index

    x_train = np.transpose(torch.tensor(torch.load('../Pytorch-Adversarial-Training-CIFAR/X_nothing')).cpu().numpy(), [0,2,3,1])[:train_size]

    rng = jax.random.PRNGKey(0)
    net_forward_init, net_forward_apply = models.get_resnet()

    checkpoint = pickle.load(open('./{}'.format(args.model_path), 'rb'))
    params = checkpoint['params']

    lin_params = checkpoint['lin_params']
    net_state = checkpoint['net_state']

    net_forward_binded = lambda a, b: bind(net_forward_apply, ..., net_state, rng, is_training = False)(a,b)[0][:, selected_class : selected_class + 1]

    labels = torch.load('../Pytorch-Adversarial-Training-CIFAR/y_train').cpu()
    y_oh = torch.nn.functional.one_hot(labels[:train_size], 10).double().cpu().numpy()
    
    network_info = (params, net_state, net_forward_apply, rng, net_forward_binded, selected_class, lin_params)


    kernel = pickle.load(open('./{}'.format(args.kernel_path), 'rb'))['kernel']

    U = np.linalg.svd(kernel)[0]
    
    pos_images = np.zeros([args.n_images, 32, 32, 3])
    
    
    for i in range(args.n_images):
        pos_images[i] = visualize_eig(kernel, U, y_oh, i, x_train, network_info, flip = False)
    
    neg_images = np.zeros([args.n_images, 32, 32, 3])
    
    for i in range(args.n_images):
        neg_images[i] = visualize_eig(kernel, y_oh, U, i, x_train, network_info, flip = True)
        
    w_image = visualize_eig(kernel, y_oh, U, 'w', x_train, network_info, flip = False)
    
    base_path = os.path.dirname(args.model_path)
    pickle.dump({'neg_images': neg_images, 'pos_images': pos_images, 'w_image': w_image}, open('./{}/{}_class_{}_{}.pkl'.format(base_path, args.save_name, args.class_index, args.dataset_size),'wb'))

@functools.partial(jax.jit, static_argnums=(3,))
def weighted_forward(params, weights, images, net_forward_binded):
    return jnp.sum(weights[None, :] @ net_forward_binded(params, images))

@functools.partial(jax.jit, static_argnums=(2, 3, 6, 7, 8))
def get_g_mag(params, net_state, net_forward_binded, net_forward_apply, rng, images, selected_class, is_training = False, centering = True):
    g = jax.grad(lambda a, b: net_forward_binded(a,b)[0, 0])(params, images)
    return -1 * models.linear_forward(params, _add(params, g), net_state, net_forward_apply, rng, images, is_training = is_training, centering = centering)[0][0, selected_class]

@functools.partial(jax.jit, static_argnums=(3, 4, 7, 8))
def get_mmd(params, feature_vec, net_state, net_forward_binded, net_forward_apply, rng, images, is_training = False, centering = True):
    g = jax.grad(lambda a, b: net_forward_binded(a,b)[0, 0])(params, images)
    return -1 * models.linear_forward(params, _sub(_add(params, feature_vec), g), net_state, net_forward_apply, rng, images, is_training = is_training, centering = centering)[0][0, selected_class]

@functools.partial(jax.jit, static_argnums=(3, 4, 8, 9, 10))
def get_cos(params, feature_vec, net_state, net_forward_binded, net_forward_apply, rng, images, feature_vec_mag, selected_class, is_training = False, centering = True):
    g = jax.grad(lambda a, b: net_forward_binded(a,b)[0, 0])(params, images)
    gtg = -1 * models.linear_forward(params, _add(params, g), net_state, net_forward_apply, rng, images, is_training = is_training, centering = centering)[0][0, selected_class]
    gtv = -1 * models.linear_forward(params, _add(params, feature_vec), net_state, net_forward_apply, rng, images, is_training = is_training, centering = centering)[0][0, selected_class]
    return gtv/ (jnp.sqrt(gtg) * jnp.sqrt(feature_vec_mag)), [gtg, gtv]

def dumb_f(a, b):
    return jnp.sum(a) + jnp.sum(b)

def visualize_eig(kernel, y_oh, U, eig_index, x_train, network_info, mode = 'cos', flip = False):
    params, net_state, net_forward_apply, rng, net_forward_binded, selected_class, lin_params = network_info
    
    if eig_index == 'w':
        weights = np.linalg.solve(kernel, y_oh[:, selected_class: selected_class + 1]).reshape(-1)
        weights = weights #* 4

    else:
        weights = U[:, eig_index]

        batch_size = 100

        feature_vec = None
        for b in range(int(x_train.shape[0]/batch_size)):
            x_batch = x_train[b * batch_size : (b+1) * batch_size]
            g = jax.grad(weighted_forward)(params, weights[b * batch_size : (b+1) * batch_size], x_batch, net_forward_binded)

            if feature_vec is None:
                feature_vec = g
            else:
                feature_vec = _add(feature_vec, g)
            
    base_image = np.zeros([1, 32, 32, 3]) + 0.5 
    feature_vec_mag = jax.tree_util.tree_reduce(dumb_f, _multiply(feature_vec, feature_vec))

    
    for i in range(600):
        if mode == 'l2':
            mag, g_combined = jax.value_and_grad(get_mmd, argnums = 6)(params, feature_vec, net_state, net_forward_binded, net_forward_apply, rng, base_image, selected_class, is_training = False, centering = True)
        elif mode == 'cos':
            [cos, aux], g_combined = jax.value_and_grad(get_cos, argnums = 6, has_aux = True)(params, feature_vec, net_state, net_forward_binded, net_forward_apply, rng, base_image, feature_vec_mag, selected_class, is_training = False, centering = True)
        
        if not flip:
            base_image += 0.001 * jnp.sign(g_combined)
        else:
            base_image -= 0.001 * jnp.sign(g_combined)
            
        base_image = np.clip(base_image, 0, 1)
        if i% 40 == 0:
            if mode == 'l2':
                print('{}, {}'.format(i, feature_vec_mag - mag))
            elif mode == 'cos':
                print('{}, {}'.format(i, cos))
                
    return base_image[0]
        
if __name__ == '__main__':
    main()