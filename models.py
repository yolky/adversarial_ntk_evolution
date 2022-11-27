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
import sys
from utils import bind, _sub, _add
import modified_resnets


@functools.partial(jax.jit, static_argnums=(3, 6, 7, 8))
def linear_forward(params, params2, state, net_fn, rng, images, is_training = False, centering = False, return_components = False):
    dparams = _sub(params2, params)
    f_0, df_0, state = jax.jvp(lambda param: net_fn(param, state, rng, images, is_training = is_training), (params,), (dparams,), has_aux = True)
    if return_components:
        if centering:
            return df_0, {'state': state, 'f': f_0, 'df': df_0}
        return _add(f_0, df_0), {'state': state, 'f': f_0, 'df': df_0}
    
    if centering:
        return df_0, state
    return _add(f_0, df_0), state

def get_resnet(n_classes):
    def _forward_resnet18(x, is_training):
        #use a 3x3 kernel size with stride 1 for the first layer because we are using 32x32 images
        net = modified_resnets.ResNet18(n_classes, initial_conv_config = {'kernel_shape': 3, 'stride': 1})
        return net(x, is_training)
    
    net_forward = hk.transform_with_state(_forward_resnet18)

    return net_forward.init, net_forward.apply

def _forward_narrow_mlp(x, is_training):
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)

def _forward_wide_mlp(x, is_training):
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(2048), jax.nn.relu,
        hk.Linear(2048), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)

def get_narrow_mlp():
    net_forward = hk.transform_with_state(_forward_narrow_mlp)

    return net_forward.init, net_forward.apply

def get_wide_mlp():
    net_forward = hk.transform_with_state(_forward_wide_mlp)

    return net_forward.init, net_forward.apply


def get_model(model_name, n_classes):
    if model_name == 'resnet18':
        return get_resnet(n_classes)
    elif model_name == 'mlp_skinny':
        return get_narrow_mlp()
    elif model_name == 'mlp_wide':
        return get_wide_mlp()
    else:
        print("Invalid model: {}".format(model_name))
        
        sys.exit()