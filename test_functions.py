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
from models import linear_forward


@functools.partial(jax.jit, static_argnums=(3, 7, 8, 9))
def loss_fn(params, lin_params, state, net_fn, rng, images, labels, lin = False, is_training = True, centering = False):
    if not lin:
        if centering:
            #use linear params as 0 parameters if centering
            logits0, state = net_fn(lin_params, state, rng, images, is_training = is_training)
            logits1, state = net_fn(params, state, rng, images, is_training = is_training)
            logits = logits1 - logits0
        else:
            logits, state = net_fn(params, state, rng, images, is_training = is_training)
    else:
        logits, state = linear_forward(params, lin_params, state, net_fn, rng, images, is_training = is_training, centering = centering)
        
    labels_oh = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits,labels_oh).mean()
    acc = jnp.mean(logits.argmax(1) == labels)
    return loss, {'net_state': state, 'acc': acc}

@functools.partial(jax.jit, static_argnums=(2))
def clamp_by_norm(x, r, norm = 'l_2'):
    if norm == 'l_2':
        norms = jnp.sqrt(jnp.sum(x ** 2, [1,2,3], keepdims = True))
        factor = jnp.minimum(r/norms, jnp.ones_like(norms))
        return x * factor
    elif norm == 'l_inf':
        return jnp.clip(x, -1 * r, r)

@functools.partial(jax.jit, static_argnums=(3, 10, 11))
def do_perturbation_step_l_inf(params, lin_params, net_state, net_fn, rng, images0, images, labels, eps, alpha, linear = False, centering = False):
    grads, _ = jax.grad(loss_fn, has_aux = True, argnums = 5)(params, lin_params, net_state, net_fn, rng, images, labels, lin = linear, is_training = False, centering = centering)
    grads = jnp.sign(grads)
    images = images + alpha * grads

    images = jnp.clip(images, 0., 1.)

    d_images = images - images0
    d_images = clamp_by_norm(d_images, eps, norm = 'l_inf')
    
    images = images0 + d_images
    
    return images

@functools.partial(jax.jit, static_argnums=(3,10,11))
def do_perturbation_step_l_2(params, lin_params, net_state, net_fn, rng, images0, images, labels, eps, alpha, linear = False, centering = False):
    grads, _ = jax.grad(loss_fn, has_aux = True, argnums = 5)(params, lin_params, net_state, net_fn, rng, images, labels, lin = linear, is_training = False, centering = centering)
    grads = grads/jnp.sqrt(jnp.sum(grads ** 2, [1,2,3], keepdims = True))
    images = images + alpha * grads

    images = jnp.clip(images, 0., 1.)

    d_images = images - images0
    d_images = clamp_by_norm(d_images, eps, norm = 'l_2')

    images = images0 + d_images
    
    return images



def perturb(params, lin_params, net_state, net_fn, rng, images0, labels, eps, alpha, iters, linear = False, centering = False, attack = 'linf'):
    images = images0
    
    #First add random noise within ball
    if attack == 'l2':
        images = images + np.random.normal(0, eps/np.sqrt(len(images[0].shape)), images.shape)
    if attack == 'linf':
        images = images + np.random.uniform(-eps, eps, images.shape)
    
    images = jnp.clip(images, 0., 1.)
        
    d_images = images - images0
    d_images = clamp_by_norm(d_images, eps)
    images = images + d_images
    for i in range(iters):
        if attack == 'linf':
            images = do_perturbation_step_l_inf(params, lin_params, net_state, net_fn, rng, images0, images, labels, eps, alpha, linear = linear, centering = centering)
        elif attack == 'l2':
            images = do_perturbation_step_l_2(params, lin_params, net_state, net_fn, rng, images0, images, labels, eps, alpha, linear = linear, centering = centering)
        
    return images

def test(params, lin_params, state, net_fn, rng, test_loader, linear = False, make_adv_examples = False, centering = False, attack = 'linf', return_examples = False, short = False, return_components = False, adv_eps = 4):
    adv_eps = adv_eps/255

    n_correct = 0
    n_total = 0
    
    n_correct_adv = 0
    
    n_correct_batch = 0
    n_batch = 0
    n_correct_adv_batch = 0
    
    adv_examples = []
    predictions = []
    
    components = []
    linear_components = []
    
    adv_components = []
    adv_linear_components = []
    
    adv_predictions = []
    
    print("testing")
    
    for i, (images, labels) in enumerate(test_loader):
        images = np.array(np.transpose(images.cpu().numpy(), [0,2,3,1]))
        labels = labels.cpu().numpy()
        
        if linear:
            logits, return_dict = linear_forward(params, lin_params, state, net_fn, rng, images, is_training = False, centering = centering, return_components = True)
            f = return_dict['f']
            df = return_dict['df']
            
            components.append(f)
            linear_components.append(df)
        else:
            if centering:
                logits0, _ = net_fn(lin_params, state, rng, images, is_training = False)
                logits1, _ = net_fn(params, state, rng, images, is_training = False)
                logits = logits1 - logits0
                
                components.append(logits0)
                linear_components.append(np.zeros_like(logits))
            else:
                logits, _ = net_fn(params, state, rng, images, is_training = False)
                components.append(logits)
                linear_components.append(np.zeros_like(logits))
            
        n_correct += np.sum(logits.argmax(1) == labels)
        n_correct_batch += np.sum(logits.argmax(1) == labels)
        
        predictions.append(logits.argmax(1))
        
        n_total += len(labels)
        n_batch += len(labels)
        
        
        if make_adv_examples:
            iters = 100
            if attack == 'l2':
                adv_images = perturb(params, lin_params, state, net_fn, rng, images, labels, 0.25, 0.01, iters, linear = linear, centering = centering, attack = attack)
            elif attack == 'linf':
                adv_images = perturb(params, lin_params, state, net_fn, rng, images, labels, adv_eps, 2 * adv_eps / iters, iters, linear = linear, centering = centering, attack = attack)
        else:
            adv_images = images
            
        if return_examples:
            adv_examples.append(adv_images)
        
        if linear:
            logits_adv, return_dict = linear_forward(params, lin_params, state, net_fn, rng, adv_images, is_training = False, centering = centering, return_components = True)
            
            f = return_dict['f']
            df = return_dict['df']
            
            adv_components.append(f)
            adv_linear_components.append(df)
        else:
            if centering:
                logits0, _ = net_fn(lin_params, state, rng, adv_images, is_training = False)
                logits1, _ = net_fn(params, state, rng, adv_images, is_training = False)
                logits_adv = logits1 - logits0
                
                adv_components.append(logits0)
                adv_linear_components.append(np.zeros_like(logits_adv))
            else:
                logits_adv, _ = net_fn(params, state, rng, adv_images, is_training = False)
                adv_components.append(logits_adv)
                adv_linear_components.append(np.zeros_like(logits_adv))
            
        n_correct_adv += np.sum(logits_adv.argmax(1) == labels)
        n_correct_adv_batch += np.sum(logits_adv.argmax(1) == labels)
        
        adv_predictions.append(logits_adv.argmax(1))
        
        if i % 10 == 9:
            print("\nTest Batch {}".format(int((i+1)/10)))
            print("Clean Acc: {:.2f}".format(100 * n_correct_batch/n_batch))
            print("Dirty Acc: {:.2f}".format(100 * n_correct_adv_batch/n_batch))
            
            n_correct_batch = 0
            n_batch = 0
            n_correct_adv_batch = 0
            
            if short:
                break
                
        
        
    print("\nTest Results Total".format(int((i+1)/10)))
    print("Clean Acc: {:.2f}".format(100 * n_correct/n_total))
    print("Dirty Acc: {:.2f}".format(100 * n_correct_adv/n_total))
    
    components_clean = {'f': np.concatenate(components), 'df': np.concatenate(linear_components)}
    components_dirty = {'f': np.concatenate(adv_components), 'df': np.concatenate(adv_linear_components)}
    
    if return_examples:
        adv_examples = np.concatenate(adv_examples, 0)
        predictions = np.concatenate(predictions)
        adv_predictions = np.concatenate(adv_predictions)
        
        if return_components:
            return n_correct/n_total, n_correct_adv/n_total, adv_examples, predictions, adv_predictions, components_clean, components_dirty
            
        return n_correct/n_total, n_correct_adv/n_total, adv_examples, predictions, adv_predictions
    
    
    return n_correct/n_total, n_correct_adv/n_total