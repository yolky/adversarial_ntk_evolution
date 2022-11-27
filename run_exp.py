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
import data
import os

import argparse

@functools.partial(jax.jit, static_argnums=(3,5, 9, 10))
def do_training_step(params, lin_params, net_state, net_fn, opt_state, optimizer_update, rng, images, labels, is_training = True, centering = False):
    [loss,lf_dict], grads = jax.value_and_grad(loss_fn, has_aux = True)(params, lin_params, net_state, net_fn, rng, images, labels, lin = False, is_training = is_training, centering = centering)
    net_state = lf_dict['net_state']
    acc = lf_dict['acc']

    updates, opt_state = optimizer_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, net_state, opt_state, acc

@functools.partial(jax.jit, static_argnums=(3, 5, 9, 10))
def do_training_step_linear(params, lin_params, net_state, net_fn, opt_state_lin, optimizer_lin_update, rng, images, labels, centering = False, is_training = False):
    [loss, lf_dict], grads = jax.value_and_grad(loss_fn, has_aux = True, argnums = 1)(params, lin_params, net_state, net_fn, rng, images, labels, lin = True, centering = centering, is_training = is_training)
    net_state = lf_dict['net_state']
    acc = lf_dict['acc']

    updates, opt_state_lin = optimizer_lin_update(grads, opt_state_lin, lin_params)
    lin_params = optax.apply_updates(lin_params, updates)

    return loss, params, lin_params, net_state, opt_state_lin, acc

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default = './X_nothing', help = 'data path')
    
    parser.add_argument('--standard_epochs', type=int, default = 100, help = 'number of epochs to run in standard dynamics before switching to phase 2')
    parser.add_argument('--linear_epochs', type=int, default = 100, help = 'number of epochs to run in stage 2 dynamics')
    
    parser.add_argument('--centering', action='store_true', help = 'whether to use centered linearized dynamics for phase 2. default is non-centering')
    parser.add_argument('--save_models', action='store_true', help = 'whether to save the models at the end of phase 1/2')
    parser.add_argument('--constant_save', action='store_true', help = 'whether to save after every epoch in phase 1')
    parser.add_argument('--constant_save_linear', action='store_true', help = 'whether to save after every epoch in phase 2')
    parser.add_argument('--loose_bn_second', action='store_true', help = 'whether to allow batch norm parameters to change in the second phase, default is frozen batch norm')
    
    parser.add_argument('--do_standard_second', action='store_true', help = 'whether to use standard dynamics in phase 2')
    parser.add_argument('--skip_first_test', action='store_true', help = 'whether to skip evaluation after phase 1')
    
    parser.add_argument('--skip_second_test', action='store_true', help = 'whether to skip evaoluation after phase 2')
    parser.add_argument('--random_seed', type = int, default = 0, help = 'random seed')
    
    parser.add_argument('--base_model_path', type=str, default = '', help = 'if this is non-empty, we load a model from the path and then skip to phase 2')
    parser.add_argument('--model', type=str, default = 'resnet18', help = 'model. all experiments in the paper use a resnet 18')
    
    parser.add_argument('--loaders', type=str, default = 'CC', help = 'first letter is what type of training in phase 1 and second letter is type of training in phase 2. C = benign/clean data. A = adversarial training. F ="flip" i.e. flip from clean data to adversarial after 50 epochs')
    parser.add_argument('--dataset', type=str, default = help = 'cifar10', 'dataset. either cifar10 or cifar100')
    parser.add_argument('--second_lr', type=float, default = 0.01, help = 'learning rate to use in phase 2')
    parser.add_argument('--eps', type=float, default = 4.00, help = 'eps value for adversarial training. scaled by 1/255')
    parser.add_argument('--save_path', type=str, default = './saved_models/', help = 'save path for the models')

    args = parser.parse_args()
    
    
    if args.save_models:
        os.makedirs(args.save_path, exist_ok=True)
    
    
    transform_train = transforms.Compose([
    transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    constant_save_extra_epochs = [0.125, 0.25, 0.375, 0.5, 0.75, 1.5, 2.5]


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


    train_data, train_labels = data.get_data_and_labels(args.dataset)
    n_classes = data.get_n_classes(args.dataset)
        

    train_dataset = TensorDataset(train_data, train_labels, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    test_loader = data.get_loader(args.dataset, train = False, batch_size = 100, shuffle = False)
    
    
    loader_order = args.loaders

    standard_loader = train_loader
    linear_loader = train_loader
    
    rng = jax.random.PRNGKey(args.random_seed)
    
    print("RANDOM SEED {}".format(args.random_seed))
    
    net_forward_init, net_forward_apply = models.get_model(args.model, n_classes)

    dummy_images, dummy_labels = next(iter(train_loader))
    dummy_images = np.transpose(dummy_images.cpu().numpy(), [0,2,3,1])
    dummy_labels = dummy_labels.cpu().numpy()
    

    params, net_state = net_forward_init(rng, dummy_images, is_training=True)
    lin_params = copy.deepcopy(params)
    

    optimizer_init, optimizer_update = optax.chain( optax.sgd(0.1, momentum = 0.9))
    opt_state = optimizer_init(params)
    
    
    if len(args.base_model_path) > 0:
        print('Loading from saved model')
        checkpoint = pickle.load(open('./{}'.format(args.base_model_path), 'rb'))
        params = checkpoint['params']
        lin_params = checkpoint['lin_params']
        net_state = checkpoint['net_state']
        
        optimizer_init, optimizer_update = optax.chain( optax.sgd(args.second_lr, momentum = 0.9))
        opt_state = optimizer_init(params)
    
    else:
        losses = []

        for epoch in range(args.standard_epochs):
            print(epoch)

            if args.constant_save:
                pickle.dump({'params' : params, 'lin_params' : lin_params, 'net_state' : net_state}, open('./{}/parameters_checkpoint_{}.pkl'.format(args.save_path, epoch),'wb'))

            optim_step = 0
            for i, (images, labels) in enumerate(standard_loader):
                
                if args.constant_save and len(constant_save_extra_epochs) > 0 and (epoch + (i/len(standard_loader))) > constant_save_extra_epochs[0]:
                    pickle.dump({'params' : params, 'lin_params' : lin_params, 'net_state' : net_state}, open('./{}/parameters_checkpoint_{}.pkl'.format(args.save_path, constant_save_extra_epochs[0]),'wb'))
                    constant_save_extra_epochs.pop(0)

                images = np.transpose(images.cpu().numpy(), [0,2,3, 1])
                labels = labels.cpu().numpy()

                loss, params, net_state, opt_state, acc = do_training_step(params, lin_params, net_state, net_forward_apply, opt_state, optimizer_update, rng, images, labels)
                if loader_order[0] in ['A'] or (loader_order[0] == 'F' and epoch >= 50):
                    adv_eps = args.eps/255
                    iters = 20
                    adv_1 = perturb(params, lin_params, net_state, net_forward_apply, rng, images, labels, adv_eps, 2 * adv_eps/iters, iters)
                    loss, params, net_state, opt_state, acc = do_training_step(params, lin_params, net_state, net_forward_apply, opt_state, optimizer_update, rng, adv_1, labels)
                optim_step += 1
                losses.append(loss)

            if epoch == 99:
                _, optimizer_update = optax.chain( optax.sgd(0.01, momentum = 0.9))
            elif epoch == 149:
                _, optimizer_update = optax.chain( optax.sgd(0.001, momentum = 0.9))
    
    
    if args.skip_first_test:
        clean_acc_l2, dirty_acc_l2 = [0, 0]
        clean_acc_linf, dirty_acc_linf = [0, 0]
    else:
        clean_acc_l2, dirty_acc_l2 = test(params, lin_params, net_state, net_forward_apply, rng, test_loader, make_adv_examples = True, linear = False, attack = 'l2', adv_eps = args.eps)
        clean_acc_linf, dirty_acc_linf = test(params, lin_params, net_state, net_forward_apply, rng, test_loader, make_adv_examples = True, linear = False, attack = 'linf', adv_eps = args.eps)
    standard_results = {
        'clean': clean_acc_l2,
        'l2': dirty_acc_l2,
        'linf': dirty_acc_linf
    }
    
    lin_params = copy.deepcopy(params)
    optimizer_lin_init, optimizer_lin_update = optax.chain( optax.sgd(args.second_lr, momentum = 0.9))
    opt_state_lin = optimizer_lin_init(lin_params)
    
    losses = []

    for epoch in range(args.linear_epochs):
        print(epoch)
        optim_step = 0
        for i, (images, labels) in enumerate(linear_loader):
            images = np.transpose(images.cpu().numpy(), [0,2,3, 1])
            labels = labels.cpu().numpy()
            
            if args.constant_save_linear:
                pickle.dump({'params' : params, 'lin_params' : lin_params, 'net_state' : net_state}, open('./{}/parameters_checkpoint_linear_{}.pkl'.format(args.save_path, epoch),'wb'))
            
            if args.do_standard_second:
                loss, params, net_state, opt_state, acc = do_training_step(params, lin_params, net_state, net_forward_apply, opt_state, optimizer_update, rng, images, labels, is_training = args.loose_bn_second, centering = args.centering)
            
            else:
                loss, params, lin_params, net_state, opt_state_lin, acc = do_training_step_linear(params, lin_params, net_state, net_forward_apply, opt_state_lin, optimizer_lin_update, rng, images, labels, centering = args.centering, is_training = args.loose_bn_second)
                
                if loader_order[1] in ['A'] or (loader_order[1] == 'F' and epoch >= 50):
                    adv_eps = args.eps/255
                    iters = 20
                    adv_1 = perturb(params, lin_params, net_state, net_forward_apply, rng, images, labels, adv_eps, 2 * adv_eps/iters, iters, linear = True, centering = args.centering)
                    loss, params, lin_params, net_state, opt_state_lin, acc = do_training_step_linear(params, lin_params, net_state, net_forward_apply, opt_state_lin, optimizer_lin_update, rng, adv_1, labels, centering = args.centering)

            optim_step += 1
            losses.append(loss)
                

    #note we test l2 norms in the code but in the paper we only use l-inf attacks/defenses

    if args.skip_second_test:
        clean_acc_l2, dirty_acc_l2 = [0, 0]
        clean_acc_linf, dirty_acc_linf = [0, 0]
    elif args.do_standard_second:
        clean_acc_l2, dirty_acc_l2 = test(params, lin_params, net_state, net_forward_apply, rng, test_loader, make_adv_examples = True, linear = False, centering = args.centering, attack = 'l2', adv_eps = args.eps)
        clean_acc_linf, dirty_acc_linf = test(params, lin_params, net_state, net_forward_apply, rng, test_loader, make_adv_examples = True, linear = False, centering = args.centering, attack = 'linf', adv_eps = args.eps)
    else:
        clean_acc_l2, dirty_acc_l2 = test(params, lin_params, net_state, net_forward_apply, rng, test_loader, make_adv_examples = True, linear = True, centering = args.centering, attack = 'l2', adv_eps = args.eps)
        clean_acc_linf, dirty_acc_linf = test(params, lin_params, net_state, net_forward_apply, rng, test_loader, make_adv_examples = True, linear = True, centering = args.centering, attack = 'linf', adv_eps = args.eps)
    linear_results = {
        'clean': clean_acc_l2,
        'l2': dirty_acc_l2,
        'linf': dirty_acc_linf
    }
    
    
    if args.save_models:
        pickle.dump({'params' : params, 'lin_params' : lin_params, 'net_state' : net_state}, open('./{}/parameters_final.pkl'.format(args.save_path),'wb'))
        pickle.dump({'standard': standard_results, 'linear': linear_results, 'standard_second': args.do_standard_second}, open('./{}/results.pkl'.format(args.save_path),'wb'))
        
if __name__ == '__main__':
    main()
    