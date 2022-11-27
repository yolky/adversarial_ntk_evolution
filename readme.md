# Evolution of Neural Tangent Kernels under Benign and Adversarial Training

Code for the NeurIPS 2022 paper ["Evolution of Neural Tangent Kernels under Benign and Adversarial Training"](https://arxiv.org/abs/2210.12030)

Contact: [Noel Loo](loo@mit.edu)

# Abstract
Two key challenges facing modern deep learning are mitigating deep networks' vulnerability to adversarial attacks and understanding deep learning's generalization capabilities. Towards the first issue, many defense strategies have been developed, with the most common being Adversarial Training (AT). Towards the second challenge, one of the dominant theories that has emerged is the Neural Tangent Kernel (NTK) -- a characterization of neural network behavior in the infinite-width limit. In this limit, the kernel is frozen, and the underlying feature map is fixed. In finite widths, however, there is evidence that feature learning happens at the earlier stages of the training (kernel learning) before a second phase where the kernel remains fixed (lazy training). While prior work has aimed at studying adversarial vulnerability through the lens of the frozen infinite-width NTK, there is no work that studies the adversarial robustness of the empirical/finite NTK during training. In this work, we perform an empirical study of the evolution of the empirical NTK under standard and adversarial training, aiming to disambiguate the effect of adversarial training on kernel learning and lazy training. We find under adversarial training, the empirical NTK rapidly converges to a different kernel (and feature map) than standard training. This new kernel provides adversarial robustness, even when non-robust training is performed on top of it. Furthermore, we find that adversarial training on top of a fixed kernel can yield a classifier with 76.1% robust accuracy under PGD attacks with ε=4/255 on CIFAR-10.



# Usage

There are 4 main files:
1. `run_exp.py` - This is the main bulk of the experiments. Trains models in phase 1/2 with etiehr standard or linearized Training
2. `eval_and_make_adv.py` - This is for evaluating models trained with run_exp.py and saving the adversarial examples
3. `calculate_kernel.py` - This is for calculating finite-width NTKs for trained models made with run_exp.py
4. `visualize_ntk_features.py` - This is for generating NTK eigenvector visualizations

# Example Usage

## Example 1: Training with benign data for 101 epochs in phase 1 and 0 epochs for phase 2 and save checkpoints every epoch
`python3 run_exp.py --standard_epochs 101 --linear_epochs 0 --loaders CC --save_models --save_path ./saved_model_path_phase_1 --constant_save --random_seed 0`

## Example 2: Training with adversarial data for 101 epochs and 0 in phase 1 epochs for phase 2 and save checkpoints every epoch
`python3 run_exp.py --standard_epochs 101 --linear_epochs 0 --loaders AC --save_models --save_path ./saved_model_path_phase_1 --constant_save --random_seed 0`

## Example 3: Loading a saved model checkpoint after 20 epochs for phase 2 training with centered dynamics for 100 epochs on benign data
`python3 run_exp.py --base_model_path ./saved_model_path_phase_1/parameters_checkpoint_20.pkl --loaders CC --linear_epochs 100 --second_lr 0.01 --save_models --centering --save_path ./saved_model_loaded_centering_phase_2  --random_seed 0`

## Example 4: Loading a saved model checkpoint after 20 epochs for phase 2 training with centered dynamics for 100 epochs on *adversarial* data
`python3 run_exp.py --base_model_path ./saved_model_path_phase_1/parameters_checkpoint_20.pkl --loaders CA --linear_epochs 100 --second_lr 0.01 --save_models --centering --save_path ./saved_model_loaded_centering_phase_2  --random_seed 0`
Note this is adversarial learning on a fixed kernel section of the paper

## Example 5: Loading a network trained with linearized and centered dyanmics for evaluation and save the test results with
`python3 eval_and_make_adv.py --model_path ./saved_model_loaded_centering_phase_2/parameters_final.pkl --linear --centering --save_examples --bonus_dir test_results  --random_seed 0`

## Example 6: Calculating the finite NTK for saved model on 500 cifar10 images on class 0
`python3 calculate_kernel.py --model_path ./saved_model_path_phase_1/parameters_checkpoint_20.pkl --dataset_size 500 --random_seed 0 --class_index 0 --dataset cifar10`

# Requirements
torch, torchvision (for datasets), numpy, seaborn, jax, neural-tangents, dm-haiku, optax

