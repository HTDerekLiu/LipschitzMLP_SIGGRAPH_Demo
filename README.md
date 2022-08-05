# Lipschitz Multilayer Perceptron SIGGRAPH Demo
This repository contains the homework exercise for "Learning Smooth Neural Functions via Lipschitz Regularization" by Liu et al. at SIGGRAPH 2022. For more information about the paper, please visit our project page [here](https://nv-tlabs.github.io/lip-mlp/).

## Dependencies
This library depends on [JAX](https://github.com/google/jax) and some common python libraries such as [numpy](https://numpy.org), [matplotlib](https://matplotlib.org/stable/), [tqdm](https://tqdm.github.io), [ffmpeg](https://tqdm.github.io).

## Repository Structure
This repository contains two executable scripts `main_mlp.py` for training a standard MLP and a `main_lipmlp.py` to train our Lipschitz MLP. The network architectures are implemented in `model_mlp.py` and `model_lipmlp.py`, respectively. However, the current `model_lipmlp.py` is incomplete, we have removed the key parts for implementing our Lipschitz MLP. Your goal is to implement these key functions so that you will obtain the reference results as we showed in the `solution` folder.

## Homework Exercise
In this exercise, your goal is to implement two key functions to turn a traditional MLP into a Lipschitz MLP. These two functions are the `weight_normalization` and `get_lipschitz_loss` listed in the `model_lipmlp.py` file. 

The `weight_normalization` function normalizes the weight matrices so that the matrix p-norm is bounded by the learned per-layer Lipschitz constant `softplus_c`. You can find more details in the Section 4.1.1 in our paper. 

The `get_lipschitz_loss` function computes the Lipschitz regularization from the per-layer Lipschitz constants. Please see more details in the Section 4.1.2 in our paper. 

Once you complete the above exercise, you can start training the model by typing 
```
python main_lipmlp.py
```
in the terminal. This will output a `lipschitz_mlp_interpolation.mp4` showing your interpolation results. This may take ~10 minutes on a single CPU. 
