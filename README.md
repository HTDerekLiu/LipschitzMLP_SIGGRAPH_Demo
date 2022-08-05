# LipschitzMLP_SIGGRAPH_Demo
This repository contains the homework exercise for "Learning Smooth Neural Functions via Lipschitz Regularization" by Liu et al. at SIGGRAPH 2022. For more information about the paper, please visit our project page [here](https://nv-tlabs.github.io/lip-mlp/).

## Dependencies
This library depends on [JAX](https://github.com/google/jax) and some common python libraries such as [numpy](https://numpy.org), [matplotlib](https://matplotlib.org/stable/), [tqdm](https://tqdm.github.io), [ffmpeg](https://tqdm.github.io).

## Getting Started
To train a standard MLP to fit signed distance functions, you can simply type
```
python main_mlp.py
```
This will outputs a video `mlp_interpolation.mp4` showing the (non-smooth) interpolation results from a standard MLP.

To see the difference after adding the Lipschitz regularization, you can run
```
python main_lipmlp.py
```
which will outputs a `lipschitz_mlp_interpolation.mp4` showing smoother interpolation after a few minutes of training on CPU. 

