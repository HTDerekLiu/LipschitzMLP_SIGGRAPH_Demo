import jax
import jax.numpy as np
from jax import jit, value_and_grad
from jax.experimental import optimizers

import numpy as onp
import numpy.random as random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm
import pickle

def sample_2D_grid(resolution, low = 0, high = 1):
  idx = onp.linspace(low,high,num=resolution)
  x, y = onp.meshgrid(idx, idx)
  V = onp.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), 1)
  return np.array(V)

def sdf_star(x, r = 0.22):
    """
    output the signed distance value of a star in 2D
    Inputs
    x: nx2 array of locations
    r: size of the star
    Outputs
    array of signed distance values at x
    Reference:
    https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
    """
    x = onp.array(x)
    kxy = onp.array([-0.5,0.86602540378])
    kyx = onp.array([0.86602540378,-0.5])
    kz = 0.57735026919
    kw = 1.73205080757

    x = onp.abs(x - 0.5)
    x -= 2.0 * onp.minimum(x.dot(kxy), 0.0)[:,None] * kxy[None,:]
    x -= 2.0 * onp.minimum(x.dot(kyx), 0.0)[:,None] * kyx[None,:]
    x[:,0] -= onp.clip(x[:,0],r*kz,r*kw)
    x[:,1] -= r
    length_x = onp.sqrt(onp.sum(x*x, 1))
    return np.array(length_x*onp.sign(x[:,1]))

def sdf_circle(x, r = 0.282, center = np.array([0.5,0.5])):
    """
    output the SDF value of a circle in 2D
    Inputs
    x: nx2 array of locations
    r: radius of the circle
    center: center point of the circle
    Outputs
    array of signed distance values at x
    """
    dx = x - center
    return np.sqrt(np.sum((dx)**2, axis = 1)) - r