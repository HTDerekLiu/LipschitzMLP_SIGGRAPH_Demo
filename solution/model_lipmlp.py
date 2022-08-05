from include import *

class lipmlp:
  def __init__(self, hyperParams):
    self.hyperParams = hyperParams
  
  def initialize_weights(self):
    """
    Initialize the parameters of the Lipschitz mlp

    Inputs
    hyperParams: hyper parameter dictionary

    Outputs
    params_net: parameters of the network (weight, bias, initial lipschitz bound)
    """
    def init_W(size_out, size_in): 
        W = onp.random.randn(size_out, size_in) * onp.sqrt(2 / size_in)
        return np.array(W)
    sizes = self.hyperParams["h_mlp"]
    sizes.insert(0, self.hyperParams["dim_in"] + self.hyperParams["dim_t"])
    sizes.append(self.hyperParams["dim_out"])
    params_net = []    
    for ii in range(len(sizes) - 1):
        W = init_W(sizes[ii+1], sizes[ii]) # initialize weights
        b = np.zeros(sizes[ii+1]) # initialize bias
        c = np.max(np.sum(np.abs(W), axis=1)) # initialize per-layer lipschitz constant (as the Lipschiz constant of the initial weights)
        params_net.append([W, b, c])
    return params_net

  def weight_normalization(self, W, softplus_c):
    """
    Lipschitz weight normalization based on the L-infinity norm (see Eq.9 in [Liu et al 2022])
    """
    absrowsum = np.sum(np.abs(W), axis=1)
    scale = np.minimum(1.0, softplus_c/absrowsum)
    return W * scale[:,None]

  def forward_single(self, params_net, t, x):
    """
    Forward pass of a lipschitz MLP f(params, t, x)
    
    Inputs
    params_net: parameters of the network
    t: the input feature of the shape
    x: a query location in the space

    Outputs
    out: implicit function value at x
    """
    # concatenate coordinate and latent code
    x = np.append(x, t)

    # forward pass
    for ii in range(len(params_net) - 1):
        W, b, c = params_net[ii]
        W = self.weight_normalization(W, jax.nn.softplus(c)) # weight normalization
        x = jax.nn.relu(np.dot(W, x) + b) # forward pass of a single layer

    # final layer
    W, b, c = params_net[-1]
    W = self.weight_normalization(W, jax.nn.softplus(c)) 
    out = np.dot(W, x) + b
    return out[0]
  forward = jax.vmap(forward_single, in_axes=(None, None, None, 0), out_axes=0)

  def get_lipschitz_loss(self, params_net):
    """
    This function computes the Lipschitz regularization Eq.7 in the [Liu et al 2022] 
    """
    loss_lip = 1.0
    for ii in range(len(params_net)):
      W, b, c = params_net[ii]
      loss_lip = loss_lip * jax.nn.softplus(c) # Lipschitz regularization as the product of c
    return loss_lip