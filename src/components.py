from flax import linen as nn

class Blood:
    mu: float
    rho: float
    rho_inv: float
    def __init__(self, mu, rho, rho_inv):
        self.mu = mu
        self.rho = rho
        self.rho_inv = rho_inv

class SimpleClassifierCompact(nn.Module):
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    @nn.compact  # Tells Flax to look for defined submodules
    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        # while defining necessary layers
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x
