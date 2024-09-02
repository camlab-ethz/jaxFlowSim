"""
This script performs optimization using different optimization algorithms on a cardiovascular model.
It uses the JAX and Optax libraries to perform gradient-based optimization on the model parameters.

Key functionalities include:
- Configuring and running simulations with JAX's Just-In-Time (JIT) compilation for performance.
- Defining an optimization process to find the best-fit model parameters.
- Using various Optax optimizers to minimize the loss function over a set number of epochs.
- Saving the results of the optimization process to files.

Modules:
- `itertools`: Provides functions to create iterators for efficient looping.
- `os`, `sys`, `time`: Standard libraries for file handling, system operations, and timing.
- `jax`: A library for high-performance machine learning research.
- `jax.numpy`: JAX's version of NumPy, with support for automatic differentiation and GPU/TPU acceleration.
- `optax`: A library for gradient-based optimization in JAX.
- `flax.training.train_state`: Provides a simple wrapper for managing training state in JAX.

Functions:
- `sim_loop_wrapper`: Wraps the simulation loop to allow parameter modification.
- `loss_fn`: Computes the loss based on the difference between the model's predictions and the observed data.

Execution:
- The script reads command-line arguments to determine which configuration file to use.
- It sets up a simulation environment, runs the simulation, and then performs optimization using optax.
- Results are saved to a specified directory for further analysis.
"""

import os
import sys
import time
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax  # type: ignore
import tqdm
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import uniform
from flax.linen.module import Module, compact
from flax.training import train_state
from jax import jit, random

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe  # noqa=E402

# Change directory to the script's location
os.chdir(os.path.dirname(__file__) + "/..")

# Enable 64-bit precision in JAX for higher accuracy in numerical computations
jax.config.update("jax_enable_x64", True)


# Set the configuration filename
CONFIG_FILENAME = "test/bifurcation/bifurcation.yml"


# Set verbosity flag to control logging
VERBOSE = True

# Configure the simulation with the given configuration file
(
    N,
    B,
    J,
    sim_dat,
    sim_dat_aux,
    sim_dat_const,
    sim_dat_const_aux,
    timepoints,
    conv_tol,
    CCFL,
    input_data,
    rho,
    masks,
    strides,
    edges,
    vessel_names,
    cardiac_T,
) = config_simulation(CONFIG_FILENAME, VERBOSE)


# Record the start time if verbose mode is enabled
if VERBOSE:
    starting_time = time.time_ns()

# Set up and execute the simulation loop using JIT compilation
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)
sim_dat_obs, t_obs, P_obs = SIM_LOOP_JIT(  # pylint: disable=E1102
    N,
    B,
    sim_dat,
    sim_dat_aux,
    sim_dat_const,
    sim_dat_const_aux,
    CCFL,
    input_data,
    rho,
    masks,
    strides,
    edges,
    upper=50000,
)

# Every 10757 the time resets to 0

# Adjust the CCFL value for the next stage of simulation
CCFL = 0.9

# Select specific indices and scales for the optimization process
R1_INDEX = 1
VAR_INDEX = 7
R1 = sim_dat_const[VAR_INDEX, strides[R1_INDEX, 1]]


@compact
def sim_loop_wrapper(params, test=False):
    """
    Wrapper function for running the simulation loop with a modified parameter value.

    Args:
        params (Array): Array containing scaling factor for the selected simulation constant.

    Returns:
        Array: Pressure values from the simulation with the modified parameter.
    """
    r = params[0] * R1
    ones = jnp.ones(strides[R1_INDEX, 1] - strides[R1_INDEX, 0] + 4)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[
        VAR_INDEX, strides[R1_INDEX, 0] - 2 : strides[R1_INDEX, 1] + 2  # noqa=E203
    ].set(r * ones)
    _, t, p = SIM_LOOP_JIT(  # pylint: disable=E1102
        N,
        B,
        sim_dat,
        sim_dat_aux,
        sim_dat_const_new,
        sim_dat_const_aux,
        CCFL,
        input_data,
        rho,
        masks,
        strides,
        edges,
        upper=50000,
    )
    if test:
        return p, t
    else:
        return p


# Define the folder to save the optimization results
RESULTS_FOLDER = "results/inference_ensemble_optax"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)


class SimDense(Module):
    kernel_init: Callable[[jax.random.PRNGKey, tuple, jnp.dtype], jnp.ndarray] = (
        uniform(2.0)
    )

    @compact
    def __call__(self) -> jnp.ndarray:
        R1 = self.param(
            "R1",
            self.kernel_init,
            (1,),
        )

        y = sim_loop_wrapper(jax.nn.softplus(R1))
        return y


class Loss(object):
    def __init__(self, axis=0, order=None):
        super(Loss, self).__init__()
        self.axis = axis
        self.order = order

    def relative_loss(self, s, s_pred):
        return jnp.power(
            jnp.linalg.norm(s_pred - s, ord=None, axis=self.axis), 2
        ) / jnp.power(jnp.linalg.norm(s, ord=None, axis=self.axis), 2)

    def __call__(self, s, s_pred):
        return jnp.mean(self.relative_loss(s, s_pred))


loss = Loss()


def calculate_loss_train(state, params, batch):
    s = batch
    s_pred = state.apply_fn(params)
    loss_value = loss(s, s_pred)
    return loss_value


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss_train, argnums=1)
    loss_value, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss_value


def train_model(state, batch, num_epochs=None):
    bar = tqdm.tqdm(np.arange(num_epochs))
    for _epoch in bar:
        state, loss = train_step(state, batch)
        bar.set_description(f"Loss: {loss}, Parameters {jax.nn.softplus(state.params["params"]["R1"])}")
        if loss < 1e-6:
            break
    return state


print("Model Initialized")
lr = 1e-1
transition_steps = 1
decay_rate = 0.8
weight_decay = 0
seed = 0
epochs = 10

model = SimDense()

params = model.init(random.key(2))

print("Initial Parameters: ", jax.nn.softplus(params["params"]["R1"]))

exponential_decay_scheduler = optax.exponential_decay(
    init_value=lr,
    transition_steps=transition_steps,
    decay_rate=decay_rate,
    transition_begin=0,
    staircase=False,
)

optimizer = optax.adamw(
    learning_rate=exponential_decay_scheduler, weight_decay=weight_decay
)

model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer
)

trained_model_state = train_model(model_state, P_obs, num_epochs=epochs)

y = model_state.apply_fn(trained_model_state.params)

print(f"Final Loss: {loss(P_obs, y)} and Parameters: {jax.nn.softplus(trained_model_state.params["params"]["R1"])}")

plt.figure()
plt.plot(t_obs, P_obs, "b-", label="Baseline")
plt.plot(t_obs, y, "r--", label="Predicted")
plt.legend()
plt.show()
