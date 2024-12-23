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
import scienceplots

plt.style.use("science")

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

UPPER = 1000

# Set up and execute the simulation loop using JIT compilation
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)

# Indices for selecting specific parts of the simulation data
VESSEL_INDEX_1 = 1
VAR_INDEX_1 = 4

# Extract a specific value from the simulation data constants
R1_1 = sim_dat_const_aux[VESSEL_INDEX_1, VAR_INDEX_1]


@compact
def sim_loop_wrapper(params, upper=UPPER):
    """
    Wrapper function for running the simulation loop with a modified parameter value.

    Args:
        params (Array): Array containing scaling factor for the selected simulation constant.

    Returns:
        Array: Pressure values from the simulation with the modified parameter.
    """
    r = params * R1_1 * 2
    sim_dat_const_aux_new = jnp.array(sim_dat_const_aux)
    sim_dat_const_aux_new = sim_dat_const_aux_new.at[VESSEL_INDEX_1, VAR_INDEX_1].set(r)
    sim_dat_wrapped, t_wrapped, p_wrapped = SIM_LOOP_JIT(  # pylint: disable=E1102
        N,
        B,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux_new,
        CCFL,
        input_data,
        rho,
        masks,
        strides,
        edges,
        upper=upper,
    )
    return sim_dat_wrapped, t_wrapped, p_wrapped


sim_dat_obs, t_obs, P_obs = sim_loop_wrapper(0.5)
sim_dat_obs_long, t_obs_long, P_obs_long = sim_loop_wrapper(0.5, 120000)

# Define the folder to save the optimization results
RESULTS_FOLDER = "results/inference_optax_1"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)


class SimDense(Module):
    features: int
    kernel_init: Callable[[jax.random.PRNGKey, tuple, jnp.dtype], jnp.ndarray] = (
        uniform(2.0)
    )

    @compact
    def __call__(self) -> jnp.ndarray:
        R1 = self.param(
            "R1",
            self.kernel_init,
            (),
        )

        _, _, y = sim_loop_wrapper(jax.nn.softplus(R1))
        return y


class Loss(object):
    def __init__(self, axis=0, order=None):
        super(Loss, self).__init__()
        self.axis = axis
        self.order = order

    def relative_loss(self, s, s_pred):
        return jnp.log(
            jnp.mean(
                jnp.power(
                    jnp.linalg.norm(
                        s_pred[:, [-1, -6]] - s[:, [-1, -6]], ord=None, axis=self.axis
                    ),
                    2,
                )
                / jnp.power(
                    jnp.linalg.norm(s[:, [-1, -6]], ord=None, axis=self.axis), 2
                )
            )
        )

    def __call__(self, s, s_pred):
        return self.relative_loss(s, s_pred)


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
    params = jax.nn.softplus(state.params["params"]["R1"])
    plt.figure()
    _, t, p = sim_loop_wrapper(params)
    loss_val = loss(p, P_obs)
    _, t, p = sim_loop_wrapper(params, upper=120000)
    sorted_indices = np.argsort(t_obs_long[-12000:])
    plt.plot(
        t_obs_long[-12000:][sorted_indices],
        P_obs_long[-12000:, -8][sorted_indices] / 133.322,
        label="ground truth",
        linewidth=0.5,
    )
    sorted_indices = np.argsort(t[-12000:])
    plt.plot(
        t[-12000:][sorted_indices],
        p[-12000:, -8][sorted_indices] / 133.322,
        label="learned",
        linewidth=0.5,
    )
    lgnd = plt.legend(loc="upper right")
    lgnd.legend_handles[0]._sizes = [30]
    lgnd.legend_handles[1]._sizes = [30]
    plt.xlabel("t/T")
    plt.ylabel("P [mmHg]")
    plt.xlim([0.0, 1.0])
    plt.ylim([30, 140])
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/{str(0)}_nt.pdf")
    plt.savefig(f"{RESULTS_FOLDER}/{str(0)}_nt.png")
    plt.savefig(f"{RESULTS_FOLDER}/{str(0)}_nt.jpeg")
    plt.savefig(f"{RESULTS_FOLDER}/{str(0)}_nt.eps")
    plt.title(
        f"learning scaled Windkessel resistance parameter of a bifurcation:\n[R1_1, R2_1, R1_2, R2_2] =\n{params},\nloss = {loss_val}, \n wall-clock time = {0.0}"
    )
    plt.savefig(f"{RESULTS_FOLDER}/{str(0)}.pdf")
    plt.savefig(f"{RESULTS_FOLDER}/{str(0)}.png")
    plt.savefig(f"{RESULTS_FOLDER}/{str(0)}.jpeg")
    plt.savefig(f"{RESULTS_FOLDER}/{str(0)}.eps")
    plt.close()
    total_time = 0
    for epoch in bar:
        starting_time = time.time_ns()
        state, loss_val = train_step(state, batch)
        total_time += time.time_ns() - starting_time
        params = jax.nn.softplus(state.params["params"]["R1"])
        bar.set_description(
            f"Loss: {loss_val}, Parameters {jax.nn.softplus(state.params["params"]["R1"])}"
        )
        # if loss < 1e-6:
        #     break
        plt.figure()
        _, t, p = sim_loop_wrapper(params, 120000)
        sorted_indices = np.argsort(t_obs_long[-12000:])
        plt.plot(
            t_obs_long[-12000:][sorted_indices],
            P_obs_long[-12000:, -8][sorted_indices] / 133.322,
            label="ground truth",
            linewidth=0.5,
        )
        sorted_indices = np.argsort(t[-12000:])
        plt.plot(
            t[-12000:][sorted_indices],
            p[-12000:, -8][sorted_indices] / 133.322,
            label="learned",
            linewidth=0.5,
        )
        lgnd = plt.legend(loc="upper right")
        lgnd.legend_handles[0]._sizes = [30]
        lgnd.legend_handles[1]._sizes = [30]
        plt.xlabel("t/T")
        plt.ylabel("P [mmHg]")
        plt.xlim([0.0, 1.0])
        plt.ylim([30, 140])
        plt.tight_layout()
        plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}_nt.pdf")
        plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}_nt.png")
        plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}_nt.jpeg")
        plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}_nt.eps")
        plt.title(
            f"learning scaled Windkessel resistance parameters of a bifurcation:\nR1 = {params},\nloss = {loss_val}, \n wall-clock time = {total_time / 1e9}"
        )
        plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}.pdf")
        plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}.png")
        plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}.jpeg")
        plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}.eps")
        plt.close()
    return state


print("Model Initialized")
lr = 1e-1
transition_steps = 1
decay_rate = 0.9
weight_decay = 0
seed = 0
epochs = 5000

model = SimDense(features=4)

params = model.init(random.key(2))

print("Initial Parameters: ", jax.nn.softplus(params["params"]["R1"]))

exponential_decay_scheduler = optax.exponential_decay(
    init_value=lr,
    transition_steps=transition_steps,
    decay_rate=decay_rate,
    transition_begin=0,
    staircase=False,
)

optimizer = optax.adafactor(lr)

model_state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer
)

trained_model_state = train_model(model_state, P_obs, num_epochs=epochs)

y = model_state.apply_fn(trained_model_state.params)

print(
    f"Final Loss: {loss(P_obs, y)} and Parameters: {jax.nn.softplus(trained_model_state.params["params"]["R1"])}"
)

plt.figure()
plt.plot(t_obs, P_obs, "b-", label="Baseline")
plt.plot(t_obs, y, "r--", label="Predicted")
plt.legend()
plt.show()
