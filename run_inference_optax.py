"""
This script performs deterministic optimization using different optimization algorithms on a cardiovascular model.
It uses the JAX and Optax libraries to perform gradient-based optimization on the model parameters.

Key functionalities include:
- Configuring and running simulations with JAX's Just-In-Time (JIT) compilation for performance.
- Defining a deterministic optimization process to find the best-fit model parameters.
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
- It sets up a simulation environment, runs the simulation, and then performs deterministic optimization using Optax.
- Results are saved to a specified directory for further analysis.
"""

import itertools
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from flax.training.train_state import TrainState
from jax import jit

from src.model import config_simulation, simulation_loop_unsafe

# Change directory to the script's location
os.chdir(os.path.dirname(__file__))

# Enable 64-bit precision in JAX for higher accuracy in numerical computations
jax.config.update("jax_enable_x64", True)


# Set the configuration filename based on command line arguments or default to a specific model
CONFIG_FILENAME = ""
if len(sys.argv) == 1:

    MODELNAME = "test/adan56/adan56.yml"

    CONFIG_FILENAME = "test/" + MODELNAME + "/" + MODELNAME + ".yml"

else:
    CONFIG_FILENAME = "test/" + sys.argv[1] + "/" + sys.argv[1] + ".yml"


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
    upper=120000,
)

# Adjust the CCFL value for the next stage of simulation
CCFL = 0.5

# Select specific indices and scales for the optimization process
R_INDEX = 1
VAR_INDEX = 7
R1 = sim_dat_const[VAR_INDEX, strides[R_INDEX, 1]]
R_scales = np.linspace(0.1, 10, 8)


def sim_loop_wrapper(params):
    """
    Wrapper function for running the simulation loop with a modified parameter value.

    Args:
        params (Array): Array containing scaling factor for the selected simulation constant.

    Returns:
        Array: Pressure values from the simulation with the modified parameter.
    """
    r = params[0] * R1
    ones = jnp.ones(strides[R_INDEX, 1] - strides[R_INDEX, 0] + 4)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[
        VAR_INDEX, strides[R_INDEX, 0] - 2 : strides[R_INDEX, 1] + 2
    ].set(r * ones)
    _, _, p = SIM_LOOP_JIT(  # pylint: disable=E1102
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
        upper=120000,
    )
    return p


# Define the folder to save the optimization results
RESULTS_FOLDER = "results/inference_ensemble_det"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)


# Define the hyperparameters for the network properties
network_properties = {
    # "tx": [optax.adam, optax.sgd, optax.lars],
    # "tx": [optax.adabelief, optax.adadelta, optax.adafactor, optax.adagrad,
    #       optax.adamw, optax.adamax, optax.adamaxw, optax.amsgrad],
    # "tx": [optax.adagrad,
    #       optax.adamw, optax.adamax, optax.adamaxw, optax.amsgrad],
    "tx": [optax.adabelief],
    "learning_rate": [1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
    "epochs": [2000],
}

# Create a list of all possible combinations of the network properties
settings = list(itertools.product(*network_properties.values()))

# Define the folder to save the optimization results
RESULTS_FOLDER = "results/inference_ensemble_sgd"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)

# Loop through each combination of settings and run the deterministic optimization
for set_num, setup in enumerate(settings):
    print(
        "###################################",
        set_num,
        "###################################",
    )
    RESULTS_FILE = (
        RESULTS_FOLDER
        + "/setup_"
        + str(setup[0].__name__)
        + "_"
        + str(setup[1])
        + "_"
        + str(setup[2])
        + "_test.txt"
    )

    # Define the model and variables for optimization
    model = sim_loop_wrapper
    variables = [R_scales[int(sys.argv[2])]]
    tx = setup[0]
    y = P_obs
    x = sim_loop_wrapper

    # Initialize the training state with the optimizer and initial variables
    state = TrainState.create(apply_fn=model, params=variables, tx=tx(setup[1]))

    def loss_fn(prediction, target):
        """
        Compute the loss based on the difference between predictions and target values.

        Args:
            prediction (Array): Model predictions.
            target (Array): Observed target values.

        Returns:
            float: Computed loss value.
        """
        loss = jnp.log(
            optax.l2_loss(predictions=prediction, targets=target).mean()
            / optax.l2_loss(predictions=np.ones_like(target), targets=target).mean()
            + 1
        )
        return loss

    # Run the optimization loop for the specified number of epochs
    for _ in range(setup[2]):
        grads = jax.jacfwd(loss_fn)(x, y)
        # print(grads)
        state = state.apply_gradients(grads=grads)
        # print(state)
        print(loss_fn(x, y))

    # Save the results of the optimization to a file
    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        file.write(
            str(R_scales[int(sys.argv[2])])
            + " "
            + str(state.params[0])
            + "  "
            + str(R1)
            + "\n"
        )
