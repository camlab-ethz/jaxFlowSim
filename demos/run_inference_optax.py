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

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe

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
    upper=120000,
)

# Adjust the CCFL value for the next stage of simulation
CCFL = 0.5

# Select specific indices and scales for the optimization process
R1_INDEX = 1
VAR_INDEX = 7
R1 = sim_dat_const[VAR_INDEX, strides[R1_INDEX, 1]]

if len(sys.argv) > 1:
    R1_scales = np.linspace(0.1, 10, int(sys.argv[2]))
else:
    R1_scales = np.linspace(0.1, 10, 1)


def sim_loop_wrapper(params):
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
        VAR_INDEX, strides[R1_INDEX, 0] - 2 : strides[R1_INDEX, 1] + 2
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
RESULTS_FOLDER = "results/inference_ensemble_optax"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)


# Define the hyperparameters for the network properties
network_properties = {
    "tx": [
        optax.adam,
        optax.sgd,
        optax.lars,
        optax.adabelief,
        optax.adadelta,
        optax.adafactor,
        optax.adagrad,
        optax.adamw,
        optax.adamax,
        optax.adamaxw,
        optax.amsgrad,
    ],
    "learning_rate": [
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
        1e7,
        1e8,
    ],
    "epochs": [100, 1000, 2000],
}

# Create a list of all possible combinations of the network properties
settings = list(itertools.product(*network_properties.values()))

# Define the folder to save the optimization results
RESULTS_FOLDER = "results/inference_ensemble_sgd"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)

# Loop through each combination of settings and run the optimization
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
    if len(sys.argv) > 1:
        variables = [R1_scales[int(sys.argv[1])]]
    else:
        variables = [R1_scales[0]]

    tx = setup[0]
    y = P_obs
    x = sim_loop_wrapper

    # Initialize the training state with the optimizer and initial variables
    state = TrainState.create(apply_fn=model, params=variables, tx=tx(setup[1]))

    def loss_fn(params, target):
        """
        Compute the loss based on the difference between predictions and target values.

        Args:
            prediction (Array): Model predictions.
            target (Array): Observed target values.

        Returns:
            float: Computed loss value.
        """
        prediction = sim_loop_wrapper(params)
        loss = jnp.log(
            optax.l2_loss(predictions=prediction, targets=target).mean()
            / optax.l2_loss(predictions=np.ones_like(target), targets=target).mean()
            + 1
        )
        return loss

    # Run the optimization loop for the specified number of epochs
    for _ in range(setup[2]):
        grads = jax.jacfwd(loss_fn)(variables, y)
        state = state.apply_gradients(grads=grads)

    # Save the results of the optimization to a file
    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        if len(sys.argv) > 1:
            file.write(
                str(R1_scales[int(sys.argv[1])])
                + " "
                + str(state.params[0])
                + "  "
                + str(R1)
                + "\n"
            )
        else:
            file.write(
                str(R1_scales[0]) + " " + str(state.params[0]) + "  " + str(R1) + "\n"
            )
