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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
import shutil
import scienceplots

plt.style.use("science")

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe  # noqa=E402
from src.inference import param_inf_optax

# Change directory to the script's location
os.chdir(os.path.dirname(__file__) + "/..")

# Enable 64-bit precision in JAX for higher accuracy in numerical computations
jax.config.update("jax_enable_x64", True)


def main():
    """
    Main function for running the parameter inference using NumPyro.
    """
    # Define the indices of the vessels and variables to select for inference
    VESSEL_INDICES = [1, 1]
    VAR_INDICES = [4, 5]

    # Define the configuration file for the simulation
    CONFIG_FILENAME = "test/bifurcation/bifurcation.yml"

    # Run the parameter inference using NumPyro
    param_inf_optax(VESSEL_INDICES, VAR_INDICES, CONFIG_FILENAME)
    print("Inference complete.")


if __name__ == "__main__":
    main()
