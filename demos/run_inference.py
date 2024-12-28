"""
This script performs probabilistic inference on a cardiovascular model using MCMC sampling in NumPyro.
The script configures and runs simulations, then uses Bayesian inference to estimate parameters of the model.

Key functionalities include:
- Configuring and running the simulation loop using JAX with JIT compilation for performance.
- Defining a probabilistic model for the parameter inference.
- Running MCMC to sample from the posterior distribution of model parameters.
- Saving the results of the inference to files.

Modules:
- `itertools`: Provides functions to create iterators for efficient looping.
- `os`, `sys`, `time`: Standard libraries for file handling, system operations, and timing.
- `jax`: A library for high-performance machine learning research.
- `jax.numpy`: JAX's version of NumPy, with support for automatic differentiation and GPU/TPU acceleration.
- `numpy`: Fundamental package for scientific computing with Python.
- `numpyro`: A library for probabilistic programming in JAX.
- `src.model`: Custom module that provides functions for setting up and running cardiovascular simulations.

Functions:
- `sim_loop_wrapper`: Wraps the simulation loop to allow parameter modification.
- `logp`: Computes the log-probability of observed data given the model parameters.
- `model`: Defines the probabilistic model for inference using NumPyro.

Execution:
- The script reads command-line arguments to determine which configuration file to use.
- It sets up a simulation environment, runs the simulation, and then performs MCMC inference.
- Results are saved to a specified directory for further analysis.
"""

import os
import sys
import jax
import numpyro


# Change directory to the script's location
os.chdir(os.path.dirname(__file__) + "/..")
sys.path.insert(0, sys.path[0] + "/..")

from src.inference import param_inf_optax

# from src.inference import param_inf_numpyro

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# Set the number of devices to 1 for running the simulation
numpyro.set_host_device_count(1)


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
    # param_inf_numpyro(VESSEL_INDICES, VAR_INDICES, CONFIG_FILENAME)
    print("Inference complete.")


if __name__ == "__main__":
    main()
