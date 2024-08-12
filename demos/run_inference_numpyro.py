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

import itertools
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax import block_until_ready, jit
from numpyro.infer import MCMC  # type: ignore
from numpyro.infer.reparam import TransformReparam  # type: ignore

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe

# Change directory to the script's location
os.chdir(os.path.dirname(__file__) + "/..")


# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# Set the number of devices to 1 for running the simulation
numpyro.set_host_device_count(1)

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
    Ccfl,
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
sim_dat_obs, t_obs, P_obs = block_until_ready(
    SIM_LOOP_JIT(  # pylint: disable=E1102
        N,
        B,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        Ccfl,
        input_data,
        rho,
        masks,
        strides,
        edges,
        upper=120000,
    )
)

# Indices for selecting specific parts of the simulation data
R_INDEX = 1
VAR_INDEX = 7

# Extract a specific value from the simulation data constants
R1 = sim_dat_const[VAR_INDEX, strides[R_INDEX, 1]]

# Generate a range of scaled values for R to be tested
R_scales = np.linspace(0.5 * R1, 0.9 * R1, 16)
if len(sys.argv) > 1:
    R_scale = R_scales[int(sys.argv[1])]
else:
    R_scale = R_scales[0]


def sim_loop_wrapper(r):
    """
    Wrapper function for running the simulation loop with a modified R value.

    Args:
        r (float): Scaling factor for the selected simulation constant.

    Returns:
        Array: Pressure values from the simulation with the modified R value.
    """
    ones = jnp.ones(strides[R_INDEX, 1] - strides[R_INDEX, 0] + 4)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[
        VAR_INDEX, strides[R_INDEX, 0] - 2 : strides[R_INDEX, 1] + 2
    ].set(r * ones)
    _, _, p = block_until_ready(
        SIM_LOOP_JIT(  # pylint: disable=E1102
            N,
            B,
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            Ccfl,
            input_data,
            rho,
            masks,
            strides,
            edges,
            upper=120000,
        )
    )
    return p


# JIT compile the wrapper function for efficiency
SIM_LOOP_WRAPPER_JIT = jit(sim_loop_wrapper)


def logp(y, r, sigma):
    """
    Compute the log-probability of the observed data given the model parameters.

    Args:
        y (Array): Observed pressure data.
        r (float): Scaling factor for the selected simulation constant.
        sigma (float): Standard deviation of the noise in the observations.

    Returns:
        float: Log-probability of the observed data given the model parameters.
    """
    y_hat = SIM_LOOP_WRAPPER_JIT(r)  # pylint: disable=E1102
    log_prob = jnp.mean(
        jax.scipy.stats.norm.pdf(((y - y_hat)).flatten(), loc=0, scale=sigma)
    )
    jax.debug.print("L = {x}", x=log_prob)
    return log_prob


def model(p_obs, sigma, scale, r_scale):
    """
    Define the probabilistic model for inference using NumPyro.

    Args:
        p_obs (Array): Observed pressure data.
        sigma (float): Standard deviation of the noise in the observations.
        scale (float): Scale parameter for the prior distribution.
        r_scale (float): Initial scale value for R.
    """
    with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
        r_dist = numpyro.sample(
            "theta",
            dist.TransformedDistribution(
                dist.Normal(), dist.transforms.AffineTransform(0, scale * r_scale)
            ),
        )
    jax.debug.print("R_dist = {x}", x=r_dist)
    log_density = logp(p_obs, r_dist, sigma=sigma)
    numpyro.factor("custom_logp", log_density)


# Define the hyperparameters for the network properties
network_properties = {
    "sigma": [1e-5],
    "scale": [10],
    "num_warmup": np.arange(10, 110, 10),
    "num_samples": np.arange(100, 1100, 100),
    "num_chains": [1],
}

# Create a list of all possible combinations of the network properties
settings = list(itertools.product(*network_properties.values()))  # type: ignore

# Define the folder to save the inference results
RESULTS_FOLDER = "results/inference_ensemble"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)

# Loop through each combination of settings and run MCMC inference
for set_num, setup in enumerate(settings):
    print(
        "###################################",
        set_num,
        "###################################",
    )
    setup_properties = {
        "sigma": setup[0],
        "scale": setup[1],
        "num_warmup": setup[2],
        "num_samples": setup[3],
        "num_chains": setup[4],
    }
    RESULTS_FILE = (
        RESULTS_FOLDER
        + "/setup_"
        + str(setup_properties["sigma"])
        + "_"
        + str(setup_properties["scale"])
        + "_"
        + str(setup_properties["num_warmup"])
        + "_"
        + str(setup_properties["num_samples"])
        + "_"
        + str(setup_properties["num_chains"])
        + ".txt"
    )
    mcmc = MCMC(
        numpyro.infer.NUTS(model, forward_mode_differentiation=True),
        num_samples=setup_properties["num_samples"],
        num_warmup=setup_properties["num_warmup"],
        num_chains=setup_properties["num_chains"],
    )
    mcmc.run(
        jax.random.PRNGKey(3450),
        P_obs,
        setup_properties["sigma"],
        setup_properties["scale"],
        R_scale,
    )
    mcmc.print_summary()
    R = jnp.mean(mcmc.get_samples()["theta"])

    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        file.write(str(R_scale) + " " + str(R) + "  " + str(R1) + "\n")
