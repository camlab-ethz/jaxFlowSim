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
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax import jit
from numpyro.infer import MCMC  # type: ignore
from numpyro.infer.reparam import TransformReparam  # type: ignore
import matplotlib.pyplot as plt
import scienceplots

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe

plt.style.use("science")

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

UPPER = 60000
Ccfl = 0.5


# Set up and execute the simulation loop using JIT compilation
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)
sim_dat_obs, t_obs, P_obs = SIM_LOOP_JIT(  # pylint: disable=E1102
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
    upper=UPPER,
)

# Indices for selecting specific parts of the simulation data
VESSEL_INDEX_1 = 1
VAR_INDEX_1 = 4

# Extract a specific value from the simulation data constants
R1_1 = sim_dat_const_aux[VESSEL_INDEX_1, VAR_INDEX_1]


def sim_loop_wrapper(params):
    """
    Wrapper function for running the simulation loop with a modified R value.

    Args:
        r (float): Scaling factor for the selected simulation constant.

    Returns:
        Array: Pressure values from the simulation with the modified R value.
    """
    r = params * R1_1
    sim_dat_const_aux_new = jnp.array(sim_dat_const_aux)
    sim_dat_const_aux_new = sim_dat_const_aux_new.at[VESSEL_INDEX_1, VAR_INDEX_1].set(r)
    _, t, p = SIM_LOOP_JIT(  # pylint: disable=E1102
        N,
        B,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux_new,
        Ccfl,
        input_data,
        rho,
        masks,
        strides,
        edges,
        upper=UPPER,
    )
    return p, t


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
    y_hat, _ = sim_loop_wrapper(r)  # pylint: disable=E1102
    log_prob = jnp.mean(
        jax.scipy.stats.norm.pdf(
            ((y[-10000:] - y_hat[-10000:])).flatten(), loc=0, scale=sigma
        )
    )
    jax.debug.print("L = {x}", x=log_prob)
    return log_prob


def model(p_obs, sigma):
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
            dist.LogNormal(),
        )
    jax.debug.print("R_dist = {x}", x=r_dist)
    log_density = logp(p_obs, r_dist, sigma=sigma)
    numpyro.factor("custom_logp", log_density)


# Define the hyperparameters for the network properties
network_properties = {
    "sigma": [1e-5],
    "scale": [10],
    "num_warmup": [4, 10, 20, 50, 100],
    "num_samples": [4, 100, 200, 500, 1000],
    "num_chains": [1],
}

# Create a list of all possible combinations of the network properties
settings = list(itertools.product(*network_properties.values()))  # type: ignore

# Define the folder to save the inference results
RESULTS_FOLDER = "results/inference_numpyro_1"
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
    # Record the start time if verbose mode is enabled
    starting_time = time.time_ns()
    mcmc.run(
        jax.random.PRNGKey(3450),
        P_obs,
        setup_properties["sigma"],
    )
    total_time = (time.time_ns() - starting_time) / 1.0e9
    mcmc.print_summary()
    R = jnp.mean(mcmc.get_samples()["theta"])

    print(R)
    y, t = sim_loop_wrapper(R)  # pylint: disable=E1102

    class Loss(object):

        def __init__(self, axis=0, order=None):

            super(Loss, self).__init__()

            self.axis = axis

            self.order = order

        def select_range(self, arr, start_idx, end_idx):

            length = end_idx - start_idx

            return jax.lax.dynamic_slice_in_dim(arr, start_idx, length)

        def relative_loss(self, s, s_pred):

            return jnp.power(
                jnp.linalg.norm(s_pred - s, ord=None, axis=self.axis), 2
            ) / jnp.power(jnp.linalg.norm(s, ord=None, axis=self.axis), 2)

        def __call__(self, s, s_pred):

            return jnp.mean(self.relative_loss(s, s_pred))

    loss = Loss()
    loss_val = loss(P_obs, y)
    print(loss_val)
    plt.scatter(t_obs[-21000:], P_obs[-21000:, -3] / 133.322, label="baseline", s=0.1)
    plt.scatter(t[-21000:], y[-21000:, -3] / 133.322, label="predicted", s=0.1)
    lgnd = plt.legend(loc="upper left")
    lgnd.legend_handles[0]._sizes = [30]
    lgnd.legend_handles[1]._sizes = [30]
    plt.xlabel("t/T")
    plt.ylabel("P [mmHg]")
    plt.title(
        f"learning scaled Windkessel resistance parameters of a bifurcation:\nR1_1 = {R},\nloss = {loss_val}, \nwallclock time = {total_time}"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([30, 140])
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_FOLDER}/numpyro_1_{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}.pdf"
    )
    plt.close()

    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        file.write(" " + str(R) + "  " + str(R1_1) + "\n")
