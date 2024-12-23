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
from jax import jit
from numpyro.infer import MCMC  # type: ignore
from numpyro.infer.reparam import TransformReparam  # type: ignore
import matplotlib.pyplot as plt

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe
import jaxtyping

jaxtyping.config.update("jaxtyping_disable", os.environ.get("JAXTYPING_DISABLE", "0"))

import arviz


import pymc

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

UPPER = 1000


# Set up and execute the simulation loop using JIT compilation
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)

# Indices for selecting specific parts of the simulation data
VESSEL_INDEX_1 = 1
VAR_INDEX_1 = 4
VESSEL_INDEX_2 = 2
VAR_INDEX_2 = 5
R1_1 = sim_dat_const_aux[VESSEL_INDEX_1, VAR_INDEX_1]
R2_1 = sim_dat_const_aux[VESSEL_INDEX_1, VAR_INDEX_2]
R1_2 = sim_dat_const_aux[VESSEL_INDEX_2, VAR_INDEX_1]
R2_2 = sim_dat_const_aux[VESSEL_INDEX_2, VAR_INDEX_2]


def sim_loop_wrapper(params, upper=UPPER):
    """
    Wrapper function for running the simulation loop with a modified R value.

    Args:
        r (float): Scaling factor for the selected simulation constant.

    Returns:
        Array: Pressure values from the simulation with the modified R value.
    """
    r1_1 = params[0] * R1_1 * 2
    r2_1 = params[1] * R2_1 * 2
    r1_2 = params[2] * R1_2 * 2
    r2_2 = params[3] * R2_2 * 2
    sim_dat_const_aux_new = jnp.array(sim_dat_const_aux)
    sim_dat_const_aux_new = sim_dat_const_aux_new.at[VESSEL_INDEX_1, VAR_INDEX_1].set(
        r1_1
    )
    sim_dat_const_aux_new = sim_dat_const_aux_new.at[VESSEL_INDEX_1, VAR_INDEX_2].set(
        r2_1
    )
    sim_dat_const_aux_new = sim_dat_const_aux_new.at[VESSEL_INDEX_2, VAR_INDEX_1].set(
        r1_2
    )
    sim_dat_const_aux_new = sim_dat_const_aux_new.at[VESSEL_INDEX_2, VAR_INDEX_2].set(
        r2_2
    )

    sim_dat_wrapped, t_wrapped, p_wrapped = SIM_LOOP_JIT(  # pylint: disable=E1102
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
        upper=upper,
    )
    return sim_dat_wrapped, t_wrapped, p_wrapped


sim_dat_obs, t_obs, P_obs = sim_loop_wrapper([1.0, 1.0, 1.0, 1.0])
sim_dat_obs_long, t_obs_long, P_obs_long = sim_loop_wrapper(
    [1.0, 1.0, 1.0, 1.0], upper=120000
)


class Loss(object):
    def __init__(self, axis=0, order=None):
        super(Loss, self).__init__()
        self.axis = axis
        self.order = order

    def relative_loss(self, s, s_pred):
        return jnp.log(
            jnp.mean(
                jnp.power(
                    jnp.linalg.norm(s_pred - s, ord=None, axis=self.axis),
                    2,
                )
                / jnp.power(jnp.linalg.norm(s, ord=None, axis=self.axis), 2)
            )
        )

    def __call__(self, s, s_pred):
        return self.relative_loss(s, s_pred)


loss = Loss()


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
    _, _, y_hat = sim_loop_wrapper(jax.nn.softplus(r))  # pylint: disable=E1102
    log_prob = jnp.log(
        jnp.mean(
            jnp.linalg.norm(
                jax.scipy.stats.norm.pdf(
                    ((y - y_hat)) / y.mean(axis=0),
                    loc=0,
                    scale=sigma,
                ),
                axis=0,
            )
        )
    )
    # jax.debug.print("L = {x}", x=y - y_hat / y.mean())
    # log_prob = -loss(y, y_hat)
    # jax.debug.print("L: {x}", x=log_prob)
    return log_prob


def model(p_obs):
    """
    Define the probabilistic model for inference using NumPyro.

    Args:
        p_obs (Array): Observed pressure data.
        sigma (float): Standard deviation of the noise in the observations.
        scale (float): Scale parameter for the prior distribution.
        r_scale (float): Initial scale value for R.
    """
    # r_dist = numpyro.sample(
    #    "theta",
    #    dist.Normal(),
    #    sample_shape=(4,),
    # )
    r_dist1 = numpyro.sample(
        "theta1",
        dist.TransformedDistribution(
            dist.Normal(0, 1), numpyro.distributions.transforms.SoftplusTransform()
        ),
    )
    r_dist2 = numpyro.sample(
        "theta2",
        dist.TransformedDistribution(
            dist.Normal(0, 1), numpyro.distributions.transforms.SoftplusTransform()
        ),
    )
    r_dist3 = numpyro.sample(
        "theta3",
        dist.TransformedDistribution(
            dist.Normal(0, 1), numpyro.distributions.transforms.SoftplusTransform()
        ),
    )
    r_dist4 = numpyro.sample(
        "theta4",
        dist.TransformedDistribution(
            dist.Normal(0, 1), numpyro.distributions.transforms.SoftplusTransform()
        ),
    )
    # sigma = numpyro.sample("sigma", dist.LogNormal(10, 1))

    # sigma = numpyro.sample(
    #    "sigma",
    #    dist.LogNormal(),
    # )
    jax.debug.print("test: {x}", x=jnp.array([r_dist1, r_dist2, r_dist3, r_dist4]))
    # log_density = logp(p_obs, r_dist, sigma=sigma)
    # numpyro.factor("custom_logp", log_density0)
    numpyro.sample(
        "obs",
        dist.Normal(
            (
                sim_loop_wrapper(jnp.array([r_dist1, r_dist2, r_dist3, r_dist4]))[2]
                - p_obs
            )
            / jnp.mean(p_obs),
            1,
        ),
        obs=jnp.zeros(P_obs.shape),
    )


# Define the hyperparameters for the network properties
network_properties = {
    "sigma": [1e-2],
    "scale": [10],
    "num_warmup": [1000],
    "num_samples": [1000],
    "num_chains": [1],
}


# Create a list of all possible combinations of the network properties
settings = list(itertools.product(*network_properties.values()))  # type: ignore


def geweke_diagnostic(samples, first_frac=0.1, last_frac=0.5):
    """
    Compute the Geweke diagnostic z-scores for a single chain.

    Parameters:
    - samples: np.ndarray
        1D array of MCMC samples for a parameter.
    - first_frac: float
        Fraction of the chain to use for the early segment.
    - last_frac: float
        Fraction of the chain to use for the late segment.

    Returns:
    - z_scores: np.ndarray
        Geweke z-scores for different points in the chain.
    - indices: np.ndarray
        Indices of the chain where z-scores are computed.
    """
    n_samples = len(samples)
    first_samples = samples[: int(first_frac * n_samples)]
    last_samples = samples[-int(last_frac * n_samples) :]

    mean_first = np.mean(first_samples, axis=0)
    mean_last = np.mean(last_samples, axis=0)

    var_first = np.var(first_samples, ddof=1, axis=0)
    var_last = np.var(last_samples, ddof=1, axis=0)

    z_score = (mean_first - mean_last) / np.sqrt(
        var_first / len(first_samples) + var_last / len(last_samples)
    )
    return z_score


def generate_geweke_plot(
    mcmc_samples, var_name, setup_properties, first_frac=0.1, last_frac=0.5, step=50
):
    """
    Generate a Geweke plot for diagnostics of convergence.

    Parameters:
    - mcmc_samples: dict
        Dictionary of MCMC samples (obtained via `mcmc.get_samples()`).
    - var_name: str
        The variable name to analyze.
    - first_frac: float
        Fraction of the chain at the beginning to use for comparison.
    - last_frac: float
        Fraction of the chain at the end to use for comparison.
    - step: int
        Step size for computing z-scores along the chain.
    """
    # Extract the samples for the specified variable
    samples = mcmc_samples[var_name]
    samples = samples[int(0.5 * len(samples)) :]
    # if samples.ndim > 1:  # Multi-chain samples
    #    samples = samples.reshape(-1)

    # Compute Geweke z-scores along the chain
    z_scores = []
    indices = []
    for i in range(step, len(samples), step):
        partial_samples = samples[:i]
        z_score = geweke_diagnostic(partial_samples, first_frac, last_frac)
        z_scores.append(z_score)
        indices.append(i)

    # Create a Geweke plot
    plt.figure()
    plt.plot(
        indices,
        z_scores,
        marker="o",
        linestyle="--",
        label=["R1_1", "R2_1", "R1_2", "R2_2"],
    )
    plt.axhline(-2, color="red", linestyle="--")
    plt.axhline(2, color="red", linestyle="--")
    plt.axhline(0, color="black", linestyle="-")
    plt.xlabel("iteration")
    plt.ylabel("z-score")
    plt.legend()
    # plt.grid(alpha=0.3)
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke_nt.pdf"
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke_nt.png"
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke_nt.eps"
    )
    plt.title(f"geweke diagnostic for {var_name}")
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke.pdf"
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke.png"
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke.eps"
    )
    plt.close()


# Define the folder to save the inference results
RESULTS_FOLDER = "results/inference_numpyro_4"
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
        numpyro.infer.NUTS(model, forward_mode_differentiation=True, dense_mass=True),
        num_samples=setup_properties["num_samples"],
        num_warmup=setup_properties["num_warmup"],
        num_chains=setup_properties["num_chains"],
    )
    starting_time = time.time_ns()
    mcmc.run(
        jax.random.PRNGKey(3450),
        P_obs,
    )
    total_time = (time.time_ns() - starting_time) / 1.0e9
    mcmc.print_summary()
    R = jnp.mean(
        mcmc.get_samples()["theta"][int(0.5 * len(mcmc.get_samples())) :], axis=0
    ).flatten()

    print(R)
    print(mcmc.get_samples()["theta"].shape)
    _, t, y = sim_loop_wrapper(jax.nn.softplus(R))  # pylint: disable=E1102
    loss_val = loss(P_obs, y)
    _, t, y = sim_loop_wrapper(
        jax.nn.softplus(R), upper=120000
    )  # pylint: disable=E1102
    indices_sorted = np.argsort(t_obs_long[-12000:])
    plt.scatter(
        t_obs_long[-12000:][indices_sorted],
        P_obs_long[-12000:, -8][indices_sorted] / 133.322,
        label="ground truth",
        s=0.1,
    )
    indices_sorted = np.argsort(t[-12000:])
    plt.scatter(
        t[-12000:][indices_sorted],
        y[-12000:, -8][indices_sorted] / 133.322,
        label="predicted",
        s=0.1,
    )
    lgnd = plt.legend(loc="upper right")
    lgnd.legend_handles[0]._sizes = [30]
    lgnd.legend_handles[1]._sizes = [30]
    plt.xlabel("t/T")
    plt.ylabel("P [mmHg]")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([30, 140])
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_nt.pdf"
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_nt.png"
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_nt.eps"
    )
    plt.title(
        f"learning scaled Windkessel resistance parameters of a bifurcation:\n[R1_1, R2_1, R1_2, R2_2] = {R},\nloss = {loss_val}, \nwallclock time = {total_time}"
    )
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}.pdf"
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}.png"
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}.eps"
    )
    plt.close()

    with open(RESULTS_FILE, "a", encoding="utf-8") as file:
        file.write(str(R) + "  " + str(R1_1) + "\n")

    # arviz_data = arviz.from_numpyro(mcmc)
    # arviz.geweke(mcmc.get_samples()["theta"])

    # arviz_data

    # geweke = pymc.diagnostics.geweke(mcmc.get_samples()["theta"])
    # arviz.geweke_plot(geweke)
    generate_geweke_plot(mcmc.get_samples(), "theta", setup_properties, step=5)
