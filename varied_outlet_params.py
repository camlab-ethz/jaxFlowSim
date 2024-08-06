"""
This module sets up and runs simulations for vascular networks with different configurations,
compares the results, and generates plots for the pressure data.

Functions included:
- `config_simulation`: Configures the simulation parameters from a YAML configuration file.
- `simulation_loop`: Executes the simulation loop for the vascular network.
- `block_until_ready`: Ensures all computations are complete before proceeding.
- `partial`: Partially applies arguments to functions.
- `os` and `shutil` are used for file and directory operations.
- `matplotlib.pyplot` is used for plotting results.

The module runs simulations for three different configurations and saves the comparison plots to a specified directory.
"""

import os
import shutil
from functools import partial

import jax
import matplotlib.pyplot as plt
from jax import jit

from src.model import config_simulation, simulation_loop

# Change the current working directory to the directory of this script
os.chdir(os.path.dirname(__file__))

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

# Load and run the first configuration
CONFIG_FILENAME = "test/bifurcation/bifurcation.yml"
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
) = config_simulation(CONFIG_FILENAME)
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
sim_dat1, t1, P1 = SIM_LOOP_JIT(  # pylint: disable=E1102
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
)

# Load and run the second configuration
CONFIG_FILENAME = "test/bifurcation2/bifurcation2.yml"
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
) = config_simulation(CONFIG_FILENAME)
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
sim_dat2, t2, P2 = SIM_LOOP_JIT(  # pylint: disable=E1102
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
)

# Load and run the third configuration
CONFIG_FILENAME = "test/bifurcation3/bifurcation3.yml"
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
) = config_simulation(CONFIG_FILENAME)
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
sim_dat3, t3, P3 = SIM_LOOP_JIT(  # pylint: disable=E1102
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
)

# Define the results folder and clear any existing results
R_FOLDER = "results/compare_output_params_results"
if os.path.isdir(R_FOLDER):
    shutil.rmtree(R_FOLDER)
os.makedirs(R_FOLDER, mode=0o777)

# Extract the network name from the configuration filename
filename = CONFIG_FILENAME.rsplit("/", maxsplit=1)[-1]
network_name = filename.split(".")[0]


# Generate and save plots for each vessel
for i, vessel_name in enumerate(vessel_names):
    index_vessel_name = vessel_names.index(vessel_name)
    NODE = 2
    index_jax = 5 * index_vessel_name + NODE
    _, ax = plt.subplots()
    ax.set_xlabel("t[s]")
    ax.set_ylabel("P[mmHg]")
    plt.plot(t1 % cardiac_T, P1[:, index_jax] / 133.322)
    plt.plot(t2 % cardiac_T, P2[:, index_jax] / 133.322)
    plt.plot(t3 % cardiac_T, P3[:, index_jax] / 133.322)
    plt.legend(["P_1", "P_2", "P_3"], loc="lower right")
    plt.tight_layout()
    plt.savefig(R_FOLDER + "/compare_output_params_" + vessel_name + "_P.eps")
    plt.close()
