"""
This script performs vascular network simulations using JAX, compares the simulation results with reference data,
and generates plots showing the relative error between the simulation and reference pressures for each vessel in the network.

The script includes the following steps:
1. Set up the environment and configuration.
2. Load the vascular network model based on the provided configuration file.
3. Run the simulation loop using JAX.
4. Compare the simulation results with reference data and calculate the relative error.
5. Plot the pressure data from the simulation and the reference data, and save the plots as PDF files.

Modules:
    - os: Provides functions for interacting with the operating system.
    - sys: Provides access to some variables used or maintained by the interpreter.
    - time: Contains functions for time-related operations.
    - functools.partial: Allows partial function application.
    - jax: The JAX library for high-performance numerical computing.
    - matplotlib.pyplot: A plotting library for visualizing data.
    - numpy: A library for numerical computations.
    - src.model: Custom module containing functions for simulation configuration and execution.

Functions:
    - config_simulation: Configures the simulation parameters based on the model's configuration file.
    - simulation_loop: Runs the main simulation loop.

Global Variables:
    - CONFIG_FILENAME (str): The configuration file path for the simulation.
    - VERBOSE (bool): Controls whether to print detailed logs and timing information.

Usage:
    To execute the script, run it with an optional model name argument:
        python script.py [model_name]
    If no model name is provided, the script defaults to using "test/adan56/adan56.yml".
    The script will generate and save plots of the simulation results in the 'results' directory.
"""

import os
import sys
import time
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from jax import block_until_ready, jit

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop

plt.style.use("science")
# Change directory to the script's location
os.chdir(os.path.dirname(__file__) + "/..")

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# Determine the configuration filename based on command line arguments
CONFIG_FILENAME = ""
if len(sys.argv) == 1:
    MODELNAME = "adan56"
    CONFIG_FILENAME = f"test/{MODELNAME}/{MODELNAME}.yml"
else:
    MODELNAME = sys.argv[1]
    CONFIG_FILENAME = f"test/{MODELNAME}/{MODELNAME}.yml"

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

# Initialize timing variables
STARTING_TIME = 0.0

# If verbose mode is enabled, record the start time
if VERBOSE:
    STARTING_TIME = time.time_ns()

# Set up and execute the simulation loop using JIT compilation
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)

sim_dat, t, P = block_until_ready(
    SIM_LOOP_JIT(  # pylint: disable=E1102
        N,
        B,
        J,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        timepoints,
        float(conv_tol),
        Ccfl,
        input_data,
        rho,
        masks,
        strides,
        edges,
    )
)

# If verbose mode is enabled, calculate and print the elapsed time
if VERBOSE:
    ending_time = (time.time_ns() - STARTING_TIME) / 1.0e9
    print(f"elapsed time = {ending_time} seconds")

# Extract the network name from the configuration filename
filename = CONFIG_FILENAME.split("/")[-1]
network_name = filename.split(".")[0]

# Define vessel names for different configurations (example data)
vessel_names_0007 = [
    "ascending aorta",
    "right subclavian artery",
    "right common carotid artery",
    "arch of aorta I",
    "brachiocephalic artery",
    "arch of aorta II",
    "left common carotid artery",
    "left subclavian artery",
    "descending aorta",
]
vessel_names_0029 = [
    "aorta I",
    "left common iliac artery I",
    "left internal iliac artery",
    "left common iliac artery II",
    "right common iliac artery I",
    "celiac trunk II",
    "celiac branch",
    "aorta IV",
    "left renal artery",
    "aorta III",
    "superior mesentric artery",
    "celiac trunk I",
    "aorta II",
    "aorta V",
    "right renal artery",
    "right common iliac artery II",
    "right internal iliac artery",
]
vessel_names_0053 = [
    "right vertebral artery I",
    "left vertebral artery II",
    "left posterior meningeal branch of vertebral artery",
    "basilar artery III",
    "left anterior inferior cerebellar artery",
    "basilar artery II",
    "right anterior inferior cerebellar artery",
    "basilar artery IV",
    "right superior cerebellar artery",
    "basilar artery I",
    "left vertebral artery I",
    "right posterior cerebellar artery I",
    "left superior cerebellar artery",
    "left posterior cerebellar artery I",
    "right posterior central artery",
    "right vertebral artery II",
    "right posterior meningeal branch of vertebral artery",
    "right posterior cerebellar artery II",
    "right posterior comunicating artery",
]

vessel_names_jl = vessel_names
if MODELNAME == "0007_H_AO_H":
    vessel_names = vessel_names_0007
elif MODELNAME == "0029_H_ABAO_H":
    vessel_names = vessel_names_0029
elif MODELNAME == "0053_H_CERE_H":
    vessel_names = vessel_names_0053

# Loop through each vessel name and compare the simulation results with reference data
for index_vessel_name, vessel_name in enumerate(vessel_names):
    P0 = np.loadtxt(
        f"/home/diego/studies/uni/thesis_maths/openBF_bckp/openBF/test/{network_name}/{network_name}_results/{vessel_names_jl[index_vessel_name]}_P.last"
    )
    NODE = 2
    INDEX_JL = 1 + NODE
    index_jax = 5 * index_vessel_name + NODE
    P0 = P0[:, INDEX_JL]
    res = np.sqrt(((P[:, index_jax] - P0).dot(P[:, index_jax] - P0) / P0.dot(P0)))
    _, ax = plt.subplots()
    ax.set_xlabel("t[s]")
    ax.set_ylabel("P[mmHg]")
    # plt.title(
    #    "network: "
    #    + network_name
    #    + r", \#vessels: "
    #    + str(N)
    #    + ", vessel name: "
    #    + vessel_name
    #    + ", \n"
    #    + r" relative error $= |P_{JAX}-P_{jl}|/|P_{jl}| \approx "
    #    + str(np.round(res, 6))
    #    + "$",
    # )
    plt.plot(t % cardiac_T, P[:, index_jax] / 133.322)
    plt.plot(t % cardiac_T, P0 / 133.322)
    plt.legend(["$P_{JAX}$", "$P_{jl}$"], loc="upper right")
    plt.tight_layout()
    plt.savefig(
        f"results/{network_name}_results/{network_name}_{vessel_names[index_vessel_name].replace(" ", "_")}_P_nt.eps"
    )
    plt.close()
