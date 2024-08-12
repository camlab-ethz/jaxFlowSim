"""
This script performs a series of vascular network simulations using JAX and plots the scaling results
as a function of the number of vessels. The script is designed to evaluate the computational performance
of different vascular network models.

The simulation process includes the following steps:
1. Set up the environment and configuration.
2. Load and iterate through a list of predefined vascular models.
3. For each model, configure the simulation, run the simulation loop, and measure the execution time.
4. Store the timing results and the number of vessels for each model.
5. Plot the number of vessels against the execution time and save the plot to a file.

Modules:
    - os: Provides functions for interacting with the operating system.
    - shutil: Offers high-level file operations such as copying and removing files.
    - time: Contains functions for time-related operations.
    - functools.partial: Allows partial function application.
    - jax: The JAX library for high-performance numerical computing.
    - matplotlib.pyplot: A plotting library for visualizing data.
    - src.model: Custom module containing functions for simulation configuration and execution.

Functions:
    - config_simulation: Configures the simulation parameters based on the model's configuration file.
    - simulation_loop: Runs the main simulation loop.

Global Variables:
    - VERBOSE (bool): Controls whether to print detailed logs and timing information.
    - timeings (list): Stores the execution time for each model simulation.
    - num_vessels (list): Stores the number of vessels for each model.
    - model_names (list): A list of vascular network models to be simulated.
    - filenames (list): A list of configuration file paths for each model.
    - CONFIG_FILENAME (str): The current configuration file path being processed.

Usage:
    To execute the script, simply run it in an environment with the necessary dependencies installed.
    The script will generate and save a plot of the simulation scaling results as an EPS file in the
    'results/scaling_results' directory.
"""

import os
import sys
import shutil
import time
from functools import partial

import jax
import matplotlib.pyplot as plt
from jax import block_until_ready, jit

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop

# Change directory to the script's location
os.chdir(os.path.dirname(__file__) + "/..")

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# Set verbosity flag to control logging
VERBOSE = True

# Initialize lists to store timings and the number of vessels for each simulation
timings = []
num_vessels = []

# List of model names to be simulated
model_names = [
    "single-artery",
    "tapering",
    "conjunction",
    "bifurcation",
    "adan56",
    "0053_H_CERE_H",
    "0007_H_AO_H",
    "0029_H_ABAO_H",
]

# Initialize a list to hold the configuration filenames
filenames = []

# Generate configuration filenames based on model names
CONFIG_FILENAME = ""
for model_name in model_names:
    CONFIG_FILENAME = "test/" + model_name + "/" + model_name + ".yml"
    filenames.append(CONFIG_FILENAME)

# Loop through each configuration file and run the simulation
for CONFIG_FILENAME in filenames:
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
    ENDING_TIME = 0.0

    # If verbose mode is enabled, record the start time
    if VERBOSE:
        STARTING_TIME = time.time_ns()

    # Set up and execute the simulation loop using JIT compilation
    SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
    _, _, _ = block_until_ready(
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
        ENDING_TIME = (time.time_ns() - STARTING_TIME) / 1.0e9
        print(f"elapsed time = {ENDING_TIME} seconds")
        timings.append(ENDING_TIME)

    # Append the elapsed time and the number of vessels to their respective lists
    num_vessels.append(len(vessel_names))

# Define the folder to save the results
R_FOLDER = "results/scaling_results"

# Delete the existing results folder if it exists, and create a new one
if os.path.isdir(R_FOLDER):
    shutil.rmtree(R_FOLDER)
os.makedirs(R_FOLDER, mode=0o777)

# Extract the network name from the configuration filename
filename = CONFIG_FILENAME.split("/")[-1]
network_name = filename.split(".")[0]

# Plotting the results: number of vessels vs. timing
_, ax = plt.subplots()
ax.set_xlabel("# vessels")
ax.set_ylabel("t[s]")
plt.scatter(num_vessels, timings)
plt.tight_layout()

# Save the plot as an EPS file
plt.savefig(R_FOLDER + "/scaling.eps")
plt.close()
