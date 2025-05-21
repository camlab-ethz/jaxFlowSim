"""
This script performs a simulation of cardiovascular dynamics using JAX, and then compares the simulated
pressure data with reference data from previously saved results. The comparison includes computing
relative errors and generating plots to visualize the differences.

Key functionalities include:
- Configuring and running multiple simulations with JAX's Just-In-Time (JIT) compilation for performance.
- Identifying cycles in the simulation output to extract relevant data for comparison.
- Comparing the simulated pressure data against reference data for specific vessels.
- Generating and saving plots of the simulation results against reference data.

Modules:
- `os`, `sys`, `time`: Standard libraries for file handling, system operations, and timing.
- `jax`: A library for high-performance machine learning research.
- `jax.numpy`: JAX's version of NumPy, with support for automatic differentiation and GPU/TPU acceleration.
- `matplotlib.pyplot`: A plotting library for creating static, animated, and interactive visualizations.
- `numpy`: A fundamental package for scientific computing with Python.

Execution:
- The script reads command-line arguments to determine which configuration file to use.
- It sets up a simulation environment, runs the simulation, and then compares the results with reference data.
- Results are saved as plots in a specified directory for further analysis.
"""

import os
import sys
import time
from functools import partial

import jax
import matplotlib.pyplot as plt
from jax import block_until_ready, jit

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe

# Change directory to the script's location
os.chdir(os.path.dirname(__file__) + "/..")

# Enable 64-bit precision in JAX for higher accuracy in numerical computations
jax.config.update("jax_enable_x64", True)

# Set the configuration filename based on command line arguments or default to a specific model
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
) = config_simulation(CONFIG_FILENAME)

# Initialize lists to store time and pressure data for each simulation
t_t = []
p_t = []

# Set up and execute the simulation loop using JIT compilation
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)
# Initialize timing variables
STARTING_TIME = 0.0

# If verbose mode is enabled, record the start time
if VERBOSE:
    print(f"Running model: {MODELNAME}")
    STARTING_TIME = time.time_ns()

# Run the simulation loop with JIT compilation
sim_dat, t_t, p_t = block_until_ready(
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

# If verbose mode is enabled, calculate and print the elapsed time
if VERBOSE:
    ending_time = (time.time_ns() - STARTING_TIME) / 1.0e9
    print(f"Finished running models, elapsed time = {ending_time} seconds")
    print("Plotting results into resluts directory")


# Plotting setup starts here, uncomment lines as necessary to plot third-party data
# Extract the network name from the configuration filename
filename = CONFIG_FILENAME.split("/")[-1]
network_name = filename.split(".")[0]

# Reference vessel names for different networks
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

# Identify cycles in the time data to extract one complete cycle of pressure data
indices = [i + 1 for i in range(len(t_t) - 1) if t_t[i] > t_t[i + 1]]
P_cycle = p_t[indices[-2] : indices[-1], :]  # type: ignore
t_cycle = t_t[indices[-2] : indices[-1]]

# Load reference pressure data for the first vessel
# P0_temp = np.loadtxt(
#     f"/home/diego/studies/uni/thesis_maths/openBF/test/{network_name}/{network_name}_results/{vessel_names[0]}_P.last"
# )
# t0 = P0_temp[:, 0] % cardiac_T

# Initialize arrays to store the new interpolated time and pressure data
# COUNTER = 0
# t_new = np.zeros(len(timepoints))
# P_new = np.zeros((len(timepoints), 5 * N))

# Interpolate the pressure data to match the reference time points
# for i in range(len(t_cycle) - 1):
#     if t0[COUNTER] >= t_cycle[i] and t0[COUNTER] <= t_cycle[i + 1]:
#         P_new[COUNTER, :] = (P_cycle[i, :] + P_cycle[i + 1, :]) / 2
#         COUNTER += 1

# t_new = t_new[:-1]
# P_new = P_new[:-1, :]

vessel_names_jl = vessel_names
if MODELNAME == "000/_H_A0_H":
    vessel_names = vessel_names_0007
elif MODELNAME == "0029_H_ABAO_H":
    vessel_names = vessel_names_0029
elif MODELNAME == "0053_H_CERE_H":
    vessel_names = vessel_names_0053

# Loop through each vessel and plot the pressure data
for i, vessel_name in enumerate(vessel_names):
    index_vessel_name = vessel_names.index(vessel_name)
    # P0_temp = np.loadtxt(
    #     f"/home/diego/studies/uni/thesis_maths/openBF/test/{network_name}/{network_name}_results/{vessel_names_jl[i]}_P.last"
    # )
    NODE = 2
    INDEX_JL = 1 + NODE
    index_jax = 5 * index_vessel_name + NODE

    # P0 = P0_temp[:-1, INDEX_JL]
    # t0 = P0_temp[:-1, 0] % cardiac_T
    # P1 = P_new[:, index_jax]

    # Compute the relative error between the simulated and reference pressures
    # res = np.sqrt(((P1 - P0).dot(P1 - P0) / P0.dot(P0)))

    # Generate and save a plot comparing the simulated and reference pressures
    _, ax = plt.subplots()
    ax.set_xlabel("t[s]")
    ax.set_ylabel("P[mmHg]")
    # plt.title(
    #     f"network: {network_name} # vessels: {N}, vessel name: {vessel_names[i]}, \n relative error = |P_JAX-P_jl|/|P_jl| = {res}"
    # )
    # plt.plot(t0, P0 / 133.322)
    plt.plot(t_cycle, P_cycle / 133.322)
    plt.legend(["P_JAX", "P_jl"], loc="lower right")
    plt.tight_layout()
    plt.savefig(
        f"results/{network_name}_results/{network_name}_{vessel_names[i].replace(' ', '_')}_P.pdf"
    )
    plt.close()
