"""
This script performs a simulation of blood flow in a vascular network using JAX for Just-In-Time (JIT) compilation,
and optimizes the simulation step size for computational efficiency and accuracy. The script loads a specified model 
configuration, runs the simulation, calculates the residual between the simulation and reference data, and iteratively 
adjusts the step size to optimize the simulation. Results, including the time taken and residuals for different step sizes, 
are plotted and saved as a PDF.

Steps:
1. Set up the environment and load the simulation configuration based on the specified model.
2. Run the simulation with an initial step size and record the timing and residual.
3. Adjust the simulation step size in a loop, re-running the simulation for each step size and recording the time and residual.
4. Plot the results showing the time taken and residuals against the step size.
5. Save the results as a PDF file.

Attributes:
    CONFIG_FILENAME (str): The path to the configuration file used for the simulation.
    VERBOSE (bool): Flag to enable or disable verbose output for timing information.
    N, B, J (int): Network parameters for the simulation.
    sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux (array): Arrays holding the simulation data.
    timepoints (array): Timepoints used in the simulation.
    conv_tol (float): Convergence tolerance for the simulation.
    Ccfl (float): CFL condition value used in the simulation.
    input_data, rho, masks, strides, edges (array): Additional arrays holding data for the simulation.
    vessel_names (list): List of vessel names in the network.
    cardiac_T (float): The period of the cardiac cycle used in the simulation.
    r_folder (str): The directory where the results will be stored.
    steps (range): Range of step sizes to test during the optimization process.
    times (list): List of times taken for each simulation with different step sizes.
    residuals (list): List of residuals calculated for each simulation with different step sizes.
    RESIDUAL_BASE (float): The residual calculated for the base simulation run.
    ENDING_TIME_BASE (float): The time taken for the base simulation run.
"""

import os
import shutil
import sys
import time
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import block_until_ready, jit

from src.model import config_simulation, simulation_loop, simulation_loop_unsafe

# Change the current working directory to the script's location
os.chdir(os.path.dirname(__file__))

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)

# Set the configuration filename based on command line arguments or default to a specific model
CONFIG_FILENAME = ""
if len(sys.argv) == 1:

    MODELNAME = "test/adan56/adan56.yml"

    CONFIG_FILENAME = "test/" + MODELNAME + "/" + MODELNAME + ".yml"
else:
    CONFIG_FILENAME = "test/" + sys.argv[1] + "/" + sys.argv[1] + ".yml"

# Enable verbose output for timing information
VERBOSE = True

# Load the simulation configuration
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

# Extract the network name from the configuration filename
filename = CONFIG_FILENAME.split("/")[-1]
network_name = filename.split(".")[0]

# Create a directory to store the results, removing any existing directory with the same name
r_folder = "results/steps_opt_" + network_name
if os.path.isdir(r_folder):
    shutil.rmtree(r_folder)
os.makedirs(r_folder, mode=0o777)

# Vessel names for different networks
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

# Record the starting time of the simulation
STARTING_TIME = 0.0
if VERBOSE:
    STARTING_TIME = time.time_ns()

# Execute the simulation loop with Just-In-Time (JIT) compilation
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
sim_dat_out, t_out, P_out = block_until_ready(
    SIM_LOOP_JIT(  # pylint: disable=E1102
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
)

# Calculate the elapsed time for the base simulation
ENDING_TIME_BASE = 0.0
if VERBOSE:
    ENDING_TIME_BASE = (time.time_ns() - STARTING_TIME) / 1.0e9
    print(f"elapsed time = {ENDING_TIME_BASE} seconds")

# Load reference pressure data for comparison
P0_temp = np.loadtxt(
    "/home/diego/studies/uni/thesis_maths/openBF/test/"
    + network_name
    + "/"
    + network_name
    + "_results/"
    + vessel_names[0]
    + "_P.last"
)
t0 = P0_temp[:, 0] % cardiac_T

# Interpolate the results to match the reference timepoints
COUNTER = 0
t_new = np.zeros(len(timepoints))
P_new = np.zeros((len(timepoints), 5 * N))
for i in range(len(t_out) - 1):
    if t0[COUNTER] >= t_out[i] and t0[COUNTER] <= t_out[i + 1]:
        P_new[COUNTER, :] = (P_out[i, :] + P_out[i + 1, :]) / 2
        COUNTER += 1

t_new = t_new[:-1]
P_new = P_new[:-1, :]

# Calculate the residual between the simulation and the reference data
RESIDUAL_BASE = 0
for i, vessel_name in enumerate(vessel_names):
    index_vessel_name = vessel_names.index(vessel_name)
    P0_temp = np.loadtxt(
        "/home/diego/studies/uni/thesis_maths/openBF/test/"
        + network_name
        + "/"
        + network_name
        + "_results/"
        + vessel_name
        + "_P.last"
    )
    NODE = 2
    INDEX_JL = 1 + NODE
    index_jax = 5 * index_vessel_name + NODE

    P0 = P0_temp[:-1, INDEX_JL]
    t0 = P0_temp[:-1, 0] % cardiac_T
    P1 = P_new[:, index_jax]
    RESIDUAL_BASE += np.sqrt(((P1 - P0).dot(P1 - P0) / P0.dot(P0)))

RESIDUAL_BASE = RESIDUAL_BASE / N

# Loop over different step sizes to optimize simulation time and residuals
times = []
steps = range(50000, 210000, 10000)
residuals = []
for m in steps:
    if VERBOSE:
        STARTING_TIME = time.time_ns()

    # Execute the simulation loop with a different step size
    SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)
    sim_dat_out, t_t, P_t = block_until_ready(
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

    # Record the elapsed time for the current step size
    if VERBOSE:
        ending_time = (time.time_ns() - STARTING_TIME) / 1.0e9
        times.append(ending_time)
        print(f"elapsed time = {ending_time} seconds")

    # Interpolate the results to match the reference timepoints
    indices = [i + 1 for i in range(len(t_t) - 1) if t_t[i] > t_t[i + 1]]
    P_cycle = P_t[indices[-2] : indices[-1], :]
    t_cycle = t_t[indices[-2] : indices[-1]]
    P0_temp = np.loadtxt(
        "/home/diego/studies/uni/thesis_maths/openBF/test/"
        + network_name
        + "/"
        + network_name
        + "_results/"
        + vessel_names[0]
        + "_P.last"
    )
    t0 = P0_temp[:, 0] % cardiac_T

    COUNTER = 0
    t_new = np.zeros(len(timepoints))
    P_new = np.zeros((len(timepoints), 5 * N))
    for i in range(len(t_cycle) - 1):
        if t0[COUNTER] >= t_cycle[i] and t0[COUNTER] <= t_cycle[i + 1]:
            P_new[COUNTER, :] = (P_cycle[i, :] + P_cycle[i + 1, :]) / 2
            COUNTER += 1

    t_new = t_new[:-1]
    P_new = P_new[:-1, :]

    # Calculate the residual for the current step size
    RESIDUAL = 0
    for i, vessel_name in enumerate(vessel_names):
        index_vessel_name = vessel_names.index(vessel_name)
        P0_temp = np.loadtxt(
            "/home/diego/studies/uni/thesis_maths/openBF/test/"
            + network_name
            + "/"
            + network_name
            + "_results/"
            + vessel_name
            + "_P.last"
        )
        NODE = 2
        INDEX_JL = 1 + NODE
        index_jax = 5 * index_vessel_name + NODE

        P0 = P0_temp[:-1, INDEX_JL]
        t0 = P0_temp[:-1, 0] % cardiac_T
        P1 = P_new[:, index_jax]
        RESIDUAL += np.sqrt(((P1 - P0).dot(P1 - P0) / P0.dot(P0)))

    residuals.append(RESIDUAL / N)

# Plot the results: time and residuals vs. step size
_, ax = plt.subplots()
ax.set_xlabel("# steps")
ax1 = ax.twinx()
ln1 = ax.plot(steps, times, "g-", label="static t[s]")
ln2 = ax.plot(steps, np.ones(len(steps)) * ENDING_TIME_BASE, "r-", label="dynamic t[s]")
ln3 = ax1.plot(steps, residuals, "b-", label="static residual")  # type: ignore
ln4 = ax.plot(
    steps, np.ones(len(steps)) * RESIDUAL_BASE, "y-", label="dynamic residual"
)
ax.set_xlabel("# steps")
ax.set_ylabel("t[s]")
ax1.set_ylabel("residual")
lns = ln1 + ln2 + ln3 + ln4
labels = [ln.get_label() for ln in lns]
plt.title("network: " + network_name + ", # vessels: " + str(N))
plt.legend(lns, labels, loc="center right")
plt.tight_layout()

# Save the plot as a PDF file
plt.savefig(r_folder + "/steps_opt_" + network_name + ".pdf")
plt.close()
