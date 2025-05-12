import os
import sys
import time
from functools import partial

import jax
from jax import jit, block_until_ready

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop

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
) = config_simulation(CONFIG_FILENAME)

# Initialize timing variables
STARTING_TIME = time.time_ns()

# Set up and execute the simulation loop using JIT compilation
SIM_LOOP_JIT = block_until_ready(
    partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
)

# If verbose mode is enabled, calculate and print the elapsed time
ENDING_TIME = (time.time_ns() - STARTING_TIME) / 1.0e9
print(f"{N} {ENDING_TIME}")
