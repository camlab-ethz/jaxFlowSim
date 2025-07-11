"""
This script configures and runs a JAX-based simulation of bifurcation, evaluates the results,
and computes gradients for potential surface analysis.

The script performs the following steps:
1. Sets up the simulation environment and loads the configuration.
2. Runs the simulation for a specified number of iterations.
3. Defines and runs a wrapper for the simulation loop to evaluate and compute gradients.
4. Saves the results and gradients to a file.

Constants:
- `CONFIG_FILENAME`: Path to the configuration file for the simulation.
- `RESULTS_FOLDER`: Directory to store the results.
- `RESULTS_FILE`: File to store the results.
- `NUM_ITERATIONS`: Number of iterations for the simulation.
- `TOTAL_NUM_POINTS`: Total number of points for potential surface analysis.

Functions:
- `config_simulation`: Configures the simulation environment.
- `simulation_loop_unsafe`: Runs the simulation loop.
- `sim_loop_wrapper`: Wrapper for the simulation loop to evaluate results.
- `sin_loop_wrapper1`: Alternate wrapper for the simulation loop.
- `check_grads`: Checks the gradients for correctness.
- `partial`: Partially applies arguments to a function.
- `jit`: Compiles a function using Just-In-Time compilation.
- `jacfwd`: Computes forward-mode Jacobian of a function.

"""

import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, jacfwd
import matplotlib.pyplot as plt
# from jax.test_util import check_grads

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe

os.chdir(os.path.dirname(__file__) + "/..")
jax.config.update("jax_enable_x64", True)


CONFIG_FILENAME = "test/bifurcation/bifurcation.yml"


VERBOSE = True
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

Ccfl = 0.5
NUM_ITERATIONS = 1000
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
    50000,
)

# Indices for selecting specific parts of the simulation data
VESSEL_INDEX_1 = 1
VAR_INDEX_1 = 4
R_INDEX = 4

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
        50000,
    )
    return jnp.mean(
        jnp.power(jnp.linalg.norm(p - P_obs, ord=None, axis=0), 2)
        / jnp.power(jnp.linalg.norm(P_obs, ord=None, axis=0), 2)
    )


TOTAL_NUM_POINTS = int(10)
R_scales = jnp.linspace(0.8, 1.2, int(TOTAL_NUM_POINTS))


SIM_LOOP_WRAPPER_JIT = partial(jit)(sim_loop_wrapper)
SIM_LOOP_WRAPPER_GRAD_JIT = partial(jit)(jacfwd(sim_loop_wrapper, 0))

RESULTS_FOLDER = "results/potential_surface"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)

if len(sys.argv) > 1:
    slices = int(TOTAL_NUM_POINTS / int(sys.argv[2]))
else:
    slices = TOTAL_NUM_POINTS
gradients = np.zeros(slices)
gradients_averaged = np.zeros(slices)
GRADIENT = 1

if len(sys.argv) > 1:
    RANGE = range(int(sys.argv[1]) * slices, (int(sys.argv[1]) + 1) * slices)
else:
    RANGE = range(slices)

# vectorized evaluation of all gradients
print("Evaluating gradients...")
gradients = jax.jit(jax.vmap(jax.jacfwd(sim_loop_wrapper), in_axes=(0,)))(R_scales)
jax.block_until_ready(gradients)
print("Gradients evaluated.")

# vectorized evaluation of values
print("Evaluating values...")
values = jax.jit(jax.vmap(sim_loop_wrapper, in_axes=(0,)))(R_scales)
jax.block_until_ready(values)
print("Values evaluated.")

# print("Checking gradients...")
# jax.vmap(
#    lambda x: check_grads(
#        sim_loop_wrapper, x, order=1, atol=1e-2, rtol=1e-2, modes="fwd"
#    ),
#    in_axes=(0,),
# )(R_scales)
# print out gradients
# check_grads(
#    sim_loop_wrapper, tuple(R_scales), order=1, atol=1e-2, rtol=1e-2, modes="fwd"
# )
# print("All gradients checked.")

# plot values and gradients
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(R_scales, values, label="loss", color="blue")
# create second y-axis for gradients
ax2.set_ylabel("gradient")
ax2.plot(R_scales, gradients, label="gradient", color="orange")
ax.set_xlabel("resistance scale of windkessel model")
ax.set_ylabel("loss")
plt.title("Potential Surface Analysis")
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
plt.show()
print("Gradients:")
for i in RANGE:
    print(f"R: {R_scales[i]}, Gradient: {gradients[i]}")
