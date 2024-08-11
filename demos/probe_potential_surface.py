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
import numpyro  # type: ignore
from jax import jit, lax, jacfwd
from jax.test_util import check_grads
from jax.tree_util import Partial

from src.model import config_simulation, simulation_loop_unsafe

os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)

numpyro.set_host_device_count(1)

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
) = config_simulation(CONFIG_FILENAME, VERBOSE)

NUM_ITERATIONS = 1000
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)
sim_dat_obs, t_obs, P_obs = SIM_LOOP_JIT(  # pylint: disable=E1102
    N,
    B,
    sim_dat,
    sim_dat_aux,
    sim_dat_const,
    sim_dat_const_aux,
    0.5,
    input_data,
    rho,
    masks,
    strides,
    edges,
    NUM_ITERATIONS,
)


R_INDEX = 1
VAR_INDEX = 7
R1 = sim_dat_const[VAR_INDEX, strides[R_INDEX, 1]]
TOTAL_NUM_POINTS = 1e3
R_scales = np.linspace(0.1, 10, int(TOTAL_NUM_POINTS))


def sim_loop_wrapper(
    r,
    r1,
    var_index,
    p_obs,
    n,
    b,
    m,
    start_in,
    sim_dat_in,
    sim_dat_aux_in,
    sim_dat_const_in,
    sim_dat_const_aux_in,
    input_data_in,
    rho_in,
    masks_in,
    strides_in,
    edges_in,
):
    """
    Wrapper for the simulation loop to evaluate the results.

    Args:
        r: Scaling factor for the variable.
        r1: Initial value of the variable.
        var_index: Index of the variable in the simulation data.
        p_obs: Observed pressure data.
        n, b, m: Simulation parameters.
        start_in, sim_dat_in, sim_dat_aux_in, sim_dat_const_in, sim_dat_const_aux_in, input_data_in, rho_in, masks_in, strides_in, edges_in: Simulation inputs.

    Returns:
        Normalized root mean square error of the pressure data.
    """
    r = r * r1
    ones = jnp.ones(m)

    sim_dat_const_new = lax.dynamic_update_slice(
        sim_dat_const_in,
        ((r * ones)[:, jnp.newaxis] * jnp.ones(1)[jnp.newaxis, :]).transpose(),
        (var_index, start_in),
    )
    _, _, p = SIM_LOOP_JIT(  # pylint: disable=E1102
        n,
        b,
        sim_dat_in,
        sim_dat_aux_in,
        sim_dat_const_new,
        sim_dat_const_aux_in,
        0.5,
        input_data_in,
        rho_in,
        masks_in,
        strides_in,
        edges_in,
        120000,
    )
    return jnp.sqrt(jnp.sum(jnp.square((p - p_obs)))) / jnp.sqrt(
        jnp.sum(jnp.square((p_obs)))
    )


def sin_loop_wrapper1(
    n,
    b,
    sim_dat_in,
    sim_dat_aux_in,
    sim_dat_const_in,
    sim_dat_const_aux_in,
    ccfl_in,
    input_data_in,
    rho_in,
    masks_in,
    strides_in,
    edges_in,
):
    """
    Alternate wrapper for the simulation loop.

    Args:
        n, b: Simulation parameters.
        sim_dat_in, sim_dat_aux_in, sim_dat_const_in, sim_dat_const_aux_in, ccfl_in, input_data_in, rho_in, masks_in, strides_in, edges_in: Simulation inputs.

    Returns:
        Updated simulation data.
    """
    sim_dat_in, _, _ = SIM_LOOP_JIT(  # pylint: disable=E1102
        n,
        b,
        sim_dat_in,
        sim_dat_aux_in,
        sim_dat_const_in,
        sim_dat_const_aux_in,
        ccfl_in,
        input_data_in,
        rho_in,
        masks_in,
        strides_in,
        edges_in,
        120000,
    )
    return sim_dat_in


SIM_LOOP_WRAPPER_JIT = partial(jit, static_argnums=(1, 2))(sim_loop_wrapper)
SIM_LOOP_WRAPPER_GRAD_JIT = partial(jit, static_argnums=(2, 6, 7, 8, 9, 10))(
    jacfwd(sim_loop_wrapper, 14)
)

RESULTS_FOLDER = "results/potential_surface"
if not os.path.isdir(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER, mode=0o777)

slices = int(TOTAL_NUM_POINTS / int(sys.argv[3]))
gradients = np.zeros(slices)
gradients_averaged = np.zeros(slices)
GRADIENT = 1

for i in range(int(sys.argv[2]) * slices, (int(sys.argv[2]) + 1) * slices):
    RESULTS_FILE = RESULTS_FOLDER + "/potential_surface_new.txt"
    R = R_scales[i]
    M = strides[R_INDEX, 1] - strides[R_INDEX, 0] + 4
    start = strides[R_INDEX, 0] - 2
    end = strides[R_INDEX, 1] + 2
    loss = SIM_LOOP_WRAPPER_JIT(  # pylint: disable=E1102
        R,
        R_INDEX,
        R1,
        VAR_INDEX,
        P_obs,
        N,
        B,
        M,
        start,
        end,
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
    )
    sim_loop_wrapper_R = Partial(
        SIM_LOOP_WRAPPER_JIT,
        R_index=R_INDEX,
        R1=R1,
        var_index=VAR_INDEX,
        P_obs=P_obs,
        N=N,
        B=B,
        M=M,
        start=start,
        end=end,
        sim_dat=sim_dat,
        sim_dat_aux=sim_dat_aux,
        sim_dat_const=sim_dat_const,
        sim_dat_const_aux=sim_dat_const_aux,
        Ccfl=Ccfl,
        input_data=input_data,
        rho=rho,
        masks=masks,
        strides=strides,
        edges=edges,
    )

    M = strides[R_INDEX, 1] - strides[R_INDEX, 0] + 4
    start = strides[R_INDEX, 0] - 2
    end = strides[R_INDEX, 1] + 2
    gradients[i - int(sys.argv[2]) * slices] = (
        SIM_LOOP_WRAPPER_GRAD_JIT(  # pylint: disable=E1102
            R,
            R_INDEX,
            R1,
            VAR_INDEX,
            P_obs,
            N,
            B,
            M,
            start,
            end,
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
        )
    )
    print(loss, gradients[i - int(sys.argv[2]) * slices])
    check_grads(sim_loop_wrapper_R, (R,), order=1, atol=1e-2, rtol=1e-2, modes="fwd")

    if i >= int(sys.argv[2]) * slices + 1000:
        gradients_averaged[i - int(sys.argv[2]) * slices] = gradients[
            i - int(sys.argv[2]) * slices - 999 : i - int(sys.argv[2]) * slices + 1
        ].mean()
        gradients_averaged[i - int(sys.argv[2]) * slices] = gradients[
            i - int(sys.argv[2]) * slices - 999 : i - int(sys.argv[2]) * slices + 1
        ].mean()
    else:
        gradients_averaged[i - int(sys.argv[2]) * slices] = gradients[
            : i - int(sys.argv[2]) * slices + 1
        ].mean()

    file = open(RESULTS_FILE, "a", encoding="utf-8")
    file.write(
        str(R)
        + " "
        + str(loss)
        + " "
        + str(gradients[i - int(sys.argv[2]) * slices])
        + " "
        + str(gradients_averaged[i - int(sys.argv[2]) * slices])
        + "\n"
    )
    file.close()
