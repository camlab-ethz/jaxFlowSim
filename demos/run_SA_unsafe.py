"""
This script performs sensitivity analysis on a cardiovascular simulation model using the SALib library
and JAX for efficient computation. The sensitivity analysis involves varying parameters of the model
within specified bounds and evaluating the effect on the output. The script supports configuration via
command-line arguments and runs the simulation using JIT compilation for improved performance.

Key functionalities include:
- Configuring and running a cardiovascular simulation model with JAX.
- Performing sensitivity analysis by varying model parameters.
- Evaluating the impact of parameter variations on the model output using SALib.
- Supporting the use of Sobol sampling for sensitivity analysis.

Modules:
- `os`, `sys`, `time`: Standard Python libraries for file handling, system operations, and timing.
- `jax`: A library for high-performance machine learning research.
- `jax.numpy`: JAX's version of NumPy, with support for automatic differentiation and GPU/TPU acceleration.
- `SALib`: A library for performing global sensitivity analyses.
- `numpy`: A fundamental package for scientific computing with Python.

Execution:
- The script reads command-line arguments to determine which configuration file to use.
- It sets up the simulation environment, runs the simulation, and performs sensitivity analysis.
- The results of the sensitivity analysis are printed and can be saved for further analysis.
"""

import os
import sys
import time
from functools import partial

import jax
import numpy as np
from jax import block_until_ready, jit
from SALib import ProblemSpec  # type: ignore

sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe

# Change directory to the script's location
os.chdir(os.path.dirname(__file__) + "/..")

# Enable 64-bit precision in JAX for higher accuracy in numerical computations
jax.config.update("jax_enable_x64", True)

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

# Record the start time if verbose mode is enabled
if VERBOSE:
    starting_time = time.time_ns()

# Set up and execute the base simulation loop using JIT compilation
SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)
sim_dat_base, t_t_base, P_t_base = block_until_ready(
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

# Extract start and end indices for each vessel from strides
starts = strides[:, 0]
ends = strides[:, 1]


def sim_loop_jit_wrapper(ccfl_in, ls, r0s, es, r1s, r2s, ccs):
    """
    Wrapper function for running the simulation loop with modified model parameters.

    Args:
        ccfl_in (float): CFL condition factor.
        ls (array): Array of lengths for each vessel segment.
        r0s (array): Array of initial radii for each vessel segment.
        es (array): Array of elasticity moduli for each vessel segment.
        r1s (array): Array of peripheral resistance parameters for each vessel segment.
        r2s (array): Array of compliance parameters for each vessel segment.
        ccs (array): Array of compliance coefficients for each vessel segment.

    Returns:
        float: The norm of the difference between the base pressure and the new pressure.
    """
    for i in range(N):
        sim_dat_const[0, starts[i] - B : ends[i] + B] = (
            np.pi * r0s[i] * r0s[i] * np.ones(ends[i] - starts[i] + 2 * B)
        )
        sim_dat_const[3, starts[i] - B : ends[i] + B] = es[i] * np.ones(
            ends[i] - starts[i] + 2 * B
        )
        sim_dat_const[7, starts[i] - B : ends[i] + B] = r1s[i] * np.ones(
            ends[i] - starts[i] + 2 * B
        )
        sim_dat_const[8, starts[i] - B : ends[i] + B] = r2s[i] * np.ones(
            ends[i] - starts[i] + 2 * B
        )
        sim_dat_const[9, starts[i] - B : ends[i] + B] = ccs[i] * np.ones(
            ends[i] - starts[i] + 2 * B
        )
        sim_dat_const[10, starts[i] - B : ends[i] + B] = ls[i] * np.ones(
            ends[i] - starts[i] + 2 * B
        )

    _, _, p_t_new = SIM_LOOP_JIT(  # pylint: disable=E1102
        N,
        B,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        ccfl_in,
        input_data,
        rho,
        masks,
        strides,
        edges,
        upper=120000,
    )
    return np.linalg.norm(P_t_base - p_t_new)


# Define a grid for sensitivity analysis and print the scaled parameter arrays
grid = np.arange(0.1, 2, 0.5)
print(np.array([0, 6.8123e7, 6.8123e7])[np.newaxis, :] * grid[:, np.newaxis])

# Define the parameter ranges for sensitivity analysis
sensitivity_dict = {
    "rho": 1060.0 * grid,
    "Ccfl": 4.0e-3 * grid,
    "Ls": np.array([8.6e-2, 8.5e-2, 8.5e-2])[np.newaxis, :] * grid[:, np.newaxis],
    "R0s": np.array([0.758242250e-2, 0.5492e-2, 0.5492e-2])[np.newaxis, :]
    * grid[:, np.newaxis],
    "Es": np.array(
        [
            500.0e3,
            700.0e3,
            700.0e3,
        ]
    )[np.newaxis, :]
    * grid[:, np.newaxis],
    "R1s": np.array([0, 6.8123e7, 6.8123e7])[np.newaxis, :] * grid[:, np.newaxis],
    "R2s": np.array([0, 3.1013e9, 3.1013e9])[np.newaxis, :] * grid[:, np.newaxis],
    "Ccs": np.array([0, 3.6664e-10, 3.6664e-10])[np.newaxis, :] * grid[:, np.newaxis],
}


def quick_wrap(
    n,
    b,
    sim_dat_in,
    sim_dat_aux_in,
    sim_dat_const_in,
    sim_dat_const_aux_in,
    ccfl,
    edges_in,
    input_data_in,
    rho_in,
    strides_in,
    masks_in,
):
    """
    Quick wrapper function for running the simulation loop with a predefined set of inputs.

    Args:
        n (int): Number of vessels.
        b (int): Number of boundary points.
        sim_dat_in (Array): Input simulation data.
        sim_dat_aux_in (Array): Auxiliary simulation data.
        sim_dat_const_in (Array): Constant simulation data.
        sim_dat_const_aux_in (Array): Auxiliary constant simulation data.
        ccfl (float): CFL condition factor.
        edges_in (Array): Array defining the edges of the simulation domain.
        input_data_in (Array): Input data for the simulation.
        rho_in (float): Density of the fluid.
        strides_in (Array): Array defining the strides in the simulation.
        masks_in (Array): Array defining the masks for the simulation.

    Returns:
        float: The norm of the difference between the base pressure and the new pressure.
    """
    _, _, p_t_new = SIM_LOOP_JIT(  # pylint: disable=E1102
        n,
        b,
        sim_dat_in,
        sim_dat_aux_in,
        sim_dat_const_in,
        sim_dat_const_aux_in,
        ccfl,
        input_data_in,
        rho_in,
        masks_in,
        strides_in,
        edges_in,
        upper=120000,
    )
    return np.linalg.norm(P_t_base - p_t_new)


def wrapped_linear(x: np.ndarray, var_index=4, func=SIM_LOOP_JIT) -> np.ndarray:
    """
    Run the simulation for a series of inputs and compute the relative norm of the difference
    between the base and new pressures for each input.

    Args:
        x (np.ndarray): Array of inputs to the simulation.
        var_index (int): Index of the variable in the simulation data to compare.
        func (function): Function to run the simulation.

    Returns:
        np.ndarray: Array of relative norms for each input.
    """
    import numpy as np  # pylint: disable=W0621,W0404,C0415

    m, _ = x.shape
    results = np.empty(m)
    for i in range(m):
        (
            a01,
            a02,
            a03,
            beta1,
            beta2,
            beta3,
            gamma1,
            gamma2,
            gamma3,
            visct1,
            visct2,
            visct3,
            r11,
            r12,
            r21,
            r22,
            cc1,
            cc2,
            l1,
            l2,
            l3,
            rho_in,
            ccfl_in,
        ) = x[i, :]

        sim_dat_const[0, starts[0] - B : ends[0] + B] = a01 * np.ones(
            ends[0] - starts[0] + 2 * B
        )
        sim_dat_const[0, starts[1] - B : ends[1] + B] = a02 * np.ones(
            ends[1] - starts[1] + 2 * B
        )
        sim_dat_const[0, starts[2] - B : ends[2] + B] = a03 * np.ones(
            ends[2] - starts[2] + 2 * B
        )

        sim_dat_const[1, starts[0] - B : ends[0] + B] = beta1 * np.ones(
            ends[0] - starts[0] + 2 * B
        )
        sim_dat_const[1, starts[1] - B : ends[1] + B] = beta2 * np.ones(
            ends[1] - starts[1] + 2 * B
        )
        sim_dat_const[1, starts[2] - B : ends[2] + B] = beta3 * np.ones(
            ends[2] - starts[2] + 2 * B
        )

        sim_dat_const[2, starts[0] - B : ends[0] + B] = gamma1 * np.ones(
            ends[0] - starts[0] + 2 * B
        )
        sim_dat_const[2, starts[1] - B : ends[1] + B] = gamma2 * np.ones(
            ends[1] - starts[1] + 2 * B
        )
        sim_dat_const[2, starts[2] - B : ends[2] + B] = gamma3 * np.ones(
            ends[2] - starts[2] + 2 * B
        )

        sim_dat_const[5, starts[0] - B : ends[0] + B] = visct1 * np.ones(
            ends[0] - starts[0] + 2 * B
        )
        sim_dat_const[5, starts[1] - B : ends[1] + B] = visct2 * np.ones(
            ends[1] - starts[1] + 2 * B
        )
        sim_dat_const[5, starts[2] - B : ends[2] + B] = visct3 * np.ones(
            ends[2] - starts[2] + 2 * B
        )

        sim_dat_const[7, starts[1] - B : ends[1] + B] = r11 * np.ones(
            ends[1] - starts[1] + 2 * B
        )
        sim_dat_const[7, starts[2] - B : ends[2] + B] = r12 * np.ones(
            ends[2] - starts[2] + 2 * B
        )

        sim_dat_const[8, starts[1] - B : ends[1] + B] = r21 * np.ones(
            ends[1] - starts[1] + 2 * B
        )
        sim_dat_const[8, starts[2] - B : ends[2] + B] = r22 * np.ones(
            ends[2] - starts[2] + 2 * B
        )

        sim_dat_const[9, starts[1] - B : ends[1] + B] = cc1 * np.ones(
            ends[1] - starts[1] + 2 * B
        )
        sim_dat_const[9, starts[2] - B : ends[2] + B] = cc2 * np.ones(
            ends[2] - starts[2] + 2 * B
        )

        sim_dat_const[10, starts[0] - B : ends[0] + B] = l1 * np.ones(
            ends[0] - starts[0] + 2 * B
        )
        sim_dat_const[10, starts[1] - B : ends[1] + B] = l2 * np.ones(
            ends[1] - starts[1] + 2 * B
        )
        sim_dat_const[10, starts[2] - B : ends[2] + B] = l3 * np.ones(
            ends[2] - starts[2] + 2 * B
        )

        # Run the simulation with the updated parameters
        sim_dat_new, _, _ = func(
            N,
            B,
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            ccfl_in,
            input_data,
            rho_in,
            masks,
            strides,
            edges,
            upper=120000,
        )

        # Compute the relative norm of the difference between the base and new pressure values
        results[i] = np.linalg.norm(
            sim_dat_base[var_index, :] - sim_dat_new[var_index, :]
        ) / np.linalg.norm(sim_dat_base[var_index, :])

    return results


# Debugging print statements for the initial simulation constants
print(sim_dat_const[:, starts[0]])
print(sim_dat_const[:, starts[1]])
print(sim_dat_const[:, starts[2]])

# Define the problem specification for SALib sensitivity analysis
sp = ProblemSpec(
    {
        # 'names': ['R01', 'R02', 'R03', 'E1', 'E2', 'E3', 'R11', 'R12', 'R21', 'R22', 'Cc1', 'Cc2'], #, 'L1', 'L2', 'L3'],
        "names": [
            "A01",
            "A02",
            "A03",
            "beta1",
            "beta2",
            "beta3",
            "gamma1",
            "gamma2",
            "gamma3",
            "viscT1",
            "viscT2",
            "viscT3",
            "R11",
            "R12",
            "R21",
            "R22",
            "Cc1",
            "Cc2",
            "L1",
            "L2",
            "L3",
            "rho",
            "Ccfl",
        ],
        # 'names': ['R01', 'R02', 'R03', 'R11', 'R12', 'R21', 'R22', 'Cc1', 'Cc2'], #, 'L1', 'L2', 'L3'],
        "bounds": [
            [1.80619998e-04 * 0.9, 1.80619998e-04 * 1.1],
            [9.47569187e-05 * 0.9, 9.47569187e-05 * 1.1],
            [9.47569187e-05 * 0.9, 9.47569187e-05 * 1.1],
            [8.51668358e04 * 0.9, 8.51668358e04 * 1.1],
            [1.32543517e05 * 0.9, 1.32543517e05 * 1.1],
            [1.32543517e05 * 0.9, 1.32543517e05 * 1.1],
            [1.99278514e03 * 0.9, 1.99278514e03 * 1.1],
            [4.28179535e03 * 0.9, 4.28179535e03 * 1.1],
            [4.28179535e03 * 0.9, 4.28179535e03 * 1.1],
            [2.60811466e-04 * 0.9, 2.60811466e-04 * 1.1],
            [2.60811466e-04 * 0.9, 2.60811466e-04 * 1.1],
            [2.60811466e-04 * 0.9, 2.60811466e-04 * 1.1],
            # [0.758242250e-2*0.9, 0.758242250e-2*1.1],
            # [0.5492e-2*0.9, 0.5492e-2*1.1],
            # [0.5492e-2*0.9, 0.5492e-2*1.1],
            # [500.0e3*0.9999, 500.0e3*1],
            # [700.0e3*0.9999, 700.0e3*1],
            # [700.0e3*0.9999, 700.0e3*1],
            [6.8123e7 * 0.9, 6.8123e7 * 1.1],
            [6.8123e7 * 0.9, 6.8123e7 * 1.1],
            [3.1013e9 * 0.9, 3.1013e9 * 1.1],
            [3.1013e9 * 0.9, 3.1013e9 * 1.1],
            [3.6664e-10 * 0.9, 3.6664e-10 * 1.1],
            [3.6664e-10 * 0.9, 3.6664e-10 * 1.1],
            [1e-3 * 0.9, 1e-3 * 1.1],
            [1e-3 * 0.9, 1e-3 * 1.1],
            [1e-3 * 0.9, 1e-3 * 1.1],
            [rho * 0.9, rho * 1.1],
            [Ccfl * 0.9, Ccfl * 1.1],
        ],
    }
)

# Execute sensitivity analysis using Sobol sampling
if __name__ == "__main__":
    (sp.sample_sobol(2**8).evaluate(wrapped_linear).analyze_sobol())  # type: ignore

# Convert the problem specification to a DataFrame and print it
sp.to_df()
print(sp.to_df())
