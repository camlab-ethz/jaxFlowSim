"""
This module provides functions to configure and run vascular network simulations using JAX.

It includes functions to:
- Configure the simulation (`config_simulation`).
- Run the main simulation loop with and without safety checks (`simulation_loop`, `simulation_loop_unsafe`).
- Execute the simulation with JIT compilation for performance optimization (`run_simulation_unsafe`, `run_simulation`).

The module makes use of the following imported utilities:
- Functions from `src.check_conv` for convergence checks and error computations.
- Functions from `src.initialise` for building the network and blood properties, and loading configurations.
- `save_temp_data` from `src.IOutils` for saving temporary data.
- Functions from `src.solver` for time-step computation and solving the model.
- `jax.numpy` and `jax.lax` for numerical operations and control flow.
- `jaxtyping` and `beartype` for type checking and ensuring type safety in the functions.
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
import numpyro  # type: ignore
from jax import block_until_ready, jit, lax
from jaxtyping import Float, Integer, jaxtyped
from beartype import beartype as typechecker

from src.check_conv import check_conv, compute_conv_error, print_conv_error
from src.initialise import (
    build_arterial_network,
    build_blood,
    load_config,
    make_results_folder,
)
from src.IOutils import save_temp_data
from src.solver import compute_dt, solve_model
from src.types import (
    PressureReturn,
    SimulationStepArgs,
    SimulationStepArgsUnsafe,
    StaticScalarInt,
    SimDat,
    SimDatAux,
    SimDatConst,
    SimDatConstAux,
    ScalarFloat,
    ScalarInt,
    Strides,
    Edges,
    Masks,
    Timepoints,
    StaticSimDat,
    StaticSimDatAux,
    StaticSimDatConst,
    StaticSimDatConstAux,
    StaticTimepoints,
    StaticScalarFloat,
    StaticInputData,
    StaticMasks,
    StaticStrides,
    StaticEdges,
    String,
    Strings,
    StaticBool,
    InputData,
    TimepointsReturn,
)


numpyro.set_platform("cpu")
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32' # Use 32 CPU devices
# jax.devices("cpu")[0]
# print(jax.local_device_count())


@jaxtyped(typechecker=typechecker)
def config_simulation(
    input_filename: String, make_results_folder_bool: StaticBool = True
) -> tuple[
    StaticScalarInt,
    StaticScalarInt,
    StaticScalarInt,
    StaticSimDat,
    StaticSimDatAux,
    StaticSimDatConst,
    StaticSimDatConstAux,
    StaticTimepoints,
    StaticScalarInt,
    StaticScalarFloat,
    StaticInputData,
    StaticScalarFloat,
    StaticMasks,
    StaticStrides,
    StaticEdges,
    Strings,
    StaticScalarFloat,
]:
    """
    Configures the simulation by loading the configuration, building the blood properties, and arterial network.

    Parameters:
    input_filename (str): Path to the YAML configuration file.
    verbose (bool): Whether to print verbose output.
    make_results_folder_bool (bool): Whether to create a results folder.

    Returns:
    tuple: Configuration parameters for the simulation.
    """
    data = load_config(input_filename)
    blood = build_blood(data["blood"])

    j = data["solver"]["num_snapshots"]

    (
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        n,
        b,
        masks,
        strides,
        edges,
        vessel_names,
        input_data,
    ) = build_arterial_network(data["network"], blood)
    if make_results_folder_bool:
        make_results_folder(data, input_filename)

    cardiac_t = sim_dat_const_aux[0, 0]
    ccfl = float(data["solver"]["Ccfl"])

    timepoints = np.linspace(0, cardiac_t, j)

    return (
        n,
        b,
        j,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        timepoints,
        1,
        ccfl,
        input_data,
        blood.rho,
        masks,
        strides,
        edges,
        vessel_names,
        cardiac_t,
    )


@jaxtyped(typechecker=typechecker)
def simulation_loop_unsafe(
    n: StaticScalarInt,
    b: StaticScalarInt,
    sim_dat: SimDat,
    sim_dat_aux: SimDatAux,
    sim_dat_const: SimDatConst,
    sim_dat_const_aux: SimDatConstAux,
    ccfl: ScalarFloat,
    input_data: InputData,
    rho: ScalarFloat,
    masks: Masks,
    strides: Strides,
    edges: Edges,
    upper: StaticScalarInt = 100000,
) -> tuple[SimDat, TimepointsReturn, PressureReturn]:
    """
    Runs the simulation loop without convergence checks.

    Parameters:
    n (int): Number of vessels.
    b (int): Buffer size.
    sim_dat (Float[Array, "..."]): Simulation data array.
    sim_dat_aux (Float[Array, "..."]): Auxiliary simulation data array.
    sim_dat_const (Float[Array, "..."]): Constant simulation data array.
    sim_dat_const_aux (Float[Array, "..."]): Auxiliary constant simulation data array.
    ccfl (Float[Array, ""]): CFL condition value.
    input_data (Float[Array, "..."]): Input data array.
    rho (Float[Array, ""]): Blood density.
    masks (Integer[Array, "..."]): Masks array.
    strides (Integer[Array, "..."]): Strides array.
    edges (Integer[Array, "..."]): Edges array.
    upper (int): Upper limit for the loop.

    Returns:
    tuple: Updated simulation data, time steps, and pressure data.
    """
    t: Float = 0.0
    dt: Float = 1.0
    p_t: PressureReturn = jnp.zeros((upper, 5 * n))
    t_t: TimepointsReturn = jnp.zeros(upper)

    def simulation_step(
        i: ScalarInt, args: SimulationStepArgsUnsafe
    ) -> SimulationStepArgsUnsafe:
        (
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            dt,
            t,
            t_t,
            edges,
            input_data,
            rho,
            p_t,
        ) = args
        dt = compute_dt(ccfl, sim_dat[0, :], sim_dat[3, :], sim_dat_const[-1, :])
        sim_dat, sim_dat_aux = solve_model(
            n,
            b,
            t,
            dt,
            input_data,
            rho,
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            masks,
            strides[:, :2],
            edges,
        )
        t = (t + dt) % sim_dat_const_aux[0, 0]
        t_t = t_t.at[i].set(t)
        p_t = p_t.at[i, :].set(save_temp_data(n, strides, sim_dat[4, :]))

        return (
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            dt,
            t,
            t_t,
            edges,
            input_data,
            rho,
            p_t,
        )

    (
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        dt,
        t,
        t_t,
        edges,
        input_data,
        rho,
        p_t,
    ) = lax.fori_loop(
        0,
        upper,
        simulation_step,
        (
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            dt,
            t,
            t_t,
            edges,
            input_data,
            rho,
            p_t,
        ),
    )

    return sim_dat, t_t, p_t


@jaxtyped(typechecker=typechecker)
def simulation_loop(
    n: StaticScalarInt,
    b: StaticScalarInt,
    num_snapshots: StaticScalarInt,
    sim_dat: SimDat,
    sim_dat_aux: SimDatAux,
    sim_dat_const: SimDatConst,
    sim_dat_const_aux: SimDatConstAux,
    timepoints: Timepoints,
    conv_tol: ScalarFloat,
    ccfl: ScalarFloat,
    input_data: InputData,
    rho: ScalarFloat,
    masks: Masks,
    strides: Strides,
    edges: Edges,
) -> tuple[SimDat, TimepointsReturn, PressureReturn]:
    """
    Runs the main simulation loop with convergence checks.

    Parameters:
    n (int): Number of vessels.
    b (int): Buffer size.
    num_snapshots (int): Number of snapshots to capture.
    sim_dat (Float[Array, "..."]): Simulation data array.
    sim_dat_aux (Float[Array, "..."]): Auxiliary simulation data array.
    sim_dat_const (Float[Array, "..."]): Constant simulation data array.
    sim_dat_const_aux (Float[Array, "..."]): Auxiliary constant simulation data array.
    timepoints (Float[Array, "..."]): Timepoints array.
    conv_tol (Float[Array, ""]): Convergence tolerance.
    ccfl (Float[Array, ""]): CFL condition value.
    input_data (Float[Array, "..."]): Input data array.
    rho (Float[Array, ""]): Blood density.
    masks (Integer[Array, "..."]): Masks array.
    strides (Integer[Array, "..."]): Strides array.
    edges (Integer[Array, "..."]): Edges array.

    Returns:
    tuple: Updated simulation data, time steps, and pressure data.
    """
    t: Float = 0.0
    passed_cycles: Integer = 0
    counter: Integer = 0
    p_t: PressureReturn = jnp.empty((num_snapshots, n * 5))
    t_t: TimepointsReturn = jnp.empty(num_snapshots)
    p_l: PressureReturn = jnp.empty((num_snapshots, n * 5))
    dt: Float = 1.0

    def conv_error_condition(args: SimulationStepArgs) -> StaticBool:
        (
            _,
            _,
            _,
            sim_dat_const_aux,
            t_i,
            _,
            _,
            passed_cycles_i,
            _,
            p_t_i,
            p_l_i,
            _,
            conv_tol,
            _,
            _,
            _,
            _,
        ) = args
        err = compute_conv_error(n, p_t_i, p_l_i)

        def print_conv_error_wrapper():
            print_conv_error(err)
            return False

        ret = lax.cond(
            (passed_cycles_i + 1 > 1)
            * (check_conv(err, conv_tol))
            * (
                t_i - sim_dat_const_aux[0, 0] * passed_cycles_i
                >= sim_dat_const_aux[0, 0]
            ),
            print_conv_error_wrapper,
            lambda: True,
        )
        return ret

    def simulation_step(
        args: SimulationStepArgs,
    ) -> SimulationStepArgs:
        (
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            t,
            counter,
            timepoints,
            passed_cycles,
            dt,
            p_t,
            p_l,
            t_t,
            _,
            ccfl,
            edges,
            input_data,
            rho,
        ) = args
        dt = compute_dt(ccfl, sim_dat[0, :], sim_dat[3, :], sim_dat_const[-1, :])
        sim_dat, sim_dat_aux = solve_model(
            n,
            b,
            t,
            dt,
            input_data,
            rho,
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            masks,
            strides[:, :2],
            edges,
        )

        (p_t_temp, counter_temp) = lax.cond(
            t >= timepoints[counter],
            lambda: (save_temp_data(n, strides, sim_dat[4, :]), counter + 1),
            lambda: (p_t[counter, :], counter),
        )
        p_t = p_t.at[counter, :].set(p_t_temp)
        t_t = t_t.at[counter].set(t)
        counter = counter_temp

        def print_conv_error_wrapper():
            err = compute_conv_error(n, p_t, p_l)
            print_conv_error(err)

        lax.cond(
            (
                (t - sim_dat_const_aux[0, 0] * passed_cycles >= sim_dat_const_aux[0, 0])
                * (passed_cycles + 1 > 1)
            ),
            print_conv_error_wrapper,
            lambda: None,
        )
        (p_l, counter, timepoints, passed_cycles) = lax.cond(
            (t - sim_dat_const_aux[0, 0] * passed_cycles >= sim_dat_const_aux[0, 0]),
            lambda: (p_t, 0, timepoints + sim_dat_const_aux[0, 0], passed_cycles + 1),
            lambda: (p_l, counter, timepoints, passed_cycles),
        )

        t += dt

        return (
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            t,
            counter,
            timepoints,
            passed_cycles,
            dt,
            p_t,
            p_l,
            t_t,
            conv_tol,
            ccfl,
            edges,
            input_data,
            rho,
        )

    (
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        t,
        counter,
        timepoints,
        passed_cycles,
        dt,
        p_t,
        p_l,
        t_t,
        conv_tol,
        ccfl,
        edges,
        input_data,
        rho,
    ) = lax.while_loop(
        conv_error_condition,
        simulation_step,
        (
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            t,
            counter,
            timepoints,
            passed_cycles,
            dt,
            p_t,
            p_l,
            t_t,
            conv_tol,
            ccfl,
            edges,
            input_data,
            rho,
        ),
    )

    return sim_dat, t_t, p_t


def run_simulation_unsafe(
    config_filename: String,
    make_results_folder_bool: StaticBool = True,
) -> tuple[SimDat, TimepointsReturn, PressureReturn]:
    """
    Runs the simulation without convergence checks.

    Parameters:
    config_filename (str): Path to the YAML configuration file.
    verbose (bool): Whether to print verbose output.
    make_results_folder_bool (bool): Whether to create a results folder.

    Returns:
    tuple: Updated simulation data, time steps, and pressure data.
    """

    (
        n,
        b,
        _,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        _,
        _,
        ccfl,
        input_data,
        rho,
        masks,
        strides,
        edges,
        _,
        _,
    ) = config_simulation(config_filename, make_results_folder_bool)

    sim_loop_unsafe_jit = partial(jit, static_argnums=(0, 1, 12))(
        simulation_loop_unsafe
    )
    sim_dat, t, p = block_until_ready(
        sim_loop_unsafe_jit(  # pylint: disable=E1102
            n,
            b,
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            ccfl,
            input_data,
            rho,
            masks,
            strides,
            edges,
            upper=120000,
        )
    )

    return sim_dat, t, p


def run_simulation(
    config_filename: String,
    make_results_folder_bool: StaticBool = True,
) -> tuple[SimDat, TimepointsReturn, PressureReturn]:
    """
    Runs the simulation with convergence checks.

    Parameters:
    config_filename (str): Path to the YAML configuration file.
    verbose (bool): Whether to print verbose output.
    make_results_folder_bool (bool): Whether to create a results folder.

    Returns:
    tuple: Updated simulation data, time steps, and pressure data.
    """
    (
        n,
        b,
        j,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        timepoints,
        conv_tol,
        ccfl,
        input_data,
        rho,
        masks,
        strides,
        edges,
        _,
        _,
    ) = config_simulation(config_filename, make_results_folder_bool)

    simulation_loop_jit = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
    sim_dat, t, p = block_until_ready(
        simulation_loop_jit(
            n,
            b,
            j,
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            timepoints,
            float(conv_tol),
            ccfl,
            input_data,
            rho,
            masks,
            strides,
            edges,
        )
    )

    return sim_dat, t, p
