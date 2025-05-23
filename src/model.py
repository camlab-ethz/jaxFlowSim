"""
Simulation configuration and execution for vascular network models using JAX.

This module handles:

- Loading user configurations and building the arterial network.
- Setting up blood rheology and numerical solver parameters.
- Running time-stepping simulations with optional convergence checks.
- Providing both "unsafe" (no convergence monitoring) and "safe" (with convergence
  diagnostics) execution paths, each JIT-compiled for performance.

Key functions
-------------
config_simulation
    Load configuration, build network and blood properties, and prepare simulation data.
simulation_loop_unsafe
    Execute the main time-stepping loop without convergence checks, capturing pressure snapshots.
simulation_loop
    Execute the main time-stepping loop with convergence diagnostics and early stopping.
run_simulation_unsafe
    High-level entry point for unsafe simulation: loads config, JIT-compiles, and runs.
run_simulation
    High-level entry point for safe simulation: loads config, JIT-compiles, and runs.

Dependencies
------------
- JAX (jax.numpy, jax.lax)                 : Array operations and control flow
- NumPy                                    : Configuration utilities and timepoint generation
- NumPyro                                  : Platform configuration
- jaxtyping.jaxtyped, beartype.beartype     : Static and runtime type enforcement
- src.check_conv                           : Convergence error computation and reporting
- src.initialise                           : Network and blood setup, configuration loading
- src.IOutils                              : Pressure data sampling utilities
- src.solver                               : Time-step computation and blood flow solver
- src.types                                : Data type aliases for simulation arrays
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

# Use CPU platform for JAX-NumPyro computations
numpyro.set_platform("cpu")
# To force multiple CPU devices, uncomment and adjust as needed:
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
    Load config, build network and blood, and initialize simulation data arrays.

    Parameters
    ----------
    input_filename : String
        Path to the YAML configuration file defining network, solver, and I/O.
    make_results_folder_bool : StaticBool, optional
        If True, create a timestamped results folder alongside config (default: True).

    Returns
    -------
    n : StaticScalarInt
        Number of vessels in the arterial network.
    b : StaticScalarInt
        Buffer size for time-stepping history.
    j : StaticScalarInt
        Number of snapshots/timepoints to record.
    sim_dat : StaticSimDat
        Initial simulation state array (e.g., [area, velocity, pressure...]).
    sim_dat_aux : StaticSimDatAux
        Auxiliary state variables for the solver.
    sim_dat_const : StaticSimDatConst
        Constant parameters per vessel (e.g., compliance coefficients).
    sim_dat_const_aux : StaticSimDatConstAux
        Global constants (e.g., cardiac period).
    timepoints : StaticTimepoints
        Linearly spaced timepoints over one cardiac cycle.
    init_snapshot_index : StaticScalarInt
        Starting index for snapshots (always 1).
    ccfl : StaticScalarFloat
        CFL number controlling time-step size.
    input_data : StaticInputData
        External inflow/outflow boundary data.
    rho : StaticScalarFloat
        Blood density (kg/m³).
    masks : StaticMasks
        Node masks for boundary condition enforcement.
    strides : StaticStrides
        Stride indices for sampling pressure along vessels.
    edges : StaticEdges
        Connectivity matrix of the arterial graph.
    vessel_names : Strings
        List of vessel identifiers corresponding to `edges`.
    cardiac_period : StaticScalarFloat
        Duration of one cardiac cycle (seconds).
    """
    # Load YAML configuration
    data = load_config(input_filename)
    # Build blood properties object
    blood = build_blood(data["blood"])

    # Number of pressure snapshots to record
    j = data["solver"]["num_snapshots"]

    # Construct arterial network and retrieve simulation arrays
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

    # Optionally create results directory for output files
    if make_results_folder_bool:
        make_results_folder(data, input_filename)

    # Extract cardiac cycle duration from constants
    cardiac_t = sim_dat_const_aux[0, 0]
    # CFL condition coefficient
    ccfl = float(data["solver"]["Ccfl"])

    # Create evenly spaced timepoints over one cycle
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
    Run time-stepping loop without convergence monitoring.

    At each step:
      1. Compute dt via CFL condition.
      2. Advance states via `solve_model`.
      3. Update simulation time and record snapshot of pressures.

    Parameters
    ----------
    n : StaticScalarInt
        Number of vessels.
    b : StaticScalarInt
        Buffer size for history arrays.
    sim_dat : SimDat
        Current simulation state.
    sim_dat_aux : SimDatAux
        Auxiliary state variables.
    sim_dat_const : SimDatConst
        Constant per-vessel parameters.
    sim_dat_const_aux : SimDatConstAux
        Global constant parameters.
    ccfl : ScalarFloat
        CFL number for dt selection.
    input_data : InputData
        External boundary condition data.
    rho : ScalarFloat
        Blood density.
    masks : Masks
        Boolean masks for boundary enforcement.
    strides : Strides
        Sampling strides for pressure snapshots.
    edges : Edges
        Connectivity graph of vessels.
    upper : StaticScalarInt, optional
        Maximum number of time steps (default: 100000).

    Returns
    -------
    sim_dat : SimDat
        Final simulation state.
    t_t : TimepointsReturn
        Recorded times of each snapshot (shape: [upper]).
    p_t : PressureReturn
        Recorded pressure snapshots (shape: [upper, 5*n]).
    """
    # Initialize loop variables
    t: Float = 0.0
    dt: Float = 1.0
    p_t: PressureReturn = jnp.zeros((upper, 5 * n))
    t_t: TimepointsReturn = jnp.zeros(upper)

    def simulation_step(
        i: ScalarInt, args: SimulationStepArgsUnsafe
    ) -> SimulationStepArgsUnsafe:
        """
        Single time step without convergence check.
        """
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

        # 1) Compute time-step based on current wave speeds and CFL
        dt = compute_dt(ccfl, sim_dat[0, :], sim_dat[3, :], sim_dat_const[-1, :])
        # 2) Advance flow model one step
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
        # 3) Update simulation time (wrap around cardiac cycle)
        t = (t + dt) % sim_dat_const_aux[0, 0]
        # 4) Record time and sample pressures
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

    # Execute fixed-number loop
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
    Run time-stepping loop with convergence checks and early exit.

    Iterates until the computed error between successive cardiac cycles
    falls below `conv_tol`, or until sufficient cycles have passed.
    Captures pressure snapshots at specified `timepoints`.

    Parameters
    ----------
    n : StaticScalarInt
        Number of vessels.
    b : StaticScalarInt
        Buffer size for history arrays.
    num_snapshots : StaticScalarInt
        Number of timepoints per cycle to record.
    sim_dat : SimDat
        Current simulation state.
    sim_dat_aux : SimDatAux
        Auxiliary state variables.
    sim_dat_const : SimDatConst
        Constant per-vessel parameters.
    sim_dat_const_aux : SimDatConstAux
        Global constants (e.g., cycle period).
    timepoints : Timepoints
        Array of times at which to sample pressures.
    conv_tol : ScalarFloat
        Convergence tolerance in mmHg.
    ccfl : ScalarFloat
        CFL number for dt selection.
    input_data : InputData
        External boundary condition data.
    rho : ScalarFloat
        Blood density.
    masks : Masks
        Boolean masks for boundary enforcement.
    strides : Strides
        Sampling strides for pressure snapshots.
    edges : Edges
        Connectivity graph of vessels.

    Returns
    -------
    sim_dat : SimDat
        Final simulation state.
    t_t : TimepointsReturn
        Recorded times of snapshots (shape: [num_snapshots]).
    p_t : PressureReturn
        Recorded pressure snapshots (shape: [num_snapshots, 5*n]).
    """
    # Initialize loop state
    t: Float = 0.0
    passed_cycles: Integer = 0
    counter: Integer = 0
    p_t: PressureReturn = jnp.empty((num_snapshots, n * 5))
    t_t: TimepointsReturn = jnp.empty(num_snapshots)
    p_l: PressureReturn = jnp.empty((num_snapshots, n * 5))
    dt: Float = 1.0

    def conv_error_condition(args: SimulationStepArgs) -> StaticBool:
        """
        Continue looping while convergence criteria not met.
        """
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

        # Compute maximum L2 error between this cycle and last cycle
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
        """
        Single time step with convergence bookkeeping and snapshot logic.
        """
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

        # 1) Compute dt and advance solution
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

        # 2) If we've reached next snapshot time, record pressure
        (p_t_temp, counter_temp) = lax.cond(
            t >= timepoints[counter],
            lambda: (save_temp_data(n, strides, sim_dat[4, :]), counter + 1),
            lambda: (p_t[counter, :], counter),
        )
        p_t = p_t.at[counter, :].set(p_t_temp)
        t_t = t_t.at[counter].set(t)
        counter = counter_temp

        # 3) At end of cardiac cycle, shift snapshots into p_l and reset
        def print_conv_error_wrapper():
            # Print diagnostics at cycle boundary
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

        # 4) Advance time
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

    # Run while-loop until convergence condition returns False
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
    Execute an "unsafe" simulation (no convergence checks) from configuration.

    This function loads the configuration, sets up the model, JIT-compiles
    `simulation_loop_unsafe`, and runs it to completion.

    Parameters
    ----------
    config_filename : String
        Path to the YAML configuration file.
    make_results_folder_bool : StaticBool, optional
        If True, create results folder (default: True).

    Returns
    -------
    sim_dat : SimDat
        Final simulation state.
    t_t : TimepointsReturn
        Recorded times for each snapshot.
    p   : PressureReturn
        Recorded pressure snapshots.
    """
    # Load and configure simulation
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

    # JIT-compile the unsafe loop (n, b, and upper bound are static)
    sim_loop_unsafe_jit = partial(jit, static_argnums=(0, 1, 12))(
        simulation_loop_unsafe
    )

    # Execute and block until GPU/CPU finish
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
    Execute a "safe" simulation (with convergence monitoring) from configuration.

    Loads configuration, builds model, JIT-compiles `simulation_loop`, and runs
    until convergence criteria are met.

    Parameters
    ----------
    config_filename : String
        Path to the YAML configuration file.
    make_results_folder_bool : StaticBool, optional
        If True, create results folder (default: True).

    Returns
    -------
    sim_dat : SimDat
        Final simulation state.
    t_t : TimepointsReturn
        Recorded times for each snapshot.
    p   : PressureReturn
        Recorded pressure snapshots.
    """
    # Load and configure simulation
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

    # JIT-compile the safe loop (n, b, j static)
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
