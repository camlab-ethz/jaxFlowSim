import jax.numpy as jnp
from jax import lax, jit, block_until_ready
import numpy as np
from src.initialise import (
    load_config,
    build_blood,
    build_arterial_network,
    make_results_folder,
)
from src.IOutils import save_temp_data
from src.solver import computeDt, solveModel
from src.check_conv import print_conf_error, compute_conv_error, check_conv
from functools import partial
import numpy as np
import numpyro
import time


numpyro.set_platform("cpu")
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32' # Use 32 CPU devices
# jax.devices("cpu")[0]
# print(jax.local_device_count())


def configSimulation(input_filename, verbose=False, make_results_folder_bool=True):
    data = load_config(input_filename)
    blood = build_blood(data["blood"])

    J = data["solver"]["num_snapshots"]

    (
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        N,
        B,
        masks,
        strides,
        edges,
        vessel_names,
        input_data,
    ) = build_arterial_network(
        data["network"], blood
    )  # , junction_functions) = buildArterialNetwork(data["network"], blood)
    if make_results_folder_bool:
        make_results_folder(data, input_filename)

    cardiac_T = sim_dat_const_aux[0, 0]
    Ccfl = float(data["solver"]["Ccfl"])

    if verbose:
        print("start simulation")

    timepoints = np.linspace(0, cardiac_T, J)

    return (
        N,
        B,
        J,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        timepoints,
        1,
        Ccfl,
        edges,
        input_data,
        blood.rho,
        masks,
        strides,
        edges,
        vessel_names,
        cardiac_T,
    )


def simulationLoopUnsafe(
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
    upper=100000,
):
    t = 0.0
    dt = 1
    P_t = jnp.zeros((upper, 5 * N))
    t_t = jnp.zeros(upper)

    def bodyFun(i, args):
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
            P_t,
        ) = args
        dt = computeDt(Ccfl, sim_dat[0, :], sim_dat[3, :], sim_dat_const[-1, :])
        sim_dat, sim_dat_aux = solveModel(
            N,
            B,
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
        P_t = P_t.at[i, :].set(save_temp_data(N, strides, sim_dat[4, :]))

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
            P_t,
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
        P_t,
    ) = lax.fori_loop(
        0,
        upper,
        bodyFun,
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
            P_t,
        ),
    )

    return sim_dat, t_t, P_t


def simulationLoop(
    N,
    B,
    num_snapshots,
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
):
    t = 0.0
    passed_cycles = 0
    counter = 0
    P_t = jnp.empty((num_snapshots, N * 5))
    t_t = jnp.empty(num_snapshots)
    P_l = jnp.empty((num_snapshots, N * 5))
    dt = 0

    def condFun(args):
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
            P_t_i,
            P_l_i,
            _,
            conv_tol,
            _,
            _,
            _,
            _,
        ) = args
        err = compute_conv_error(N, P_t_i, P_l_i)

        def printConvErrorWrapper():
            print_conf_error(err)
            return False

        ret = lax.cond(
            (passed_cycles_i + 1 > 1)
            * (check_conv(err, conv_tol))
            * (
                (
                    t_i - sim_dat_const_aux[0, 0] * passed_cycles_i
                    >= sim_dat_const_aux[0, 0]
                )
            ),
            printConvErrorWrapper,
            lambda: True,
        )
        return ret

    def bodyFun(args):
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
            P_t,
            P_l,
            t_t,
            _,
            Ccfl,
            edges,
            input_data,
            rho,
        ) = args
        dt = computeDt(Ccfl, sim_dat[0, :], sim_dat[3, :], sim_dat_const[-1, :])
        sim_dat, sim_dat_aux = solveModel(
            N,
            B,
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

        (P_t_temp, counter_temp) = lax.cond(
            t >= timepoints[counter],
            lambda: (save_temp_data(N, strides, sim_dat[4, :]), counter + 1),
            lambda: (P_t[counter, :], counter),
        )
        P_t = P_t.at[counter, :].set(P_t_temp)
        t_t = t_t.at[counter].set(t)
        counter = counter_temp

        def checkConv():
            err = compute_conv_error(N, P_t, P_l)
            print_conf_error(err)

        lax.cond(
            (
                (t - sim_dat_const_aux[0, 0] * passed_cycles >= sim_dat_const_aux[0, 0])
                * (passed_cycles + 1 > 1)
            ),
            checkConv,
            lambda: None,
        )
        (P_l, counter, timepoints, passed_cycles) = lax.cond(
            (t - sim_dat_const_aux[0, 0] * passed_cycles >= sim_dat_const_aux[0, 0]),
            lambda: (P_t, 0, timepoints + sim_dat_const_aux[0, 0], passed_cycles + 1),
            lambda: (P_l, counter, timepoints, passed_cycles),
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
            P_t,
            P_l,
            t_t,
            conv_tol,
            Ccfl,
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
        P_t,
        P_l,
        t_t,
        conv_tol,
        Ccfl,
        edges,
        input_data,
        rho,
    ) = lax.while_loop(
        condFun,
        bodyFun,
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
            P_t,
            P_l,
            t_t,
            conv_tol,
            Ccfl,
            edges,
            input_data,
            rho,
        ),
    )

    return sim_dat, t_t, P_t


def runSimulationUnsafe(config_filename, verbose=False, make_results_folder=True):
    (
        N,
        B,
        _,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        _,
        _,
        Ccfl,
        edges,
        input_data,
        rho,
        masks,
        strides,
        edges,
        _,
        _,
    ) = configSimulation(config_filename, verbose, make_results_folder)

    if verbose:
        starting_time = time.time_ns()

    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 12))(simulationLoopUnsafe)
    sim_dat, t, P = block_until_ready(
        sim_loop_old_jit(
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

    if verbose:
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"elapsed time = {ending_time} seconds")
    return sim_dat, t, P


def runSimulation(config_filename, verbose=False, make_results_folder=True):
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
        edges,
        input_data,
        rho,
        masks,
        strides,
        edges,
        _,
        _,
    ) = configSimulation(config_filename, verbose, make_results_folder)

    if verbose:
        starting_time = time.time_ns()
    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulationLoop)
    sim_dat, t, P = block_until_ready(
        sim_loop_old_jit(
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

    if verbose:
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"elapsed time = {ending_time} seconds")

    return sim_dat, t, P
