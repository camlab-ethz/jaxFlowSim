import jax
import sys
import time
import os
from functools import partial
from jax import block_until_ready, jit
from src.model import config_simulation, simulation_loop_unsafe

# os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)

print(sys.argv)
samples = int(sys.argv[1])
num_vessels_file = sys.argv[2]
network_names = sys.argv[3:]


for network_name in network_names:
    verbose = True
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
    ) = config_simulation("test/" + network_name + "/" + network_name + ".yml")

    # warmup step
    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)
    sim_dat, P_t, t_t = block_until_ready(
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

    for i in range(samples):
        if verbose:
            starting_time = time.time_ns()

        sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 12))(
            simulation_loop_unsafe
        )
        sim_dat, P_t, t_t = block_until_ready(
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

    file1 = open(num_vessels_file, "a")
    file1.write(str(N) + "\n")
    file1.close()
