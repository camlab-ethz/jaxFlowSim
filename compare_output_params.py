from src.model import config_simulation, simulation_loop
import jax
import os
from functools import partial
from jax import block_until_ready, jit
import matplotlib.pyplot as plt
import shutil

os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)

config_filename = "test/bifurcation/bifurcation.yml"
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
    vessel_names,
    cardiac_T,
) = config_simulation(config_filename)
sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
sim_dat1, t1, P1 = block_until_ready(
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

config_filename = "test/bifurcation2/bifurcation2.yml"
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
    vessel_names,
    cardiac_T,
) = config_simulation(config_filename)
sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
sim_dat2, t2, P2 = block_until_ready(
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

config_filename = "test/bifurcation3/bifurcation3.yml"
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
    vessel_names,
    cardiac_T,
) = config_simulation(config_filename)
sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
sim_dat3, t3, P3 = block_until_ready(
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

r_folder = "results/compare_output_params_results"
# delete existing folder and results
if os.path.isdir(r_folder):
    shutil.rmtree(r_folder)
os.makedirs(r_folder, mode=0o777)

filename = config_filename.split("/")[-1]
network_name = filename.split(".")[0]

# plt.rcParams.update({'font.size': 20})

for i, vessel_name in enumerate(vessel_names):
    index_vessel_name = vessel_names.index(vessel_name)
    node = 2
    index_jax = 5 * index_vessel_name + node
    _, ax = plt.subplots()
    ax.set_xlabel("t[s]")
    ax.set_ylabel("P[mmHg]")
    plt.plot(t1 % cardiac_T, P1[:, index_jax] / 133.322)
    plt.plot(t2 % cardiac_T, P2[:, index_jax] / 133.322)
    plt.plot(t3 % cardiac_T, P3[:, index_jax] / 133.322)
    plt.legend(["P_1", "P_2", "P_3"], loc="lower right")
    plt.tight_layout()
    plt.savefig(r_folder + "/compare_output_params_" + vessel_name + "_P.eps")
    plt.close()
