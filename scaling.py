from src.model import configSimulation, simulationLoop
import jax
import sys
import time
import os
import shutil
from functools import partial
from jax import block_until_ready, jit
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)




verbose = True
timeings = []
num_vessels = []
model_names = ["single-artery", "tapering", "conjunction", "bifurcation", "adan56", "0053_H_CERE_H", "0007_H_AO_H", "0029_H_ABAO_H"]
filenames = []
for model_name in model_names:
    config_filename = "test/" + model_name + "/" + model_name + ".yml" 
    filenames.append(config_filename)

for config_filename in filenames:
    (N, B, J, 
     sim_dat, sim_dat_aux, 
     sim_dat_const, sim_dat_const_aux, 
     timepoints, conv_tol, Ccfl, edges, input_data, rho, 
     masks, strides, edges,
     vessel_names, cardiac_T) = configSimulation(config_filename, verbose)
    if verbose:
        starting_time = time.time_ns()
    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulationLoop)
    _, _, _  = block_until_ready(sim_loop_old_jit(N, B, J, 
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          timepoints, conv_tol, Ccfl, input_data, rho, 
                                          masks, strides, edges))
    if verbose:
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"elapsed time = {ending_time} seconds")
    timeings.append(ending_time)
    num_vessels.append(len(vessel_names))

r_folder = "results/scaling_results"
# delete existing folder and results
if os.path.isdir(r_folder):
    shutil.rmtree(r_folder)
os.makedirs(r_folder, mode = 0o777)

filename = config_filename.split("/")[-1]
network_name = filename.split(".")[0]

#plt.rcParams.update({'font.size': 20})

_, ax = plt.subplots()
ax.set_xlabel("# vessels")
ax.set_ylabel("t[s]")
plt.scatter(num_vessels,timeings)
plt.tight_layout()
plt.savefig(r_folder + "/scaling.eps")
plt.close()
