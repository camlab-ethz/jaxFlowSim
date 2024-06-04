from src.model import configSimulation, simulationLoopUnsafe
from numpyro.infer.reparam import TransformReparam
import os
import sys
import time
from functools import partial
from jax import jit
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC
import numpy as np
import itertools

os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)

numpyro.set_host_device_count(1)

config_filename = ""
if len(sys.argv) == 1:

    # base cases
    #modelname = "single-artery"
    #modelname = "tapering"
    #modelname = "conjunction"
    #modelname = "bifurcation"
    #modelname = "aspirator"

    # openBF-hub 
    modelname = "test/adan56/adan56.yml"

    # vascularmodels.com
    #modelname = "0007_H_AO_H"
    #modelname = "0029_H_ABAO_H"
    #modelname = "0053_H_CERE_H"
    input_filename = "test/" + modelname + "/" + modelname + ".yml"

else:
    config_filename = "test/" + sys.argv[1] + "/" + sys.argv[1] + ".yml"



verbose = True
(N, B, J, 
 sim_dat, sim_dat_aux, 
 sim_dat_const, sim_dat_const_aux, 
 timepoints, conv_tol, Ccfl, edges, input_data, rho, 
 masks, strides, edges,
 vessel_names, cardiac_T) = configSimulation(config_filename, verbose)

sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 12))(simulationLoopUnsafe)
sim_dat_obs, t_obs, P_obs = sim_loop_old_jit(N, B,
                                      sim_dat, sim_dat_aux, 
                                      sim_dat_const, sim_dat_const_aux, 
                                      Ccfl, input_data, rho, 
                                      masks, strides, edges,
                                      upper=120000)

R_index = 1
var_index = 7
R1 = sim_dat_const[var_index,strides[R_index,1]]
#R_scales = np.linspace(1.1*R1, 2*R1, 16)
R_scales = np.linspace(0.1*R1, 10*R1, int(1e6))
def simLoopWrapper(R):
    ones = jnp.ones(strides[R_index,1]-strides[R_index,0]+4)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[var_index,strides[R_index,0]-2:strides[R_index,1]+2].set(R*ones)
    _, _, P = sim_loop_old_jit(N, B,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const_new, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges,
                                          upper=120000)
    return jnp.sqrt(jnp.sum(jnp.square((P-P_obs))))

sim_loop_wrapper_jit = jit(simLoopWrapper)

results_folder = "results/potential_surface"
if not os.path.isdir(results_folder):
    os.makedirs(results_folder, mode = 0o777)

slices = int(1e6/int(sys.argv[3]))

for i in range(int(sys.argv[2])*slices,(int(sys.argv[2])+1)*slices):
    results_file = results_folder  + "/potential_surface.txt"
    R = R_scales[i]
    loss = simLoopWrapper(R)
    file = open(results_file, "a")  
    file.write(str(R) + " " + str(loss) + "\n")
    file.close()