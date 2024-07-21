from src.model import configSimulation, simulationLoopUnsafe
from jax.tree_util import Partial
from numpyro.infer.reparam import TransformReparam
import os
import sys
import time
from functools import partial
from jax import jit, lax
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC
import numpy as np
import itertools
from jax import jit, grad, jacfwd, value_and_grad
from jax.test_util import check_grads

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

num_iterations = 1000
sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 12))(simulationLoopUnsafe)
sim_dat_obs, t_obs, P_obs = sim_loop_old_jit(N, B,
                                      sim_dat, sim_dat_aux, 
                                      sim_dat_const, sim_dat_const_aux, 
                                      0.5, input_data, rho, 
                                      masks, strides, edges,
                                      num_iterations)





R_index = 1
var_index = 7
R1 = sim_dat_const[var_index,strides[R_index,1]]
#R_scales = np.linspace(1.1*R1, 2*R1, 16)
total_num_points = 1e3
R_scales = np.linspace(0.1, 10, int(total_num_points))
def simLoopWrapper(R, R_index, R1, var_index, P_obs, N, B, M, start, end,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges):
    R=R*R1
    ones = jnp.ones(M)
    
    #sim_dat_const_new = sim_dat_const_new.at[var_index, start:end].set(R*ones)
    sim_dat_const_new = lax.dynamic_update_slice(sim_dat_const,
                                                 ((R*ones)[:,jnp.newaxis]*jnp.ones(1)[jnp.newaxis,:]).transpose(),
                                                 (var_index, start))
    _, _, P = sim_loop_old_jit(N, B,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const_new, sim_dat_const_aux, 
                                          0.5, input_data, rho, 
                                          masks, strides, edges,
                                          120000)
    return jnp.sqrt(jnp.sum(jnp.square((P-P_obs))))/jnp.sqrt(jnp.sum(jnp.square((P_obs))))
def simLoopWrapper1(R, R_index, R1, var_index, P_obs, N, B, M, start, end,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges):
    sim_dat, _, P = sim_loop_old_jit(N, B,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges,
                                          120000)
                                          120000)
    return sim_dat

sim_loop_wrapper_jit = partial(jit, static_argnums=(1, 2))(simLoopWrapper)
#sim_loop_wrapper_grad_jit = partial(jit, static_argnums=(2, 6, 7, 8, 9, 10))(jacfwd(simLoopWrapper,14))
#sim_loop_wrapper_grad_jit1 = partial(jit, static_argnums=(1, 5, 6, 7, 8, 9))(value_and_grad(simLoopWrapper1,14))
#sim_loop_wrapper_grad_jit = jit(jacfwd(simLoopWrapper, 0))

results_folder = "results/potential_surface"
if not os.path.isdir(results_folder):
    os.makedirs(results_folder, mode = 0o777)

slices = int(total_num_points/int(sys.argv[3]))
gradients = np.zeros(slices)
gradients_averaged = np.zeros(slices)
gradient = 1
#for i in range(1):
#    results_file = results_folder  + "/potential_surface_new.txt"
#    R = R_scales[100]
#    M = strides[R_index,1]-strides[R_index,0]+4
#    start = strides[R_index,0]-2
#    end = strides[R_index,1]+2
#    sim_dat = simLoopWrapper1(R, R_index, R1, var_index, P_obs, N, B, M, start, end,
#                                          sim_dat, sim_dat_aux, 
#                                          sim_dat_const, sim_dat_const_aux, 
#                                          Ccfl, input_data, rho, 
#                                          masks, strides, edges)
#    gradient *= sim_loop_wrapper_grad_jit1(R, R_index, R1, var_index, P_obs, N, B, M, start, end,
#                                          sim_dat, sim_dat_aux, 
#                                          sim_dat_const, sim_dat_const_aux, 
#                                          Ccfl, input_data, rho, 
#                                          masks, strides, edges)
#    #gradient *= sim_loop_wrapper_grad_jit(R, R_index, R1, var_index, P_obs, N, B, M, start, end,
#    #                                      sim_dat, sim_dat_aux, 
#    #                                      sim_dat_const, sim_dat_const_aux, 
#    #                                      Ccfl, input_data, rho, 
#    #                                      masks, strides, edges)
#    print(gradient)


for i in range(int(sys.argv[2])*slices,(int(sys.argv[2])+1)*slices):
    results_file = results_folder  + "/potential_surface_new.txt"
    R = R_scales[i]
    M = strides[R_index,1]-strides[R_index,0]+4
    start = strides[R_index,0]-2
    end = strides[R_index,1]+2
    loss = sim_loop_wrapper_jit(R, R_index, R1, var_index, P_obs, N, B, M, start, end,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges)
    sim_loop_wrapper_R = Partial(sim_loop_wrapper_jit, R_index=R_index, R1=R1, var_index=var_index, P_obs=P_obs, N=N, B=B, M=M, start=start, end=end,
                                          sim_dat=sim_dat, sim_dat_aux=sim_dat_aux, 
                                          sim_dat_const=sim_dat_const, sim_dat_const_aux=sim_dat_const_aux, 
                                          Ccfl=Ccfl, input_data=input_data, rho=rho, 
                                          masks=masks, strides=strides, edges=edges)

    M = strides[R_index,1]-strides[R_index,0]+4
    start = strides[R_index,0]-2
    end = strides[R_index,1]+2
    gradients[i-int(sys.argv[2])*slices]= sim_loop_wrapper_grad_jit(R, R_index, R1, var_index, P_obs, N, B, M, start, end,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges)
    print(loss, gradients[i-int(sys.argv[2])*slices])
    check_grads(sim_loop_wrapper_R, (R,), order=1, atol=1e-2, rtol=1e-2, modes='fwd')

    
    #if abs(gradients[i-int(sys.argv[2])*slices]) > 1e3:
    #    gradients[i-int(sys.argv[2])*slices] = np.sign(gradients[i-int(sys.argv[2])*slices])*1e3


    if i >= int(sys.argv[2])*slices + 1000:
        gradients_averaged[i-int(sys.argv[2])*slices] = gradients[i-int(sys.argv[2])*slices-999:i-int(sys.argv[2])*slices+1].mean()
        gradients_averaged[i-int(sys.argv[2])*slices] = gradients[i-int(sys.argv[2])*slices-999:i-int(sys.argv[2])*slices+1].mean()
    else:
        gradients_averaged[i-int(sys.argv[2])*slices] = gradients[:i-int(sys.argv[2])*slices+1].mean()


       
    file = open(results_file, "a")  
    file.write(str(R) + " " + str(loss) + " " + str(gradients[i-int(sys.argv[2])*slices]) + " " + str(gradients_averaged[i-int(sys.argv[2])*slices]) + "\n")
    file.close()