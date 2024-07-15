from src.model import configSimulation, simulationLoopUnsafe
from numpyro.infer.reparam import TransformReparam
import os
import sys
import time
from functools import partial
from jax import jit, grad, jacfwd
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import numpy as np
import itertools
import optax
from flax.training.train_state import TrainState

os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)


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

if verbose:
    starting_time = time.time_ns()

sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 12))(simulationLoopUnsafe)
sim_dat_obs, t_obs, P_obs = sim_loop_old_jit(N, B,
                                      sim_dat, sim_dat_aux, 
                                      sim_dat_const, sim_dat_const_aux, 
                                      Ccfl, input_data, rho, 
                                      masks, strides, edges,
                                      upper=120000)

Ccfl = 0.5 

R_index = 1
var_index = 7
R1 = sim_dat_const[var_index,strides[R_index,1]]
#R_scales = np.linspace(1.1*R1, 2*R1, 16)
R_scales = np.linspace(0.1*R1, 10*R1, 8)
def simLoopWrapper(params):
    R = params[0]
    ones = jnp.ones(strides[R_index,1]-strides[R_index,0]+4)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[var_index,strides[R_index,0]-2:strides[R_index,1]+2].set(R*ones)
    _, _, P = sim_loop_old_jit(N, B,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const_new, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges,
                                          upper=120000)
    #jax.debug.print("P={x}", x=P)
    #jax.debug.print("R={x}", x=R)
    #jax.debug.print("P_obs={x}", x=P_obs)
    return P

results_folder = "results/inference_ensemble_det"
if not os.path.isdir(results_folder):
    os.makedirs(results_folder, mode = 0o777)


network_properties = {
        #"tx": [optax.adam, optax.sgd, optax.lars],
        #"tx": [optax.adabelief, optax.adadelta, optax.adafactor, optax.adagrad, 
        #       optax.adamw, optax.adamax, optax.adamaxw, optax.amsgrad],
        #"tx": [optax.adagrad, 
        #       optax.adamw, optax.adamax, optax.adamaxw, optax.amsgrad],
        "tx": [optax.adamw, optax.adamax, optax.adamaxw, optax.amsgrad],
        "learning_rate": [1e7],
        "learning_rate": [1e7],
        "epochs": [100,1000,2000]
        }

settings = list(itertools.product(*network_properties.values()))

results_folder = "results/inference_ensemble_sgd"
if not os.path.isdir(results_folder):
    os.makedirs(results_folder, mode = 0o777)

for set_num, setup in enumerate(settings):
    print("###################################", set_num, "###################################")
    results_file = results_folder  + "/setup_" + str(setup[0].__name__) + "_" + str(setup[1]) +  "_" + str(setup[2]) +".txt"

    model = simLoopWrapper
    variables = [R_scales[int(sys.argv[2])]]
    tx = setup[0]
    y = P_obs
    x = simLoopWrapper

    state = TrainState.create(
        apply_fn=model,
        params=variables,
        tx=tx(setup[1]))

    def loss_fn(params, x, y):
      predictions = state.apply_fn(params)
      loss = optax.l2_loss(predictions=predictions, targets=y).mean()
      return loss


    for _ in range(setup[2]):
        grads = jax.jacfwd(loss_fn)(state.params, x, y)
        print(grads)
        state = state.apply_gradients(grads=grads)
        print(state)
        print(loss_fn(state.params, x, y))
    file = open(results_file, "a")  
    file.write(str(R_scales[int(sys.argv[2])]) + " " + str(state.params[0]) + "  " + str(R1) + "\n")
    file.close()

#for (j, learning_rate) in enumerate(learning_rates):
#    R_star = R_scales[int(sys.argv[2])]
#    for i in range(1000):
#        print(j,i)
#        gradient = sim_loop_wrapper_jit(R_star)
#        R_star -= learning_rate*gradient
#
#    results_file = results_folder  + "/setup_" + str(learning_rate) + ".txt"
#    file = open(results_file, "a")  
#    file.write(str(R_scales[int(sys.argv[2])]) + " " + str(R_star) + "  " + str(R1) + "\n")
#    file.close()

    


network_properties = {
        "sigma": [1e-5],
        "scale": [10],
        "num_warmup": np.arange(10, 110, 10),
        "num_samples": np.arange(100, 1100, 100),
        "num_chains": [1]
        }
