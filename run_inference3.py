from src.model import configSimulation, simulationLoopUnsafe
from numpyro.infer.reparam import TransformReparam
import os
from jax.config import config
import sys
import time
from functools import partial
from jax import block_until_ready, jit
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import optax
import numpyro
from numpyro.infer import MCMC,HMC

os.chdir(os.path.dirname(__file__))
config.update("jax_enable_x64", True)

numpyro.set_host_device_count(8)

config_filename = ""
if len(sys.argv) == 1:
    # base cases
    #config_filename = "test/single-artery/single-artery.yml"
    #config_filename = "test/tapering/tapering.yml"
    #config_filename = "test/conjunction/conjunction.yml"
    config_filename = "test/bifurcation/bifurcation.yml"
    #config_filename = "test/aspirator/aspirator.yml"

    # openBF-hub 
    #config_filename = "test/adan56/adan56.yml"

    # vascularmodels.com
    #modelname = "0007_H_AO_H"
    #modelname = "0029_H_ABAO_H"
    #modelname = "0053_H_CERE_H"
    #config_filename = "test/" + modelname + "/" + modelname + ".yml"
else:
    config_filename = "test/" + sys.argv[1] + "/" + sys.argv[1] + ".yml"



verbose = True
(N, B, J, 
 sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
 timepoints, conv_toll, Ccfl, edges, input_data, 
 rho, nodes, 
 starts, ends,
 indices1, indices2,
 vessel_names, cardiac_T) = configSimulation(config_filename, verbose)

sim_loop_jit = partial(jit, static_argnums=(0, 1, 15))(simulationLoopUnsafe)
sim_dat_new, t, P_obs  = block_until_ready(sim_loop_jit(N, B,
                                      sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                      Ccfl, edges, input_data, 
                                      rho, nodes, 
                                      starts, ends,
                                      indices1, indices2, 120000))
R1_vessel_index = 1
R1_const_index = 7
R1_real = sim_dat_const[R1_const_index,ends[R1_vessel_index]]
R1_init = 1e8


def simLoopWrapper(R, R_scale):
    ones = jnp.ones(ends[R1_vessel_index]-starts[R1_vessel_index]+4)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[R1_const_index,starts[R1_vessel_index]-2:ends[R1_vessel_index]+2].set(R*R_scale*ones)
    _, _, P = simulationLoopUnsafe(N, B,
                    sim_dat, sim_dat_aux, sim_dat_const_new, sim_dat_const_aux, 
                    Ccfl, edges, input_data, 
                    rho, nodes, 
                    starts, ends,
                    indices1, indices2, 120000)
    return P

sim_loop_wrapper_jit = jit(simLoopWrapper)

def logp(y, R, R_scale, sigma):
    y_hat = sim_loop_wrapper_jit(R, R_scale)
    L = jnp.mean(jnp.log(jax.scipy.stats.norm.pdf(((y - y_hat)).flatten(), loc = 0, scale=sigma)))

    jax.debug.print("L = {x}", x=L)
    return L
def model(P_obs, R_scale):
    std = numpyro.sample("std", dist.HalfNormal())
    loc = numpyro.sample("loc", dist.Normal())
    R_dist=numpyro.sample("R", dist.Normal(loc,std))
    jax.debug.print("R_dist = {x}", x=R_dist)
    #sigma = numpyro.sample("sigma", dist.Normal())
    sigma = numpyro.sample("sigma", dist.HalfNormal())
    log_density = logp(P_obs, R_dist, R_scale, sigma=sigma) 
    numpyro.factor("custom_logp", log_density)

mcmc = MCMC(numpyro.infer.NUTS(model, forward_mode_differentiation=True),num_samples=100,num_warmup=10,num_chains=1)
mcmc.run(jax.random.PRNGKey(3450), P_obs, R_scale)
mcmc.print_summary()
R = jnp.mean(mcmc.get_samples()["R"])
print(R1)
print(R) 
#### adam optimizer example
#start_learning_rate = 1e-2
## Exponential decay of the learning rate.
#scheduler = optax.exponential_decay(
#init_value=start_learning_rate, 
#transition_steps=1000,
#decay_rate=0.99)

## Combining gradient transforms using `optax.chain`.
#gradient_transform = optax.chain(
#    optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
#    optax.scale_by_adam(),  # Use the updates from adam.
#    optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
#    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
#    optax.scale(-1.0)
#)
#num_weights = 1
##optimizer = optax.adam(learning_rate)
#params = jnp.array([1, 0.0])  # Recall target_params=0.5.
#opt_state = gradient_transform.init(params)
##params = {'w': R_scale*jnp.ones((num_weights,))}
##opt_state = optimizer.init(params)
#compute_loss = lambda params, y: jnp.mean(optax.l2_loss(sim_loop_wrapper_jit(params[0]), y))/jnp.mean(y)
##compute_loss = lambda params, y: jnp.linalg.norm(sim_loop_wrapper_jit(params[0]), y)/jnp.linalg.norm(y)
#print("loss", compute_loss(jnp.array([R1, 0.0]), sim_dat_new[2,:].flatten()))

#for i in range(100):
#    grads = jax.jacfwd(compute_loss)(params, sim_dat_new[2,:].flatten())
#    print(grads)

#    updates, opt_state = gradient_transform.update(grads, opt_state)
#    print(opt_state)
#    params = optax.apply_updates(params, updates)
#    print(params)



if verbose:
    print("\n")
    ending_time = (time.time_ns() - starting_time) / 1.0e9
    print(f"Elapsed time = {ending_time} seconds")

    