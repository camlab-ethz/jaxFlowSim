from src.model import configSimulation, simulationLoopUnsafe
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

if verbose:
    starting_time = time.time_ns()

sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 15))(simulationLoopUnsafe)
sim_dat_new, t, P_obs  = block_until_ready(sim_loop_old_jit(N, B,
                                      sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                      Ccfl, edges, input_data, 
                                      rho, nodes, 
                                      starts, ends,
                                      indices1, indices2, 120000))
R_index = 1
var_index = 7
R1 = sim_dat_const[var_index,ends[R_index]]
R_scale = 1e8
#R_scale = 1.1*R1
print(R1, R_scale)
def simLoopWrapper(R, R_scale):
    #R = R*1e8
    #R = 0.5*R*R_scale + R_scale
    ones = jnp.ones(ends[R_index]-starts[R_index]+4)
    #jax.debug.print("{x}", x = sim_dat)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[var_index,starts[R_index]-2:ends[R_index]+2].set(R*R_scale*ones)
    _, _, P = simulationLoopUnsafe(N, B,
                    sim_dat, sim_dat_aux, sim_dat_const_new, sim_dat_const_aux, 
                    Ccfl, edges, input_data, 
                    rho, nodes, 
                    starts, ends,
                    indices1, indices2, 120000)
    return P/jnp.linalg.norm(P)

sim_loop_wrapper_jit = jit(simLoopWrapper)

def logp(y, R, R_scale):#, sigma):
    #y_hat = jnp.zeros_like(y)
    #y_hat = jax.lax.cond(R>0, lambda: sim_loop_wrapper_jit(R), lambda: y_hat)
    y_hat = sim_loop_wrapper_jit(R, R_scale)
    #L = jnp.sum(jnp.log(jax.scipy.stats.norm.pdf(((y - y_hat)).flatten(), loc = 0, scale=sigma)))

    L = jnp.linalg.norm(y - y_hat)/jnp.linalg.norm(y)
    jax.debug.print("L = {x}", x=L)
    #jax.debug.print("{x}", x=jnp.linalg.norm(y))
    #jax.debug.print("L = {x}", x=L)
    #jax.debug.print("{x}", x=L)
    return L
#def model(obs, R_scale):
#    R_dist = numpyro.sample("R", dist.LogNormal())
#    #sigma = numpyro.sample("sigma", dist.Normal())
#    print("R_dist",R_dist)
#    #sigma = numpyro.sample("sigma", dist.Normal())
#    log_density = logp(obs, (R_dist+0.1)*R_scale)#, sigma)
#    return numpyro.factor("custom_logp", -log_density)
### NUTS model with bultin loss
#def model(obs, R_scale):
#    R_dist=numpyro.sample("R", dist.LogNormal(loc=0,scale=0.25))
#    sigma = numpyro.sample("sigma", dist.HalfNormal())
#    with numpyro.plate("size", jnp.size(obs)):
#        numpyro.sample("obs", dist.Normal(sim_loop_wrapper_jit(R_scale*(R_dist+0.1)),scale=sigma), obs=obs)
def model(P_obs_norm, R_scale):
    R_dist=numpyro.sample("R", dist.Normal())
    jax.debug.print("R_dist = {x}", x=R_dist)
    #sigma = numpyro.sample("sigma", dist.Normal())
    log_density = logp(P_obs_norm, R_dist, R_scale)#, sigma)
    numpyro.factor("custom_logp", log_density)

#mcmc = MCMC(numpyro.infer.HMC(model, forward_mode_differentiation=True, step_size=1e-3),num_samples=200,num_warmup=12,num_chains=1)
#mcmc.run(jax.random.PRNGKey(3450), P_obs/jnp.linalg.norm(P_obs), R_scale)
#mcmc.print_summary()
#R = jnp.mean(mcmc.get_samples()["R"])
#print(R1)
#print(R) 
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
#compute_loss = lambda params, y: jnp.linalg.norm(sim_loop_wrapper_jit(params[0]), y)/jnp.linalg.norm(y)
#print("loss", compute_loss(jnp.array([R1, 0.0]), sim_dat_new[2,:].flatten()))

#for i in range(100):
#    grads = jax.jacfwd(compute_loss)(params, sim_dat_new[2,:].flatten())
#    print(grads)
#
#    updates, opt_state = gradient_transform.update(grads, opt_state)
#    print(opt_state)
#    params = optax.apply_updates(params, updates)
#    print(params)

compute_loss = lambda R: jnp.mean(sim_loop_wrapper_jit(R, R_scale)-P_obs)/jnp.mean(P_obs)
grad_compute_loss = jax.grad(compute_loss)
R_new = R_scale
for i in range(100):
    grad = grad_compute_loss(R_new)
    print(grad)
    R_new += grad
    print(R_new)

    #updates, opt_state = gradient_transform.update(grads, opt_state)
    #print(opt_state)
    #params = optax.apply_updates(params, updates)
    #print(params)



if verbose:
    print("\n")
    ending_time = (time.time_ns() - starting_time) / 1.0e9
    print(f"Elapsed time = {ending_time} seconds")

    