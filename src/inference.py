from functools import partial
import jax.numpy as jnp
import jax
from jax import block_until_ready, jit, lax, grad, jacfwd
import numpy as np
from src.initialise import loadConfig, buildBlood, buildArterialNetwork, makeResultsFolder
from src.IOutils import saveTempDatas#, writeResults
from src.solver import computeDt, solveModel
from src.check_convergence import printConvError, computeConvError, checkConvergence
import time
import os
import sys
import matplotlib.pyplot as plt
import numpyro
import numpy as np
import numpyro.distributions as dist
import optax

from numpyro.infer import MCMC,HMC


numpyro.set_platform("cpu")
numpyro.enable_x64()
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
#numpyro.set_host_device_count(9)
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32' # Use 8 CPU devices
os.chdir(os.path.dirname(__file__))

#jax.devices("cpu")[0]
#numpyro.set_platform("cpu")
#numpyro.set_host_device_count(4)
#import os
#import random
#import sys
print(jax.local_device_count())
#numpyro.enable_validation()



def runSimulation_opt(config_filename, verbose=False):
    data = loadConfig(config_filename)
    blood = buildBlood(data["blood"])

    #if verbose:
    #    print(f"Build {input_filename} arterial network \n")

    J =  data["solver"]["jump"]

    (sim_dat, sim_dat_aux, sim_dat_const, 
    sim_dat_const_aux, N, B, 
    edges, input_data, nodes, 
    vessel_names, starts, ends, 
    indices1, indices2) = buildArterialNetwork(data["network"], blood)
    makeResultsFolder(data, config_filename)

    cardiac_T = sim_dat_const_aux[0,0]
    total_time = data["solver"]["cycles"]*cardiac_T
    Ccfl = float(data["solver"]["Ccfl"])
    
    if verbose:
        print("Start simulation")

    timepoints = np.linspace(0, cardiac_T, J)
    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    if verbose:
        print("Solving cardiac cycle no: 1")
        starting_time = time.time_ns()
    
    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simLoop)
    sim_dat_new, t, P_obs  = block_until_ready(sim_loop_old_jit(N, B, J, 
                                          sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                          timepoints, 1, Ccfl, edges, input_data, 
                                          blood.rho, total_time, nodes, 
                                          starts, ends,
                                          indices1, indices2))
    
    R_index = 1
    var_index = 7
    R1 = sim_dat_const[var_index,ends[R_index]]
    R_scale = 1.1*R1
    print(R1, R_scale)
    def simLoopWrapper(R):#, R_scale):
        R = R*1e8
        #R = 0.5*R*R_scale + R_scale
        ones = jnp.ones(ends[R_index]-starts[R_index]+4)
        #jax.debug.print("{x}", x = sim_dat)
        sim_dat_const_new = jnp.array(sim_dat_const)
        sim_dat_const_new = sim_dat_const_new.at[var_index,starts[R_index]-2:ends[R_index]+2].set(R*ones)
        sim_dat_temp, _, P = sim_loop_old_jit(N, B, J, 
                        sim_dat, sim_dat_aux, sim_dat_const_new, sim_dat_const_aux, 
                        timepoints, 1, Ccfl, edges, input_data, 
                        blood.rho, total_time, nodes, 
                        starts, ends,
                        indices1, indices2)
        return sim_dat_temp[2,:].flatten()/jnp.linalg.norm(sim_dat_temp[2,:].flatten())
    sim_loop_wrapper_jit = jit(simLoopWrapper)

    def logp(y, R, sigma):
        #y_hat = ((R+1)*(R+1)-y)*((R+1)*(R+1)-y)
        y_hat = jnp.zeros_like(y)
        y_hat = jax.lax.cond(R>0, lambda: sim_loop_wrapper_jit(R), lambda: y_hat)
        #y_hat = sim_loop_wrapper_jit(R)

        L = jnp.sum(jnp.log(jax.scipy.stats.halfnorm.pdf((y-y_hat)*(y-y_hat), loc = 0, scale=sigma)))

        #L = jnp.linalg.norm(y - y_hat)/jnp.linalg.norm(y)
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
    def model():
        R_dist=numpyro.sample("R", dist.Normal(1,10))
        sigma = numpyro.sample("sigma", dist.HalfNormal())
        log_density = logp(sim_dat_new[0,:], R_dist, sigma)
        return numpyro.factor("custom_logp", log_density)
    mcmc = MCMC(numpyro.infer.NUTS(model, forward_mode_differentiation=True),num_samples=1000,num_warmup=100,num_chains=1)
    mcmc.run(jax.random.PRNGKey(3450))
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

def simLoop(N, B, jump, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, timepoints, conv_toll, Ccfl, edges, input_data, rho, total_time, nodes, starts, ends, indices1, indices2):
    jax.debug.print("R1 = {R}", R=sim_dat_const[7,starts[1]])
    #jax.debug.print("{x}", x=sim_dat)
    t = 0.0
    passed_cycles = 0
    counter = 0
    P_t = jnp.empty((jump, N*5))
    t_t = jnp.empty((jump))
    P_l = jnp.empty((jump, N*5))
    dt = 0 

    def cond_fun(args):
        _, _, _, sim_dat_const_aux, t_i, _, _, passed_cycles_i, _, P_t_i, P_l_i, _, conv_toll, _, _, _, _, _, _ = args
        err = computeConvError(N, P_t_i, P_l_i)
        def printConvErrorWrapper():
            printConvError(err)
            return False
        ret = lax.cond((passed_cycles_i + 1 > 1)*(checkConvergence(err, conv_toll))*
                           ((t_i - sim_dat_const_aux[0,0] * passed_cycles_i >= sim_dat_const_aux[0,0])), 
                            printConvErrorWrapper,
                            lambda: True)
        return ret

    def body_fun(args):
        sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, t, counter, timepoints, passed_cycles, dt, P_t, P_l, t_t, _, Ccfl, edges, input_data, rho, total_time, nodes = args
        dt = computeDt(Ccfl, sim_dat[0,:],sim_dat[3,:], sim_dat_const[-1,:])
        sim_dat, sim_dat_aux = solveModel(N, B, starts, ends,
                                          indices1, indices2,
                                          t, dt, sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          edges, input_data, rho)
        #sim_dat_aux = sim_dat_aux.at[:,2:10].set(updateGhostCells(M, N, sim_dat))
        #sim_dat_aux[:,2:10] = updateGhostCells(M, N, sim_dat)


        (P_t_temp,counter_temp) = lax.cond(t >= timepoints[counter], 
                                         lambda: (saveTempDatas(N, starts, ends, nodes, sim_dat[4,:]),counter+1), 
                                         lambda: (P_t[counter,:],counter))
        P_t = P_t.at[counter,:].set(P_t_temp)
        t_t = t_t.at[counter].set(t)
        counter = counter_temp

        def checkConv():
            err = computeConvError(N, P_t, P_l)
            printConvError(err)

        lax.cond(((t - sim_dat_const_aux[0,0] * passed_cycles >= sim_dat_const_aux[0,0])*
                       (passed_cycles + 1 > 1)), 
                       checkConv,
                        lambda: None)
        (P_l,counter,timepoints,passed_cycles) = lax.cond((t - sim_dat_const_aux[0,0] * passed_cycles >= sim_dat_const_aux[0,0]),
                                         lambda: (P_t,0,timepoints + sim_dat_const_aux[0,0], passed_cycles+1), 
                                         lambda: (P_l,counter,timepoints, passed_cycles))
        


        t += dt
        (passed_cycles) = lax.cond(t >= total_time,
                                         lambda: (passed_cycles+1), 
                                         lambda: (passed_cycles))

        return (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                t, counter, timepoints, passed_cycles, dt, P_t, P_l, t_t, 
                conv_toll, Ccfl, edges, input_data, rho, total_time, nodes)


    (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
     t, counter, timepoints, passed_cycles, dt, P_t, P_l, t_t,  
     conv_toll, Ccfl, edges, input_data, rho, total_time, nodes) = lax.while_loop(cond_fun, body_fun, (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, t, 
                                                                                                           counter, timepoints, passed_cycles, dt, P_t, P_l, t_t, conv_toll, 
                                                                                                           Ccfl, edges, input_data, rho, total_time, nodes))
    
    return sim_dat, t_t, P_t
    
