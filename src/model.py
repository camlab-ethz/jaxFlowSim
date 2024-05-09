import jax.numpy as jnp
from jax import lax, jit, block_until_ready
import numpy as np
from src.initialise import loadConfig, buildBlood, buildArterialNetwork, makeResultsFolder
from src.IOutils import saveTempData
from src.solver import computeDt, solveModel
from src.check_conv import printConvError, computeConvError, checkConv
from functools import partial 
import numpy as np
import numpyro
import time


numpyro.set_platform("cpu")
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32' # Use 32 CPU devices
#jax.devices("cpu")[0]
#print(jax.local_device_count())



def configSimulation(input_filename, verbose=False, make_results_folder=True):
    data = loadConfig(input_filename)
    blood = buildBlood(data["blood"])

    J =  data["solver"]["num_snapshots"]

    (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
     N, B, edges, 
     input_data, strides, vessel_names, 
    indices_1, 
    indices_2) = buildArterialNetwork(data["network"], blood)#, junction_functions) = buildArterialNetwork(data["network"], blood)
    if make_results_folder:
        makeResultsFolder(data, input_filename)

    cardiac_T = sim_dat_const_aux[0,0]
    Ccfl = float(data["solver"]["Ccfl"])
    
    if verbose:
        print("start simulation")

    timepoints = np.linspace(0, cardiac_T, J)
    
    return (N, B, J, 
            sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
            timepoints, 1, Ccfl, edges, input_data, 
            blood.rho, strides, 
            indices_1, indices_2, 
            vessel_names, cardiac_T) #, junction_functions)


def simulationLoopUnsafe(N, B,
                        sim_dat, sim_dat_aux, sim_dat_const, 
                        sim_dat_const_aux,
                        Ccfl, edges, input_data, 
                        rho, strides,
                        indices1, 
                        indices2, upper = 100000):
    t = 0.0
    dt = 1
    P_t = jnp.zeros((upper,5*N))
    t_t = jnp.zeros(upper)


    def bodyFun(i, args):
        (sim_dat, sim_dat_aux, sim_dat_const, 
         sim_dat_const_aux,
         dt, t, t_t, 
         edges, 
         input_data, rho, 
         P_t) = args
        dt = computeDt(Ccfl, sim_dat[0,:],sim_dat[3,:], sim_dat_const[-1,:])
        sim_dat, sim_dat_aux = solveModel(N, B, strides[:,:2], 
                                          indices1, indices2,
                                          t, dt, sim_dat, 
                                          sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                          edges, input_data, rho)
        t = (t + dt)%sim_dat_const_aux[0,0]
        t_t = t_t.at[i].set(t)
        P_t = P_t.at[i,:].set(saveTempData(N, strides, sim_dat[4,:]))


        return (sim_dat, sim_dat_aux, sim_dat_const, 
                sim_dat_const_aux,
                dt, t, t_t,
                edges, 
                input_data, rho, 
                P_t)

    (sim_dat, sim_dat_aux, sim_dat_const, 
     sim_dat_const_aux,
     dt, t, t_t, 
     edges, 
     input_data, rho, 
     P_t) = lax.fori_loop(0, upper, bodyFun, (sim_dat, sim_dat_aux, sim_dat_const, 
                                                sim_dat_const_aux,
                                                dt, t, t_t,
                                                edges, 
                                                input_data, rho,
                                                P_t))
    
    return sim_dat, t_t, P_t
    


def simulationLoop(N, B, num_snapshots, 
                        sim_dat, sim_dat_aux, sim_dat_const, 
                        sim_dat_const_aux, timepoints, conv_tol, 
                        Ccfl, edges, input_data, 
                        rho, strides, 
                        indices1, 
                        indices2): #, junction_functions):
    t = 0.0
    passed_cycles = 0
    counter = 0
    P_t = jnp.empty((num_snapshots, N*5))
    t_t = jnp.empty(num_snapshots)
    P_l = jnp.empty((num_snapshots, N*5))
    dt = 0 

    def condFun(args):
        (_, _, _, 
         sim_dat_const_aux, t_i, _, 
         _, passed_cycles_i, _, 
         P_t_i, P_l_i, _, 
         conv_tol, _, _, 
         _, _,) = args
        err = computeConvError(N, P_t_i, P_l_i)
        def printConvErrorWrapper():
            printConvError(err)
            return False
        ret = lax.cond((passed_cycles_i + 1 > 1)*(checkConv(err, conv_tol))*
                           ((t_i - sim_dat_const_aux[0,0] * passed_cycles_i >= sim_dat_const_aux[0,0])), 
                            printConvErrorWrapper,
                            lambda: True)
        return ret

    def bodyFun(args):
        (sim_dat, sim_dat_aux, sim_dat_const, 
         sim_dat_const_aux, t, counter, 
         timepoints, passed_cycles, dt, 
         P_t, P_l, t_t, 
         _, Ccfl, edges, 
         input_data, rho) = args
        dt = computeDt(Ccfl, sim_dat[0,:],sim_dat[3,:], sim_dat_const[-1,:])
        sim_dat, sim_dat_aux = solveModel(N, B, strides[:,:2], 
                                          indices1, indices2,
                                          t, dt, sim_dat, 
                                          sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                          edges, input_data, rho) #, junction_functions)

        (P_t_temp,counter_temp) = lax.cond(t >= timepoints[counter], 
                                         lambda: (saveTempData(N, strides,
                                                                sim_dat[4,:]),counter+1), 
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

        return (sim_dat, sim_dat_aux, sim_dat_const, 
                sim_dat_const_aux, t, counter, 
                timepoints, passed_cycles, dt, 
                P_t, P_l, t_t, 
                conv_tol, Ccfl, edges, 
                input_data, rho)

    (sim_dat, sim_dat_aux, sim_dat_const, 
     sim_dat_const_aux, t, counter, 
     timepoints, passed_cycles, dt, 
     P_t, P_l, t_t,  
     conv_tol, Ccfl, edges, 
     input_data, rho) = lax.while_loop(condFun, bodyFun, (sim_dat, sim_dat_aux, sim_dat_const, 
                                                sim_dat_const_aux, t, counter, 
                                                timepoints, passed_cycles, dt, 
                                                P_t, P_l, t_t, 
                                                conv_tol, Ccfl, edges, 
                                                input_data, rho))
    
    return sim_dat, t_t, P_t

def runSimulationUnsafe(config_filename, verbose=False, make_results_folder=True):
    
    (N, B, J, 
     sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
     timepoints, conv_toll, Ccfl, edges, input_data, 
                rho, strides, 
                indices_1, indices_2,
                vessel_names, cardiac_T) = configSimulation(config_filename, verbose)

    if verbose:
        starting_time = time.time_ns()

    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 13))(simulationLoopUnsafe)
    sim_dat, t, P = block_until_ready(sim_loop_old_jit(N, B,
                                          sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                          Ccfl, edges, input_data, 
                                          rho, strides, 
                                          indices_1, indices_2, upper=120000))

    if verbose:
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"elapsed time = {ending_time} seconds")
    return sim_dat, t, P
    
def runSimulation(config_filename, verbose=False, make_results_folder=True):
    (N, B, J, 
     sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
     timepoints, conv_tol, Ccfl, edges, input_data, 
                rho, strides, 
                indices_1, indices_2,
                vessel_names, cardiac_T) = configSimulation(config_filename, verbose, make_results_folder)

    if verbose:
        starting_time = time.time_ns()
    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulationLoop)
    sim_dat, t, P  = block_until_ready(sim_loop_old_jit(N, B, J, 
                                          sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                          timepoints, conv_tol, Ccfl, edges, input_data, 
                                          rho, strides, 
                                          indices_1, indices_2))

    if verbose:
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"elapsed time = {ending_time} seconds")
    
    return sim_dat, t, P
