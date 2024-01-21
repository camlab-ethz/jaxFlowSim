import jax.numpy as jnp
from jax import lax
import numpy as np
from src.initialise import loadConfig, buildBlood, buildArterialNetwork, makeResultsFolder
from src.IOutils import saveTempDatas#, writeResults
from src.solver import computeDt, solveModel
from src.check_convergence import printConvError, computeConvError, checkConvergence
import numpy as np
import numpyro


numpyro.set_platform("cpu")
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32' # Use 32 CPU devices
#jax.devices("cpu")[0]
#print(jax.local_device_count())



def configSimulation(input_filename, verbose=False, make_results_folder=True):
    data = loadConfig(input_filename)
    blood = buildBlood(data["blood"])

    J =  data["solver"]["jump"]

    (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
     N, B, edges, 
     input_data, nodes, vessel_names, 
    starts, ends, indices_1, 
    indices_2) = buildArterialNetwork(data["network"], blood)
    if make_results_folder:
        makeResultsFolder(data, input_filename)

    cardiac_T = sim_dat_const_aux[0,0]
    total_time = data["solver"]["cycles"]*cardiac_T
    Ccfl = float(data["solver"]["Ccfl"])
    
    if verbose:
        print("start simulation")

    timepoints = np.linspace(0, cardiac_T, J)
    
    return (N, B, J, 
            sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
            timepoints, 1, Ccfl, edges, input_data, 
            blood.rho, total_time, nodes, 
            starts, ends,
            indices_1, indices_2, 
            vessel_names, cardiac_T)



def simulationLoop(N, B, jump, 
                        sim_dat, sim_dat_aux, sim_dat_const, 
                        sim_dat_const_aux, timepoints, conv_toll, 
                        Ccfl, edges, input_data, 
                        rho, total_time, nodes, 
                        starts, ends, indices1, 
                        indices2):
    t = 0.0
    passed_cycles = 0
    counter = 0
    P_t = jnp.empty((jump, N*5))
    t_t = jnp.empty((jump))
    P_l = jnp.empty((jump, N*5))
    dt = 0 

    def condFun(args):
        (_, _, _, 
         sim_dat_const_aux, t_i, _, 
         _, passed_cycles_i, _, 
         P_t_i, P_l_i, _, 
         conv_toll, _, _, 
         _, _, _, 
         _) = args
        err = computeConvError(N, P_t_i, P_l_i)
        def printConvErrorWrapper():
            printConvError(err)
            return False
        ret = lax.cond((passed_cycles_i + 1 > 1)*(checkConvergence(err, conv_toll))*
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
         input_data, rho, total_time, 
         nodes) = args
        dt = computeDt(Ccfl, sim_dat[0,:],sim_dat[3,:], sim_dat_const[-1,:])
        sim_dat, sim_dat_aux = solveModel(N, B, starts, 
                                          ends, indices1, indices2,
                                          t, dt, sim_dat, 
                                          sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                          edges, input_data, rho)

        (P_t_temp,counter_temp) = lax.cond(t >= timepoints[counter], 
                                         lambda: (saveTempDatas(N, starts, ends, 
                                                                nodes, sim_dat[4,:]),counter+1), 
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

        return (sim_dat, sim_dat_aux, sim_dat_const, 
                sim_dat_const_aux, t, counter, 
                timepoints, passed_cycles, dt, 
                P_t, P_l, t_t, 
                conv_toll, Ccfl, edges, 
                input_data, rho, total_time, 
                nodes)

    (sim_dat, sim_dat_aux, sim_dat_const, 
     sim_dat_const_aux, t, counter, 
     timepoints, passed_cycles, dt, 
     P_t, P_l, t_t,  
     conv_toll, Ccfl, edges, 
     input_data, rho, total_time, 
     nodes) = lax.while_loop(condFun, bodyFun, (sim_dat, sim_dat_aux, sim_dat_const, 
                                                sim_dat_const_aux, t, counter, 
                                                timepoints, passed_cycles, dt, 
                                                P_t, P_l, t_t, 
                                                conv_toll, Ccfl, edges, 
                                                input_data, rho, total_time, 
                                                nodes))
    
    return sim_dat, t_t, P_t
    
