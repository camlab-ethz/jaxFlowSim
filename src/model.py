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
import numpy as np
import numpyro


numpyro.set_platform("cpu")
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32' # Use 32 CPU devices
#jax.devices("cpu")[0]
#print(jax.local_device_count())



def runSimulation(input_filename, verbose=False):
    data = loadConfig(input_filename)
    blood = buildBlood(data["blood"])

    J =  data["solver"]["jump"]

    (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
     N, B, edges, 
     input_data, nodes, vessel_names, 
    starts, ends, indices_1, 
    indices_2) = buildArterialNetwork(data["network"], blood)
    makeResultsFolder(data, input_filename)

    cardiac_T = sim_dat_const_aux[0,0]
    total_time = data["solver"]["cycles"]*cardiac_T
    Ccfl = float(data["solver"]["Ccfl"])
    
    if verbose:
        print("Start simulation")

    timepoints = np.linspace(0, cardiac_T, J)
    if verbose:
        print("Solving cardiac cycle no: 1")
        starting_time = time.time_ns()
    
    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulation_loop_old)
    sim_dat, t, P  = block_until_ready(sim_loop_old_jit(N, B, J, 
                                          sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                          timepoints, 1, Ccfl, edges, input_data, 
                                          blood.rho, total_time, nodes, 
                                          starts, ends,
                                          indices_1, indices_2))
    
    if verbose:
        print("\n")
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"Elapsed time = {ending_time} seconds")

    #jnp.set_printoptions(threshold=sys.maxsize)
    filename = input_filename.split("/")[-1]
    network_name = filename.split(".")[0]

    vessel_names_0007 = ["ascending aorta", "right subclavian artery", "right common carotid artery", 
                    "arch of aorta I", "brachiocephalic artery", 
                    "arch of aorta II",
                    "left common carotid artery", 
                    "left subclavian artery",
                    "descending aorta", 
                    ]
    vessel_names_0029 = [
                    "aorta I",
                    "left common iliac artery I",
                    "left internal iliac artery",
                    "left common iliac artery II",
                    "right common iliac artery I",
                    "celiac trunk II",
                    "celiac branch",
                    "aorta IV",
                    "left renal artery",
                    "aorta III",
                    "superior mesentric artery",
                    "celiac trunk I",
                    "aorta II",
                    "aorta V",
                    "right renal artery",
                    "right common iliac artery II",
                    "right internal iliac artery",
                    ]
    vessel_names_0053 = [
                    "right vertebral artery I", 
                    "left vertebral artery II",
                    "left posterior meningeal branch of vertebral artery",
                    "basilar artery III",
                    "left anterior inferior cerebellar artery",
                    "basilar artery II",
                    "right anterior inferior cerebellar artery",
                    "basilar artery IV",
                    "right superior cerebellar artery", 
                    "basilar artery I",
                    "left vertebral artery I",
                    "right posterior cerebellar artery I",
                    "left superior cerebellar artery",
                    "left posterior cerebellar artery I",
                    "right posterior central artery",
                    "right vertebral artery II",
                    "right posterior meningeal branch of vertebral artery",
                    "right posterior cerebellar artery II",
                    "right posterior comunicating artery",
                    ]
 
    plt.rcParams.update({'font.size': 20})

    for i,vessel_name in enumerate(vessel_names):
        index_vessel_name = vessel_names.index(vessel_name)
        P0 = np.loadtxt("/home/diego/studies/uni/thesis_maths/openBF/test/" + network_name + "/" + network_name + "_results/" + vessel_name + "_P.last")
        node = 2
        index_jl  = 1 + node
        index_jax  = 5*index_vessel_name + node
        P0 = P0[:,index_jl]
        res = np.sqrt(((P[:,index_jax]-P0).dot(P[:,index_jax]-P0)/P0.dot(P0)))
        _, ax = plt.subplots()
        ax.set_xlabel("t[s]")
        ax.set_ylabel("P[mmHg]")
        plt.title("network: " + network_name + ", # vessels: " + str(N) + ", vessel name: " + vessel_names[i] + ", \n relative error = |P_JAX-P_jl|/|P_jl| = " + str(res) + "%")
        #plt.title("network: " + network_name + ", vessel name: " + vessel_names_0053[i])
        #plt.title(vessel_names_0053[i])
        #plt.title("vessel name: " + vessel_name)
        plt.plot(t%cardiac_T,P[:,index_jax]/133.322)
        plt.plot(t%cardiac_T,P0/133.322)
        plt.legend(["P_JAX", "P_jl"], loc="lower right")
        plt.tight_layout()
        plt.savefig("results/" + network_name + "_results/" + network_name + "_" + vessel_names[i].replace(" ", "_") + "_P.pdf")
        plt.close()



def simulation_loop_old(N, B, jump, 
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
    
