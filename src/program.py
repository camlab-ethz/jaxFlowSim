import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
from src.initialise import loadSimulationFiles, buildBlood, buildArterialNetwork, buildConst, makeResultsFolder
from src.IOutils import saveTempDatas#, writeResults
from src.boundary_conditions import updateGhostCells
from src.solver import calculateDeltaT, solveModel
from src.check_convergence import printConvError, computeConvError, checkConvergence
import time
import src.initialise as ini
import sys
import matplotlib.pyplot as plt



def runSimulation_opt(input_filename, verbose=False):
    data = loadSimulationFiles(input_filename)
    buildBlood(data["blood"])

    if verbose:
        print(f"Build {input_filename} arterial network \n")

    ini.JUMP =  data["solver"]["jump"]

    sim_dat, sim_dat_aux = buildArterialNetwork(data["network"])
    makeResultsFolder(data, input_filename)

    buildConst(float(data["solver"]["Ccfl"]), 
               data["solver"]["cycles"], 
               data["solver"]["convergence tolerance"],
               ini.SIM_DAT_CONST_AUX[0,1])
    
    if verbose:
        print("Start simulation")

    if verbose:
        #print("Solving cardiac cycle no: 1")
        starting_time = time.time_ns()

    timepoints = np.linspace(0, ini.SIM_DAT_CONST_AUX[0,0], ini.JUMP)
    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    sim_dat, t, P  = jax.block_until_ready(simulation_loop(ini.NUM_VESSELS, ini.PADDING, ini.JUMP, 
                                          sim_dat, sim_dat_aux, ini.SIM_DAT_CONST, ini.SIM_DAT_CONST_AUX, 
                                          timepoints, 1, ini.CCFL, ini.EDGES, ini.INPUT_DATA, 
                                          ini.BLOOD.rho, ini.TOTAL_TIME, ini.NODES))

    if verbose:
        #print("\n")
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"Elapsed time = {ending_time} seconds")

    jnp.set_printoptions(threshold=sys.maxsize)
    #print(sim_dat[4,0:100])
    #plt.figure()
    #plt.plot(t,P[:,0])
    #plt.show()
    filename = input_filename.split("/")[-1]
    network_name = filename.split(".")[0]
    #vessel_name = "ulnar_R_I"

    vessel_names = ini.VESSEL_NAMES

    for vessel_name in vessel_names:
        index_vessel_name = ini.VESSEL_NAMES.index(vessel_name)
        P0 = np.loadtxt("/home/diego/studies/uni/thesis_maths/openBF/test/" + network_name + "/" + network_name + "_results/" + vessel_name + "_P.last")
        #P0 = np.loadtxt("/home/diego/studies/uni/thesis_maths/openBF/test/adan56/adan56_results/adan56_results/aortic_arch_I_P.last")
        node = 2
        index_jl  = 1 + node
        index_jax  = 5*index_vessel_name + node
        P0 = P0[:,index_jl]
        res = np.sqrt(((P[:,index_jax]-P0).dot(P[:,index_jax]-P0)/P0.dot(P0)))
        #print(res)
        fig, ax = plt.subplots()
        ax.set_xlabel("t")
        ax.set_ylabel("P")
        plt.title("network: " + network_name + ", # vessels: " + str(ini.NUM_VESSELS) + ", vessel name: " + vessel_name + ", \n relative error = |P_JAX-P_jl|/|P_jl| = " + str(res) + "%")
        plt.plot(t%ini.SIM_DAT_CONST_AUX[0,0],P[:,index_jax])
        plt.plot(t%ini.SIM_DAT_CONST_AUX[0,0],P0)
        plt.legend(["P_JAX", "P_jl"], loc="lower right")
        #print(network_name + "_" + vessel_name + "_P.pdf")
        plt.savefig(network_name + "_" + vessel_name + "_P.pdf")
        plt.close()

    #plt.show()

    #writeResults(vessels)

#@jax.jit
@partial(jax.jit, static_argnums=(0, 1, 2))
def simulation_loop(N, B, jump, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, timepoints, conv_toll, Ccfl, edges, input_data, rho, total_time, nodes):
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
        ret = jax.lax.cond((passed_cycles_i + 1 > 1)*(checkConvergence(err, conv_toll))*
                           ((t_i - sim_dat_const_aux[0,0] * passed_cycles_i >= sim_dat_const_aux[0,0])), 
                            printConvErrorWrapper,
                            lambda: True)
        return ret

    def body_fun(args):
        sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, t, counter, timepoints, passed_cycles, dt, P_t, P_l, t_t, _, Ccfl, edges, input_data, rho, total_time, nodes = args
        dt = calculateDeltaT(Ccfl, sim_dat[0,:],sim_dat[3,:], sim_dat_const[-1,:])
        sim_dat, sim_dat_aux = solveModel(N, B, 
                                          t, dt, sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          edges, input_data, rho)
        #sim_dat_aux = sim_dat_aux.at[:,2:10].set(updateGhostCells(M, N, sim_dat))
        #sim_dat_aux[:,2:10] = updateGhostCells(M, N, sim_dat)


        (P_t_temp,counter_temp) = jax.lax.cond(t >= timepoints[counter], 
                                         lambda: (saveTempDatas(N, ini.STARTS, ini.ENDS, nodes, sim_dat[4,:]),counter+1), 
                                         lambda: (P_t[counter,:],counter))
        P_t = P_t.at[counter,:].set(P_t_temp)
        t_t = t_t.at[counter].set(t)
        counter = counter_temp

        def checkConv():
            err = computeConvError(N, P_t, P_l)
            printConvError(err)

        jax.lax.cond(((t - sim_dat_const_aux[0,0] * passed_cycles >= sim_dat_const_aux[0,0])*
                       (passed_cycles + 1 > 1)), 
                       checkConv,
                        lambda: None)
        (P_l,counter,timepoints,passed_cycles) = jax.lax.cond((t - sim_dat_const_aux[0,0] * passed_cycles >= sim_dat_const_aux[0,0]),
                                         lambda: (P_t,0,timepoints + sim_dat_const_aux[0,0], passed_cycles+1), 
                                         lambda: (P_l,counter,timepoints, passed_cycles))
        


        t += dt
        (passed_cycles) = jax.lax.cond(t >= total_time,
                                         lambda: (passed_cycles+1), 
                                         lambda: (passed_cycles))

        return (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                t, counter, timepoints, passed_cycles, dt, P_t, P_l, t_t, 
                conv_toll, Ccfl, edges, input_data, rho, total_time, nodes)


    (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
     t, counter, timepoints, passed_cycles, dt, P_t, P_l, t_t,  
     conv_toll, Ccfl, edges, input_data, rho, total_time, nodes) = jax.lax.while_loop(cond_fun, body_fun, (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, t, 
                                                                                                           counter, timepoints, passed_cycles, dt, P_t, P_l, t_t, conv_toll, 
                                                                                                           Ccfl, edges, input_data, rho, total_time, nodes))
    
    return sim_dat, t_t, P_t
    
