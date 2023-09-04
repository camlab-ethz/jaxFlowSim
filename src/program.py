import jax
import jax.numpy as jnp
import numpy as np
from src.initialise import loadSimulationFiles, buildBlood, buildArterialNetwork, buildConst, makeResultsFolder
from src.IOutils import saveTempDatas, transferTempToLasts, writeResults
from src.boundary_conditions import updateGhostCells
from src.solver import calculateDeltaT, solveModel
from src.check_convergence import printConvError, computeConvError, checkConvergence
import time
import src.initialise as ini



def runSimulation_opt(input_filename, verbose=False):
    data = loadSimulationFiles(input_filename)
    buildBlood(data["blood"])

    if verbose:
        print(f"Build {input_filename} arterial network \n")

    ini.JUMP =  data["solver"]["jump"]

    vessels, sim_dat, sim_dat_aux = buildArterialNetwork(data["network"])
    makeResultsFolder(data, input_filename)

    buildConst(float(data["solver"]["Ccfl"]), data["solver"]["cycles"], data["solver"]["convergence tolerance"])
    
    if verbose:
        print("Start simulation \n")

    if verbose:
        print("Solving cardiac cycle no: 1")
        starting_time = time.time_ns()

    simulation_loop(sim_dat, sim_dat_aux)

    if verbose:
        print("\n")
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"Elapsed time = {ending_time} seconds")

    #writeResults(vessels)

def simulation_loop(sim_dat, sim_dat_aux):
    current_time = 0.0
    passed_cycles = 0
    counter = 0
    timepoints = np.linspace(0, ini.HEART.cardiac_T, ini.JUMP)
    P_t = jnp.zeros((ini.JUMP, ini.NUM_VESSELS*5), dtype=jnp.float64)
    P_l = jnp.zeros((ini.JUMP, ini.NUM_VESSELS*5), dtype=jnp.float64)
    dt = 0 #calculateDeltaT(sim_dat[0,:], sim_dat[3,:])

    @jax.jit
    def cond_fun(args):
        _, _, current_time, _, _, passed_cycles, dt, P_t, P_l = args
        err = computeConvError(P_t, P_l)
        def printConvErrorWrapper():
            printConvError(err)
            return False
        ret = jax.lax.cond((passed_cycles + 1 > 1)*(checkConvergence(err))*
                           ((current_time - ini.HEART.cardiac_T * passed_cycles >= ini.HEART.cardiac_T)*
                            (current_time - ini.HEART.cardiac_T * passed_cycles + dt > ini.HEART.cardiac_T)), 
                            printConvErrorWrapper,
                            lambda: True)
        return ret

    @jax.jit
    def body_fun(args):
        sim_dat, sim_dat_aux, current_time, counter, timepoints, passed_cycles, dt, P_t, P_l = args
        dt = calculateDeltaT(sim_dat[0,:],sim_dat[3,:])
        sim_dat = solveModel(sim_dat, sim_dat_aux, dt, current_time)
        sim_dat_aux = sim_dat_aux.at[2:,:].set(updateGhostCells(sim_dat))


        (P_t_temp,counter_temp) = jax.lax.cond(current_time >= timepoints[counter], 
                                         lambda: (saveTempDatas(sim_dat[4,:]),counter+1), 
                                         lambda: (P_t[counter,:],counter))
        P_t = P_t.at[counter,:].set(P_t_temp)
        counter = counter_temp

        def checkConv():
            err = computeConvError(P_t, P_l)
            printConvError(err)

        jax.lax.cond(((current_time - ini.HEART.cardiac_T * passed_cycles >= ini.HEART.cardiac_T)*
                       (current_time - ini.HEART.cardiac_T * passed_cycles + dt > ini.HEART.cardiac_T)*
                       (passed_cycles + 1 > 1)), 
                       checkConv,
                        lambda: None)
        (P_l,counter,timepoints,passed_cycles) = jax.lax.cond(((current_time - ini.HEART.cardiac_T * passed_cycles >= ini.HEART.cardiac_T)*
                                            (current_time - ini.HEART.cardiac_T * passed_cycles + dt > ini.HEART.cardiac_T)), 
                                         lambda: (P_t,0,timepoints + ini.HEART.cardiac_T, passed_cycles+1), 
                                         lambda: (P_l,counter,timepoints, passed_cycles))
        


        current_time += dt
        (passed_cycles) = jax.lax.cond(current_time >= ini.TOTAL_TIME,
                                         lambda: (passed_cycles+1), 
                                         lambda: (passed_cycles))

        return sim_dat, sim_dat_aux, current_time, counter, timepoints, passed_cycles, dt, P_t, P_l


    jax.lax.while_loop(cond_fun, body_fun, (sim_dat, sim_dat_aux, current_time, counter, timepoints, passed_cycles, dt, P_t, P_l))
    
    return current_time
    
