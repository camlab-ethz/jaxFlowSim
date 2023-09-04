import jax
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

    simulation_loop(vessels, sim_dat, sim_dat_aux)

    if verbose:
        print("\n")
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"Elapsed time = {ending_time} seconds")

    #writeResults(vessels)

def simulation_loop(vessels, sim_dat, sim_dat_aux):
    jax.checking_leaks()
    current_time = 0.0
    passed_cycles = 0
    counter = 0
    timepoints = np.linspace(0, ini.HEART.cardiac_T, ini.JUMP)
    dt = 0 #calculateDeltaT(sim_dat[0,:], sim_dat[3,:])

    @jax.jit
    def cond_fun(args):
        vessels, sim_dat, sim_dat_aux, current_time, counter, timepoints, passed_cycles, dt = args
        err = computeConvError(vessels)
        def fun1():
            printConvError(err)
            return False
        ret = jax.lax.cond((passed_cycles + 1 > 1)*(checkConvergence(err))*
                           ((current_time - ini.HEART.cardiac_T * passed_cycles >= ini.HEART.cardiac_T)*
                            (current_time - ini.HEART.cardiac_T * passed_cycles + dt > ini.HEART.cardiac_T)), 
                            fun1,
                            lambda: True)
        return ret

    @jax.jit
    def body_fun(args):
        vessels, sim_dat, sim_dat_aux, current_time, counter, timepoints, passed_cycles, dt = args
        dt = calculateDeltaT(sim_dat[0,:],sim_dat[3,:])
        #jax.debug.breakpoint()
        sim_dat = solveModel(sim_dat, sim_dat_aux, dt, current_time)
        for i in range(len(vessels)):
            vessels[i].u = sim_dat[0,ini.MESH_SIZES[i]:ini.MESH_SIZES[i+1]]
            vessels[i].Q = sim_dat[1,ini.MESH_SIZES[i]:ini.MESH_SIZES[i+1]]
            vessels[i].A = sim_dat[2,ini.MESH_SIZES[i]:ini.MESH_SIZES[i+1]]
            vessels[i].c = sim_dat[3,ini.MESH_SIZES[i]:ini.MESH_SIZES[i+1]]
            vessels[i].P = sim_dat[4,ini.MESH_SIZES[i]:ini.MESH_SIZES[i+1]]
        #jax.debug.breakpoint()
        sim_dat_aux = sim_dat_aux.at[2:,:].set(updateGhostCells(sim_dat))
        #jax.debug.breakpoint()
        #for i in range(len(vessels)):
            #vessels[i].W1M0 = sim_dat_aux[0,i]
            #vessels[i].W2M0 = sim_dat_aux[1,i]
            #vessels[i].U00Q = sim_dat_aux[2,i]
            #vessels[i].U00A = sim_dat_aux[3,i]
            #vessels[i].U01Q = sim_dat_aux[4,i]
            #vessels[i].U01A = sim_dat_aux[5,i]
            #vessels[i].UM1Q = sim_dat_aux[6,i]
            #vessels[i].UM1A = sim_dat_aux[7,i]
            #vessels[i].UM2Q = sim_dat_aux[8,i]
            #vessels[i].UM2A = sim_dat_aux[9,i]
            #sim_dat_aux = sim_dat_aux.at[0,i].set(vessels[i].W1M0)
            #sim_dat_aux = sim_dat_aux.at[1,i].set(vessels[i].W2M0)
            #sim_dat_aux = sim_dat_aux.at[2,i].set(vessels[i].U00Q)
            #sim_dat_aux = sim_dat_aux.at[3,i].set(vessels[i].U00A)
            #sim_dat_aux = sim_dat_aux.at[4,i].set(vessels[i].U01Q)
            #sim_dat_aux = sim_dat_aux.at[5,i].set(vessels[i].U01A)
            #sim_dat_aux = sim_dat_aux.at[6,i].set(vessels[i].UM1Q)
            #sim_dat_aux = sim_dat_aux.at[7,i].set(vessels[i].UM1A)
            #sim_dat_aux = sim_dat_aux.at[8,i].set(vessels[i].UM2Q)
            #sim_dat_aux = sim_dat_aux.at[9,i].set(vessels[i].UM2A)


        (vessels,counter) = jax.lax.cond(current_time >= timepoints[counter], 
                                         lambda: (saveTempDatas(current_time,vessels,counter),counter+1), 
                                         lambda: (vessels,counter))

        def checkConv():
            err = computeConvError(vessels)
            printConvError(err)

        jax.lax.cond(((current_time - ini.HEART.cardiac_T * passed_cycles >= ini.HEART.cardiac_T)*
                       (current_time - ini.HEART.cardiac_T * passed_cycles + dt > ini.HEART.cardiac_T)*
                       (passed_cycles + 1 > 1)), 
                       checkConv,
                        lambda: None)
        (vessels,counter,timepoints,passed_cycles) = jax.lax.cond(((current_time - ini.HEART.cardiac_T * passed_cycles >= ini.HEART.cardiac_T)*
                                            (current_time - ini.HEART.cardiac_T * passed_cycles + dt > ini.HEART.cardiac_T)), 
                                         lambda: (transferTempToLasts(vessels),0,timepoints + ini.HEART.cardiac_T, passed_cycles+1), 
                                         lambda: (vessels,counter,timepoints, passed_cycles))
        


        current_time += dt
        (passed_cycles) = jax.lax.cond(current_time >= ini.TOTAL_TIME,
                                         lambda: (passed_cycles+1), 
                                         lambda: (passed_cycles))

        return vessels, sim_dat, sim_dat_aux, current_time, counter, timepoints, passed_cycles, dt


    jax.lax.while_loop(cond_fun, body_fun, (vessels, sim_dat, sim_dat_aux, current_time,counter,timepoints,passed_cycles,dt))
    
    return current_time
    
