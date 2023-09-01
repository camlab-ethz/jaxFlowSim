from functools import partial
import jax
import numpy as np
from src.initialise import loadSimulationFiles, buildBlood, buildArterialNetwork, makeResultsFolder
from src.IOutils import saveTempDatas, transferTempToLasts, writeResults
from src.boundary_conditions import updateGhostCells
from src.solver import calculateDeltaT, solveModel
from src.check_convergence import printConvError, computeConvError, checkConvergence
import time


def runSimulation_opt(input_filename, verbose=False):
    data = loadSimulationFiles(input_filename)
    blood = buildBlood(data["blood"])

    if verbose:
        print(f"Build {input_filename} arterial network \n")

    jump = data["solver"]["jump"]

    vessels, edges = buildArterialNetwork(data["network"], blood, jump)
    makeResultsFolder(data, input_filename)

    Ccfl = float(data["solver"]["Ccfl"])
    heart = vessels[0].heart
    total_time = data["solver"]["cycles"] * float(heart.cardiac_T)

    
    if verbose:
        print("Start simulation \n")



    conv_toll = data["solver"]["convergence tolerance"]

    if verbose:
        print("Solving cardiac cycle no: 1")
        starting_time = time.time_ns()

    simulation_loop(vessels, Ccfl, edges, blood, heart, conv_toll, total_time, jump)

    if verbose:
        print("\n")
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"Elapsed time = {ending_time} seconds")

    writeResults(vessels)

@partial(jax.jit, static_argnums=(1,2,3,4,5,6,7, ))
def simulation_loop(vessels, Ccfl, edges, blood, heart, conv_toll, total_time, jump):
    current_time = 0.0
    passed_cycles = 0
    counter = 0
    timepoints = np.linspace(0, heart.cardiac_T, jump)
    dt = calculateDeltaT(vessels, Ccfl)

    @jax.jit
    def cond_fun(args):
        vessels, current_time, counter, timepoints, passed_cycles, dt = args
        err = computeConvError(vessels)
        def fun1():
            printConvError(err)
            return False
        ret = jax.lax.cond((passed_cycles + 1 > 1)*(checkConvergence(err, conv_toll))*
                           ((current_time - heart.cardiac_T * passed_cycles >= heart.cardiac_T)*
                            (current_time - heart.cardiac_T * passed_cycles + dt > heart.cardiac_T)), 
                            fun1,
                            lambda: True)
        return ret

    @jax.jit
    def body_fun(args):
        vessels, current_time, counter, timepoints, passed_cycles, dt = args
        dt = calculateDeltaT(vessels, Ccfl)
        #jax.debug.breakpoint()
        vessels = solveModel(vessels, edges, blood, dt, current_time)
        #jax.debug.breakpoint()
        vessels = updateGhostCells(vessels)
        #jax.debug.breakpoint()

        (vessels,counter) = jax.lax.cond(current_time >= timepoints[counter], 
                                         lambda: (saveTempDatas(current_time,vessels,counter),counter+1), 
                                         lambda: (vessels,counter))

        def checkConv():
            err = computeConvError(vessels)
            printConvError(err)

        jax.lax.cond(((current_time - heart.cardiac_T * passed_cycles >= heart.cardiac_T)*
                       (current_time - heart.cardiac_T * passed_cycles + dt > heart.cardiac_T)*
                       (passed_cycles + 1 > 1)), 
                       checkConv,
                        lambda: None)
        (vessels,counter,timepoints,passed_cycles) = jax.lax.cond(((current_time - heart.cardiac_T * passed_cycles >= heart.cardiac_T)*
                                            (current_time - heart.cardiac_T * passed_cycles + dt > heart.cardiac_T)), 
                                         lambda: (transferTempToLasts(vessels),0,timepoints + heart.cardiac_T, passed_cycles+1), 
                                         lambda: (vessels,counter,timepoints, passed_cycles))
        


        current_time += dt
        (passed_cycles) = jax.lax.cond(current_time >= total_time,
                                         lambda: (passed_cycles+1), 
                                         lambda: (passed_cycles))

        return vessels, current_time, counter, timepoints, passed_cycles, dt


    jax.lax.while_loop(cond_fun, body_fun, (vessels,current_time,counter,timepoints,passed_cycles,dt))
    
    return current_time
    
