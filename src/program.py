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

    vessels = buildArterialNetwork(data["network"])
    makeResultsFolder(data, input_filename)

    buildConst(float(data["solver"]["Ccfl"]), data["solver"]["cycles"], data["solver"]["convergence tolerance"])
    
    if verbose:
        print("Start simulation \n")

    if verbose:
        print("Solving cardiac cycle no: 1")
        starting_time = time.time_ns()

    simulation_loop(vessels)

    if verbose:
        print("\n")
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"Elapsed time = {ending_time} seconds")

    writeResults(vessels)

def simulation_loop(vessels):
    current_time = 0.0
    passed_cycles = 0
    counter = 0
    timepoints = np.linspace(0, ini.HEART.cardiac_T, ini.JUMP)
    dt = calculateDeltaT(vessels)

    @jax.jit
    def cond_fun(args):
        vessels, current_time, counter, timepoints, passed_cycles, dt = args
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
        vessels, current_time, counter, timepoints, passed_cycles, dt = args
        dt = calculateDeltaT(vessels)
        #jax.debug.breakpoint()
        vessels = solveModel(vessels, dt, current_time)
        #jax.debug.breakpoint()
        vessels = updateGhostCells(vessels)
        #jax.debug.breakpoint()

        
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

        return vessels, current_time, counter, timepoints, passed_cycles, dt


    jax.lax.while_loop(cond_fun, body_fun, (vessels,current_time,counter,timepoints,passed_cycles,dt))
    
    return current_time
    
