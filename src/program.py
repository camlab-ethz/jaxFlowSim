import jax
import jax.numpy as jnp
import numpy as np
from src.initialise import loadSimulationFiles, buildBlood, buildArterialNetwork, buildConst, makeResultsFolder
from src.IOutils import saveTempDatas, writeResults
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

    sim_dat, sim_dat_aux = buildArterialNetwork(data["network"])
    makeResultsFolder(data, input_filename)

    buildConst(float(data["solver"]["Ccfl"]), data["solver"]["cycles"], data["solver"]["convergence tolerance"])
    
    if verbose:
        print("Start simulation")

    if verbose:
        #print("Solving cardiac cycle no: 1")
        starting_time = time.time_ns()

        #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        simulation_loop(sim_dat, sim_dat_aux)

    if verbose:
        #print("\n")
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"Elapsed time = {ending_time} seconds")

    #writeResults(vessels)

@jax.jit
def simulation_loop(sim_dat, sim_dat_aux):
    t = 0.0
    passed_cycles = 0
    counter = 0
    timepoints = np.linspace(0, ini.CARDIAC_TS[0], ini.JUMP)
    P_t = jnp.empty((ini.JUMP, ini.NUM_VESSELS*5), dtype=jnp.float64)
    P_l = jnp.empty((ini.JUMP, ini.NUM_VESSELS*5), dtype=jnp.float64)
    dt = 0 

    def cond_fun(args):
        _, _, t_i, _, _, passed_cycles_i, _, P_t_i, P_l_i = args
        err = computeConvError(P_t_i, P_l_i)
        def printConvErrorWrapper():
            printConvError(err)
            return False
        ret = jax.lax.cond((passed_cycles_i + 1 > 1)*(checkConvergence(err))*
                           ((t_i - ini.CARDIAC_TS[0] * passed_cycles_i >= ini.CARDIAC_TS[0])), 
                            printConvErrorWrapper,
                            lambda: True)
        return ret

    def body_fun(args):
        sim_dat, sim_dat_aux, t, counter, timepoints, passed_cycles, dt, P_t, P_l = args
        dt = calculateDeltaT(sim_dat[0,:],sim_dat[3,:])
        sim_dat, sim_dat_aux = solveModel(sim_dat, sim_dat_aux, dt, t)
        sim_dat_aux = sim_dat_aux.at[2:10,:].set(updateGhostCells(sim_dat))


        (P_t_temp,counter_temp) = jax.lax.cond(t >= timepoints[counter], 
                                         lambda: (saveTempDatas(sim_dat[4,:]),counter+1), 
                                         lambda: (P_t[counter,:],counter))
        P_t = P_t.at[counter,:].set(P_t_temp)
        counter = counter_temp

        def checkConv():
            err = computeConvError(P_t, P_l)
            printConvError(err)

        jax.lax.cond(((t - ini.CARDIAC_TS[0] * passed_cycles >= ini.CARDIAC_TS[0])*
                       (passed_cycles + 1 > 1)), 
                       checkConv,
                        lambda: None)
        (P_l,counter,timepoints,passed_cycles) = jax.lax.cond(((t - ini.CARDIAC_TS[0] * passed_cycles >= ini.CARDIAC_TS[0])*
                                            (t - ini.CARDIAC_TS[0] * passed_cycles + dt > ini.CARDIAC_TS[0])), 
                                         lambda: (P_t,0,timepoints + ini.CARDIAC_TS[0], passed_cycles+1), 
                                         lambda: (P_l,counter,timepoints, passed_cycles))
        


        t += dt
        (passed_cycles) = jax.lax.cond(t >= ini.TOTAL_TIME,
                                         lambda: (passed_cycles+1), 
                                         lambda: (passed_cycles))

        return sim_dat, sim_dat_aux, t, counter, timepoints, passed_cycles, dt, P_t, P_l


    jax.lax.while_loop(cond_fun, body_fun, (sim_dat, sim_dat_aux, t, counter, timepoints, passed_cycles, dt, P_t, P_l))
    
    return t
    
