from functools import partial
import jax.numpy as jnp
import jax
from jax import block_until_ready, jit, lax, grad, jacfwd
import numpy as np
from src.initialise import loadConfig, buildBlood, buildArterialNetwork, makeResultsFolder
from src.IOutils import saveTempDatas#, writeResults
from src.solver import calculateDeltaT, solveModel
from src.check_convergence import printConvError, computeConvError, checkConvergence
import time
import os
import sys
import matplotlib.pyplot as plt
import numpyro
import numpy as np
import numpyro.distributions as dist

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



def runSimulation_opt(input_filename, verbose=False):
    data = loadConfig(input_filename)
    blood = buildBlood(data["blood"])

    #if verbose:
    #    print(f"Build {input_filename} arterial network \n")

    J =  data["solver"]["jump"]

    sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, N, B, edges, input_data, nodes, vessel_names, starts, ends, indices1, indices2 = buildArterialNetwork(data["network"], J, blood)
    makeResultsFolder(data, input_filename)

    cardiac_T = sim_dat_const_aux[0,0]
    total_time = data["solver"]["cycles"]*cardiac_T
    Ccfl = float(data["solver"]["Ccfl"])
    conv_tol = data["solver"]["convergence tolerance"],
    
    if verbose:
        print("Start simulation")


    print(edges)
    timepoints = np.linspace(0, cardiac_T, J)
    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    if verbose:
        print("Solving cardiac cycle no: 1")
        starting_time = time.time_ns()
    
    sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulation_loop_old)
    sim_loop_jit = partial(jit, static_argnums=(0, 1, 2))(simulation_loop)
    sim_dat, t, P_obs  = block_until_ready(sim_loop_old_jit(N, B, J, 
                                          sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                          timepoints, 1, Ccfl, edges, input_data, 
                                          blood.rho, total_time, nodes, 
                                          starts, ends,
                                          indices1, indices2))
    
    R_index = 0
    R1 = sim_dat_const[8,ends[R_index]]
    R = 0.8
    def simulation_loop_wrapper(R):
        ones = jnp.ones(ends[R_index]-starts[R_index]+4)
        sim_dat_const_new = jnp.array(sim_dat_const)
        sim_dat_const_new = sim_dat_const_new.at[8,starts[R_index]-2:ends[R_index]+2].set(R*ones)
        _, _, P = sim_loop_old_jit(N, B, J, 
                        sim_dat, sim_dat_aux, sim_dat_const_new, sim_dat_const_aux, 
                        timepoints, 1, Ccfl, edges, input_data, 
                        blood.rho, total_time, nodes, 
                        starts, ends,
                        indices1, indices2)
        return P[:,2]

    sim_loop_wrapper_jit = jit(simulation_loop_wrapper)
    def simulation_loop_loss_wrapper(R):
        ones = jnp.ones(ends[R_index]-starts[R_index])
        sim_dat_const_new = jnp.array(sim_dat_const)
        sim_dat_const_new = sim_dat_const_new.at[7,starts[R_index]:ends[R_index]].set(R*ones)
        _, _, P = sim_loop_old_jit(N, B, J, 
                        sim_dat, sim_dat_aux, sim_dat_const_new, sim_dat_const_aux, 
                        timepoints, 1, Ccfl, edges, input_data, 
                        blood.rho, total_time, nodes, 
                        starts, ends,
                        indices1, indices2) 
        return jnp.linalg.norm(P[:,2]-P_obs[:,2])/jnp.linalg.norm(P_obs[:,2])
    
    

    #print(jacfwd(simulation_loop_loss_wrapper,)(11700000.0*0.9))
    #print(np.size(P_obs))
    def logp(y, R):#, sigma):
        """The likelihood function for a linear model
        y ~ ax+b+error
        """
        jax.debug.print("R = {x}", x = R)
        y_hat = sim_loop_wrapper_jit(R*1e8) 
        jax.debug.print("shapes = {x}", x = (y, y_hat))
        #L = jnp.mean(jax.scipy.stats.norm.pdf((y - y_hat)/133.33, loc = 0, scale=sigma))
        L = jnp.mean((y - y_hat)/133.33)
        jax.debug.print("L = {x}", x = L)
        return L

    def model():
        R_dist=numpyro.sample("R", dist.Normal())
        jax.debug.print("R = {x}", x = R_dist)
        #sigma=numpyro.sample("sigma", dist.HalfNormal())
        log_density = logp(y=P_obs[:,2], R=R_dist)#, sigma=sigma)
        numpyro.factor("custom_logp", log_density)

    #def model():
    #    R_dist=numpyro.sample("R", dist.Normal(1,0.0001))
    #    with numpyro.plate("size", jnp.size(P_obs[:,2])):
    #        numpyro.sample("obs", dist.Normal(sim_loop_wrapper_jit(R*R_dist)), obs=P_obs[:,2])
    
    mcmc = MCMC(numpyro.infer.NUTS(model, forward_mode_differentiation=True,),num_samples=100,num_warmup=1,num_chains=1)
    mcmc.run(jax.random.PRNGKey(0))
    mcmc.print_summary()
    R = jnp.mean(mcmc.get_samples()["R"])


    print(R1)
    print(R) 
    #sim_dat, t, P  = simulation_loop(N, B, J, 
    #                                      sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
    #                                      timepoints, 1, Ccfl, edges, input_data, 
    #                                      blood.rho, total_time, nodes, 
    #                                      starts, ends, starts_rep, ends_rep,
    #                                      indices1, indices2).lower(42).compile()
    #simulation_loop.lower(N, B, J, 
    #                                      sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
    #                                      timepoints, 1, Ccfl, edges, input_data, 
    #                                      blood.rho, total_time, nodes, 
    #                                      starts, ends, starts_rep, ends_rep,
    #                                      indices1, indices2).compile()
    
    

    if verbose:
        print("\n")
        ending_time = (time.time_ns() - starting_time) / 1.0e9
        print(f"Elapsed time = {ending_time} seconds")
        #print(len(vessel_names), ending_time)

    #jnp.set_printoptions(threshold=sys.maxsize)
    #print(P)
    #plt.figure()
    #plt.plot(t,P[:,0])
    #plt.show()
    filename = input_filename.split("/")[-1]
    network_name = filename.split(".")[0]
    #vessel_name = "ulnar_R_I"

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
 
    #matplotlib.rcParams.update({'font.size': 20})

    #for i,vessel_name in enumerate(vessel_names):
    #    index_vessel_name = vessel_names.index(vessel_name)
    #    P0 = np.loadtxt("/home/diego/studies/uni/thesis_maths/openBF/test/" + network_name + "/" + network_name + "_results/" + vessel_name + "_P.last")
    #    #P0 = np.loadtxt("/home/diego/studies/uni/thesis_maths/openBF/test/adan56/adan56_results/adan56_results/aortic_arch_I_P.last")
    #    node = 2
    #    index_jl  = 1 + node
    #    index_jax  = 5*index_vessel_name + node
    #    P0 = P0[:,index_jl]
    #    res = np.sqrt(((P[:,index_jax]-P0).dot(P[:,index_jax]-P0)/P0.dot(P0)))
    #    #print(res)
    #    _, ax = plt.subplots()
    #    ax.set_xlabel("t[s]")
    #    ax.set_ylabel("P[mmHg]")
    #    plt.title("network: " + network_name + ", # vessels: " + str(N) + ", vessel name: " + vessel_names_0053[i] + ", \n relative error = |P_JAX-P_jl|/|P_jl| = " + str(res) + "%")
    #    #plt.title("network: " + network_name + ", vessel name: " + vessel_names_0053[i])
    #    #plt.title(vessel_names_0053[i])
    #    #plt.title("vessel name: " + vessel_name)
    #    plt.plot(t%cardiac_T,P[:,index_jax]/133.322)
    #    plt.plot(t%cardiac_T,P0/133.322)
    #    plt.legend(["P_JAX", "P_jl"], loc="lower right")
    #    plt.tight_layout()
    #    #print(network_name + "_" + vessel_name + "_P.eps")
    #    plt.savefig("results/" + network_name + "_results/" + network_name + "_" + vessel_names_0053[i].replace(" ", "_") + "_P.eps")
    #    plt.close()

    #plt.show()

    #print(edges)
    #writeResults(vessels)

def simulation_loop(N, B, jump, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, timepoints, conv_toll, Ccfl, edges, input_data, rho, total_time, nodes, starts, ends, indices1, indices2):
    t = 0.0
    passed_cycles = 0
    counter = 0
    P_t = jnp.empty((jump, N*5))
    t_t = jnp.empty((jump))
    P_l = jnp.empty((jump, N*5))
    dt = 0 
    not_conv = True

    while not_conv:
        dt = calculateDeltaT(Ccfl, sim_dat[0,:],sim_dat[3,:], sim_dat_const[-1,:])
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
        err = computeConvError(N, P_t, P_l)
        def printConvErrorWrapper():
            printConvError(err)
            return False
        not_conv = lax.cond((passed_cycles + 1 > 1)*(checkConvergence(err, conv_toll))*
                           ((t - sim_dat_const_aux[0,0] * passed_cycles >= sim_dat_const_aux[0,0])), 
                            printConvErrorWrapper,
                            lambda: True)


    
    return sim_dat, t_t, P_t


def simulation_loop_old(N, B, jump, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, timepoints, conv_toll, Ccfl, edges, input_data, rho, total_time, nodes, starts, ends, indices1, indices2):
    jax.debug.print("starting simulation")
    jax.debug.print("R2 = {R}", R=sim_dat_const[8,0])
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
        dt = calculateDeltaT(Ccfl, sim_dat[0,:],sim_dat[3,:], sim_dat_const[-1,:])
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
    
