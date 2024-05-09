from src.model import configSimulation, simulationLoopUnsafe
import jax
import sys
import time
import os
from functools import partial
from jax import block_until_ready, jit
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)

config_filename = ""
if len(sys.argv) == 1:
    # base cases
    #input_filename = "test/single-artery/single-artery.yml"
    #input_filename = "test/tapering/tapering.yml"
    #input_filename = "test/conjunction/conjunction.yml"
    #input_filename = "test/bifurcation/bifurcation.yml"
    #input_filename = "test/aspirator/aspirator.yml"

    # openBF-hub 
    config_filename = "test/adan56/adan56.yml"

    # vascularmodels.com
    #modelname = "0007_H_AO_H"
    #modelname = "0029_H_ABAO_H"
    #modelname = "0053_H_CERE_H"
    #input_filename = "test/" + modelname + "/" + modelname + ".yml"
else:
    config_filename = "test/" + sys.argv[1] + "/" + sys.argv[1] + ".yml"

verbose = True
(N, B, J, 
 sim_dat, sim_dat_aux, 
 sim_dat_const, sim_dat_const_aux, 
 timepoints, conv_tol, Ccfl, edges, input_data, rho, 
 masks, strides, edges,
 vessel_names, cardiac_T) = configSimulation(config_filename, verbose)

if verbose:
    starting_time = time.time_ns()

sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 12))(simulationLoopUnsafe)
sim_dat, t_t, P_t = block_until_ready(sim_loop_old_jit(N, B,
                                      sim_dat, sim_dat_aux, 
                                      sim_dat_const, sim_dat_const_aux, 
                                      Ccfl, input_data, rho, 
                                      masks, strides, edges,
                                      upper=120000))

if verbose:
    ending_time = (time.time_ns() - starting_time) / 1.0e9
    print(f"elapsed time = {ending_time} seconds")

#jnp.set_printoptions(threshold=sys.maxsize)
filename = config_filename.split("/")[-1]
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

#plt.rcParams.update({'font.size': 20})

#print(sim_dat)

indices = [i+1 for i in range(len(t_t)-1) if t_t[i]>t_t[i+1]]
P_cycle = P_t[indices[-2]:indices[-1],:]
t_cycle = t_t[indices[-2]:indices[-1]]
P_cycle = P_t[indices[-2]:indices[-1],:]
t_cycle = t_t[indices[-2]:indices[-1]]
P0_temp = np.loadtxt("/home/diego/studies/uni/thesis_maths/openBF/test/" + network_name + "/" + network_name + "_results/" + vessel_names[0] + "_P.last")
t0 = P0_temp[:,0]%cardiac_T

counter = 0
t_new = np.zeros(len(timepoints))
P_new = np.zeros((len(timepoints), 5*N))
for i in range(len(t_cycle)-1):
    if t0[counter] >= t_cycle[i] and t0[counter] <= t_cycle[i+1]:
        P_new[counter,:] = (P_cycle[i,:] + P_cycle[i+1,:])/2
        counter += 1
for i in range(len(t_cycle)-1):
    if t0[counter] >= t_cycle[i] and t0[counter] <= t_cycle[i+1]:
        P_new[counter,:] = (P_cycle[i,:] + P_cycle[i+1,:])/2
        counter += 1



#even_indices = np.round(np.linspace(0, len(t) - 1, 100)).astype(int)
#P  = P[even_indices,:]
#t  = t[even_indices]

#counter = 0
#for i, t_temp in enumerate(t):
#    if t_temp >= timepoints[counter]:
#        t_new[counter] = t_temp
#        P_new[counter:] = P[i,:]
#        counter += 1

t_new = t_new[:-1]
P_new = P_new[:-1,:]


for i,vessel_name in enumerate(vessel_names):
    index_vessel_name = vessel_names.index(vessel_name)
    P0_temp = np.loadtxt("/home/diego/studies/uni/thesis_maths/openBF/test/" + network_name + "/" + network_name + "_results/" + vessel_name + "_P.last")
    node = 2
    index_jl  = 1 + node
    index_jax  = 5*index_vessel_name + node

    P0 = P0_temp[:-1,index_jl]
    t0 = P0_temp[:-1,0]%cardiac_T
    P1 = P_new[:,index_jax]
    res = np.sqrt(((P1-P0).dot(P1-P0)/P0.dot(P0)))
    print(res)
    _, ax = plt.subplots()
    ax.set_xlabel("t[s]")
    ax.set_ylabel("P[mmHg]")
    plt.title("network: " + network_name + ", # vessels: " + str(N) + ", vessel name: " + vessel_names[i] + ", \n relative error = |P_JAX-P_jl|/|P_jl| = " + str(res) + "%")
    plt.title("network: " + network_name + ", # vessels: " + str(N) + ", vessel name: " + vessel_names[i] + ", \n relative error = |P_JAX-P_jl|/|P_jl| = " + str(res) + "%")
    #plt.title("network: " + network_name + ", vessel name: " + vessel_names_0053[i])
    #plt.title(vessel_names_0053[i])
    #plt.title("vessel name: " + vessel_name)
    plt.plot(t0, P0/133.322)
    plt.plot(t0, P1/133.322)
    plt.plot(t0, P0/133.322)
    plt.plot(t0, P1/133.322)
    #print(P)
    #plt.plot(t%cardiac_T,P[:,index_jax]/133.322)
    #plt.plot(t0,P0/133.322)
    plt.legend(["P_JAX", "P_jl"], loc="lower right")
    plt.legend(["P_JAX", "P_jl"], loc="lower right")
    plt.tight_layout()
    plt.savefig("results/" + network_name + "_results/" + network_name + "_" + vessel_names[i].replace(" ", "_") + "_P.pdf")
    plt.savefig("results/" + network_name + "_results/" + network_name + "_" + vessel_names[i].replace(" ", "_") + "_P.pdf")
    plt.close()
