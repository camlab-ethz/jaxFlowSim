from src.model import configSimulation, simulationLoop
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
 sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
 timepoints, conv_tol, Ccfl, edges, input_data, 
            rho, strides, 
            indices_1, indices_2,
            vessel_names, cardiac_T, junction_functions, mask, mask1) = configSimulation(config_filename, verbose)#, junction_functions) = configSimulation(config_filename, verbose)
            

if verbose:
    starting_time = time.time_ns()
sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 2))(simulationLoop)
sim_dat, t, P  = block_until_ready(sim_loop_old_jit(N, B, J, 
                                      sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                      timepoints, conv_tol, Ccfl, edges, input_data, 
                                      rho, strides, 
                                      indices_1, indices_2)) #, junction_functions))

if verbose:
    ending_time = (time.time_ns() - starting_time) / 1.0e9
    print(f"elapsed time = {ending_time} seconds")

# save data for unittests
#np.savetxt("test/test_data/bifurcation_sim_dat.dat", sim_dat)
#np.savetxt("test/test_data/bifurcation_t.dat", t)
#np.savetxt("test/test_data/bifurcation_P.dat", P)

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
    #plt.legend(["P_JAX", "P_jl"], loc="lower right")
    #plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/" + network_name + "_results/" + network_name + "_" + vessel_names[i].replace(" ", "_") + "_P.pdf")#, bbox_inches='tight')
    plt.close()
