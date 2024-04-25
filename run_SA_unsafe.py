from src.model import configSimulation, simulationLoopUnsafe
import jax
import sys
import time
import os
from functools import partial
from jax import block_until_ready, jit
import matplotlib.pyplot as plt
import numpy as np
from sensitivity import SensitivityAnalyzer
from SALib import ProblemSpec
from SALib.sample.sobol import sample


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
 timepoints, conv_toll, Ccfl, edges, input_data, 
            rho, nodes, 
            starts, ends,
            indices_1, indices_2,
            vessel_names, cardiac_T) = configSimulation(config_filename, verbose)

if verbose:
    starting_time = time.time_ns()

sim_loop_jit = partial(jit, static_argnums=(0, 1, 15))(simulationLoopUnsafe)
sim_dat_base, P_t_base, t_t_base = block_until_ready(sim_loop_jit(N, B,
                                      sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                                      Ccfl, edges, input_data, 
                                      rho, nodes, 
                                      starts, ends,
                                      indices_1, indices_2, upper=120000))

def sim_loop_jit_wrapper(rho, Ccfl, Ls, R0s, Es, R1s, R2s, Ccs):
    
    for i in range(N):
       sim_dat_const[0,starts[i]-B:ends[i]+B] = np.pi*R0s[i]*R0s[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[3,starts[i]-B:ends[i]+B] = Es[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[7,starts[i]-B:ends[i]+B] = R1s[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[8,starts[i]-B:ends[i]+B] = R2s[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[9,starts[i]-B:ends[i]+B] = Ccs[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[10,starts[i]-B:ends[i]+B] = Ls[i]*np.ones(ends[i]-starts[i]+2*B)


    sim_dat_new, P_t_new, t_t_new =sim_loop_jit(N, B,
                sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                Ccfl, edges, input_data, 
                rho, nodes, 
                starts, ends,
                indices_1, indices_2, upper=120000)
    return np.linalg.norm(P_t_base-P_t_new)




grid = np.arange(0.1,2,0.5)
print(np.array([0, 6.8123e7, 6.8123e7])[np.newaxis,:]*grid[:,np.newaxis])

sensitivity_dict = {
    'rho': 1060.0*grid,
    'Ccfl': 4.e-3*grid,
    'Ls': np.array([8.6e-2, 8.5e-2, 8.5e-2])[np.newaxis,:]*grid[:,np.newaxis],
    'R0s': np.array([0.758242250e-2, 0.5492e-2, 0.5492e-2])[np.newaxis,:]*grid[:,np.newaxis],
    'Es': np.array([500.0e3,700.0e3,700.0e3,])[np.newaxis,:]*grid[:,np.newaxis],
    'R1s': np.array([0, 6.8123e7, 6.8123e7])[np.newaxis,:]*grid[:,np.newaxis],
    'R2s': np.array([0, 3.1013e9, 3.1013e9])[np.newaxis,:]*grid[:,np.newaxis],
    'Ccs': np.array([0, 3.6664e-10, 3.6664e-10])[np.newaxis,:]*grid[:,np.newaxis],
}
    
#sa = SensitivityAnalyzer(sensitivity_dict, sim_loop_jit_wrapper)
#
#sa.df
#print(sa.df)


def wrapped_linear(X: np.ndarray, rho=rho, Ccfl=Ccfl, func=sim_loop_jit) -> np.ndarray:
    import numpy as np
    N, D = X.shape
    results = np.empty(N)
    for i in range(N):
        R01, R02, R03, E1, E2, E3, R11, R12, R21, R22, Cc1, Cc2, L1, L2, L3 = X[i, :]

        sim_dat_const[0,starts[0]-B:ends[0]+B] = np.pi*R01*R01*np.ones(ends[0]-starts[0]+2*B)
        sim_dat_const[0,starts[1]-B:ends[1]+B] = np.pi*R02*R02*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[0,starts[2]-B:ends[2]+B] = np.pi*R03*R03*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[3,starts[0]-B:ends[0]+B] = E1*np.ones(ends[0]-starts[0]+2*B)
        sim_dat_const[3,starts[1]-B:ends[1]+B] = E2*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[3,starts[2]-B:ends[2]+B] = E3*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[7,starts[1]-B:ends[1]+B] = R11*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[7,starts[2]-B:ends[2]+B] = R12*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[8,starts[1]-B:ends[1]+B] = R21*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[8,starts[2]-B:ends[2]+B] = R22*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[9,starts[1]-B:ends[1]+B] = Cc1*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[9,starts[2]-B:ends[2]+B] = Cc2*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[10,starts[0]-B:ends[0]+B] = L1*np.ones(ends[0]-starts[0]+2*B)
        sim_dat_const[10,starts[1]-B:ends[1]+B] = L2*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[10,starts[2]-B:ends[2]+B] = L3*np.ones(ends[2]-starts[2]+2*B)

        _, P_t_new, _ = func(N, B,
                    sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                    Ccfl, edges, input_data, 
                    rho, nodes, 
                    starts, ends,
                    indices_1, indices_2, upper=120000)
        results[i] = np.linalg.norm(P_t_base-P_t_new)

    return results

sp = ProblemSpec({
    'names': ['R01', 'R02', 'R03', 'E1', 'E2', 'E3', 'R11', 'R12', 'R21', 'R22', 'Cc1', 'Cc2', 'L1', 'L2', 'L3'],
    'bounds': [
        [0.758242250e-2*0.1, 0.758242250e-2*10],
        [0.5492e-2*0.1, 0.5492e-2*10],
        [0.5492e-2*0.1, 0.5492e-2*10],
        [500.0e3*0.1, 500.0e3*10],
        [700.0e3*0.1, 700.0e3*10],
        [700.0e3*0.1, 700.0e3*10],
        [6.8123e7*0.1, 6.8123e7*10],
        [6.8123e7*0.1, 6.8123e7*10],
        [3.1013e9*0.1, 3.1013e9*10],
        [3.1013e9*0.1, 3.1013e9*10],
        [3.6664e-10*0.1, 3.6664e-10+10],
        [3.6664e-10*0.1, 3.6664e-10+10],
        [8.6e-2*0.1, 8.6e-2*10],
        [8.5e-2*0.1, 8.5e-2*10],
        [8.5e-2*0.1, 8.5e-2*10],
    ],
})

sp.sample_sobol(2**6).evaluate(wrapped_linear).analyze_sobol()####