from src.model import configSimulation, simulationLoopUnsafe
import jax
import sys
import time
import os
from functools import partial
from jax import block_until_ready, jit
import numpy as np
from SALib import ProblemSpec


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

sim_loop_jit = partial(jit, static_argnums=(0, 1, 12))(simulationLoopUnsafe)
sim_dat_base, t_t_base, P_t_base = block_until_ready(sim_loop_jit(N, B,
                                      sim_dat, sim_dat_aux, 
                                      sim_dat_const, sim_dat_const_aux, 
                                      Ccfl, input_data, rho, 
                                      masks, strides, edges,
                                      upper=120000))

starts = strides[:,0]
ends = strides[:,1]
def sim_loop_jit_wrapper(rho, Ccfl, Ls, R0s, Es, R1s, R2s, Ccs):
    
    for i in range(N):
       sim_dat_const[0,starts[i]-B:ends[i]+B] = np.pi*R0s[i]*R0s[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[3,starts[i]-B:ends[i]+B] = Es[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[7,starts[i]-B:ends[i]+B] = R1s[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[8,starts[i]-B:ends[i]+B] = R2s[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[9,starts[i]-B:ends[i]+B] = Ccs[i]*np.ones(ends[i]-starts[i]+2*B)
       sim_dat_const[10,starts[i]-B:ends[i]+B] = Ls[i]*np.ones(ends[i]-starts[i]+2*B)


    sim_dat_new, t_t_new, P_t_new = sim_loop_jit(N, B,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges,
                                          upper=120000)
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

print(P_t_base)

def quick_wrap(N, B,
                    sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                    Ccfl, edges, input_data, 
                    rho, strides,
                    masks, upper=120000):

                    _, _, P_t_new = sim_loop_jit(N, B,
                                sim_dat, sim_dat_aux, 
                                sim_dat_const, sim_dat_const_aux, 
                                Ccfl, input_data, rho, 
                                masks, strides, edges,
                                upper=120000)
                    return np.linalg.norm(P_t_base-P_t_new)

def wrapped_linear(X: np.ndarray, l = 4, func=sim_loop_jit) -> np.ndarray:
    import numpy as np
    M, D = X.shape
    results = np.empty(M)
    for i in range(M):
        #R01, R02, R03, E1, E2, E3, R11, R12, R21, R22, Cc1, Cc2 = X[i, :] #, L1, L2, L3 = X[i, :]
        #E1, E2, E3, R11, R12, R21, R22, Cc1, Cc2 = X[i, :] #, L1, L2, L3 = X[i, :]
        #R01, R02, R03, R11, R12, R21, R22, Cc1, Cc2 = X[i, :] #, L1, L2, L3 = X[i, :]
        (A01, A02, A03, 
            beta1, beta2, beta3, 
            gamma1, gamma2, gamma3, 
            viscT1, viscT2, viscT3, 
            R11, R12, R21, R22, Cc1, Cc2, 
            L1, L2, L3, rho, Ccfl) = X[i,:]

        sim_dat_const[0,starts[0]-B:ends[0]+B] = A01*np.ones(ends[0]-starts[0]+2*B)
        sim_dat_const[0,starts[1]-B:ends[1]+B] = A02*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[0,starts[2]-B:ends[2]+B] = A03*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[1,starts[0]-B:ends[0]+B] = beta1*np.ones(ends[0]-starts[0]+2*B)
        sim_dat_const[1,starts[1]-B:ends[1]+B] = beta2*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[1,starts[2]-B:ends[2]+B] = beta3*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[2,starts[0]-B:ends[0]+B] = gamma1*np.ones(ends[0]-starts[0]+2*B)
        sim_dat_const[2,starts[1]-B:ends[1]+B] = gamma2*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[2,starts[2]-B:ends[2]+B] = gamma3*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[5,starts[0]-B:ends[0]+B] = viscT1*np.ones(ends[0]-starts[0]+2*B)
        sim_dat_const[5,starts[1]-B:ends[1]+B] = viscT2*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[5,starts[2]-B:ends[2]+B] = viscT3*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[7,starts[1]-B:ends[1]+B] = R11*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[7,starts[2]-B:ends[2]+B] = R12*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[8,starts[1]-B:ends[1]+B] = R21*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[8,starts[2]-B:ends[2]+B] = R22*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[9,starts[1]-B:ends[1]+B] = Cc1*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[9,starts[2]-B:ends[2]+B] = Cc2*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_const[10,starts[0]-B:ends[0]+B] = L1*np.ones(ends[0]-starts[0]+2*B)
        sim_dat_const[10,starts[1]-B:ends[1]+B] = L2*np.ones(ends[1]-starts[1]+2*B)
        sim_dat_const[10,starts[2]-B:ends[2]+B] = L3*np.ones(ends[2]-starts[2]+2*B)

        sim_dat_new, _, _ = func(N, B,
                                sim_dat, sim_dat_aux, 
                                sim_dat_const, sim_dat_const_aux, 
                                Ccfl, input_data, rho, 
                                masks, strides, edges,
                                upper=120000)
        
        #print(np.linalg.norm(sim_dat_base[l,:]-sim_dat_new[l,:])/np.linalg.norm(sim_dat_base[l,:]))
        results[i] = np.linalg.norm(sim_dat_base[l,:]-sim_dat_new[l,:])/np.linalg.norm(sim_dat_base[l,:])

    return results

print(sim_dat_const[:,starts[0]])
print(sim_dat_const[:,starts[1]])
print(sim_dat_const[:,starts[2]])

sp = ProblemSpec({
    #'names': ['R01', 'R02', 'R03', 'E1', 'E2', 'E3', 'R11', 'R12', 'R21', 'R22', 'Cc1', 'Cc2'], #, 'L1', 'L2', 'L3'],
    'names': ['A01', 'A02', 'A03', 
              'beta1', 'beta2', 'beta3', 
              'gamma1', 'gamma2', 'gamma3', 
              'viscT1', 'viscT2', 'viscT3', 
              'R11', 'R12', 'R21', 'R22', 'Cc1', 'Cc2', 
              'L1', 'L2', 'L3', 'rho', 'Ccfl'],
    #'names': ['R01', 'R02', 'R03', 'R11', 'R12', 'R21', 'R22', 'Cc1', 'Cc2'], #, 'L1', 'L2', 'L3'],
    'bounds': [
        [1.80619998e-04*0.9,1.80619998e-04*1.1],
        [9.47569187e-05*0.9,9.47569187e-05*1.1],
        [9.47569187e-05*0.9,9.47569187e-05*1.1],
        [8.51668358e+04*0.9,8.51668358e+04*1.1],
        [1.32543517e+05*0.9,1.32543517e+05*1.1],
        [1.32543517e+05*0.9,1.32543517e+05*1.1],
        [1.99278514e+03*0.9,1.99278514e+03*1.1],
        [4.28179535e+03*0.9,4.28179535e+03*1.1],
        [4.28179535e+03*0.9,4.28179535e+03*1.1],
        [2.60811466e-04*0.9,2.60811466e-04*1.1],
        [2.60811466e-04*0.9,2.60811466e-04*1.1],
        [2.60811466e-04*0.9,2.60811466e-04*1.1],
        #[0.758242250e-2*0.9, 0.758242250e-2*1.1],
        #[0.5492e-2*0.9, 0.5492e-2*1.1],
        #[0.5492e-2*0.9, 0.5492e-2*1.1],
        #[500.0e3*0.9999, 500.0e3*1],
        #[700.0e3*0.9999, 700.0e3*1],
        #[700.0e3*0.9999, 700.0e3*1],
        [6.8123e7*0.9, 6.8123e7*1.1],
        [6.8123e7*0.9, 6.8123e7*1.1],
        [3.1013e9*0.9, 3.1013e9*1.1],
        [3.1013e9*0.9, 3.1013e9*1.1],
        [3.6664e-10*0.9, 3.6664e-10*1.1],
        [3.6664e-10*0.9, 3.6664e-10*1.1],
        [1e-3*0.9, 1e-3*1.1],
        [1e-3*0.9, 1e-3*1.1],
        [1e-3*0.9, 1e-3*1.1],
        [rho*0.9, rho*1.1],
        [Ccfl*0.9, Ccfl*1.1],
    ],
})

if __name__ == "__main__":
    (sp.sample_sobol(2**8).evaluate(wrapped_linear).analyze_sobol()
    )

sp.to_df()
print(sp.to_df())