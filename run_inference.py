from src.model import configSimulation, simulationLoopUnsafe
from numpyro.infer.reparam import TransformReparam
import os
import sys
import time
from functools import partial
from jax import jit
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC
import numpy as np
import itertools

os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)

numpyro.set_host_device_count(1)

config_filename = ""
if len(sys.argv) == 1:

    # base cases
    #modelname = "single-artery"
    #modelname = "tapering"
    #modelname = "conjunction"
    #modelname = "bifurcation"
    #modelname = "aspirator"

    # openBF-hub 
    modelname = "test/adan56/adan56.yml"

    # vascularmodels.com
    #modelname = "0007_H_AO_H"
    #modelname = "0029_H_ABAO_H"
    #modelname = "0053_H_CERE_H"
    input_filename = "test/" + modelname + "/" + modelname + ".yml"

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
sim_dat_obs, t_obs, P_obs = sim_loop_old_jit(N, B,
                                      sim_dat, sim_dat_aux, 
                                      sim_dat_const, sim_dat_const_aux, 
                                      Ccfl, input_data, rho, 
                                      masks, strides, edges,
                                      upper=120000)

R_index = 1
var_index = 7
R1 = sim_dat_const[var_index,strides[R_index,1]]
#R_scales = np.linspace(1.1*R1, 2*R1, 16)
R_scales = np.linspace(0.5*R1, 0.9*R1, 16)
R_scale = R_scales[int(sys.argv[2])]
print(R1, R_scale)
def simLoopWrapper(R):
    ones = jnp.ones(strides[R_index,1]-strides[R_index,0]+4)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[var_index,strides[R_index,0]-2:strides[R_index,1]+2].set(R*ones)
    _, _, P = sim_loop_old_jit(N, B,
                                          sim_dat, sim_dat_aux, 
                                          sim_dat_const_new, sim_dat_const_aux, 
                                          Ccfl, input_data, rho, 
                                          masks, strides, edges,
                                          upper=120000)
    return P

sim_loop_wrapper_jit = jit(simLoopWrapper)

def logp(y, R, sigma):
    y_hat = sim_loop_wrapper_jit(R)
    L = jnp.mean(jax.scipy.stats.norm.pdf(((y - y_hat)).flatten(), loc = 0, scale=sigma))
    jax.debug.print("L = {x}", x=L)
    return L

def model(P_obs, sigma, scale, R_scale):
    with numpyro.handlers.reparam(config={'theta': TransformReparam()}):
        R_dist = numpyro.sample(
            'theta',
            dist.TransformedDistribution(dist.Normal(),
                                         dist.transforms.AffineTransform(0, scale*R_scale)))
    jax.debug.print("R_dist = {x}", x=R_dist)
    log_density = logp(P_obs, R_dist, sigma=sigma) 
    numpyro.factor("custom_logp", log_density)

#network_properties = {
#        "sigma": [1e-1, 1e-3, 1e-5, 1e-7, 1e-9],
#        "scale": [1, 5, 10],
#        "num_warmup": np.arange(10, 60, 10),
#        "num_samples": np.arange(100, 600, 100),
#        "num_chains": [1]
#        }

network_properties = {
        "sigma": [1e-5],
        "scale": [10],
        "num_warmup": np.arange(10, 110, 10),
        "num_samples": np.arange(100, 1100, 100),
        "num_chains": [1]
        }

#network_properties = {
#        "sigma": [1e-5],
#        "scale": [10],
#        "num_warmup": np.arange(10, 20, 10),
#        "num_samples": np.arange(10, 20, 10),
#        "num_chains": [1]
#        }

settings = list(itertools.product(*network_properties.values()))

results_folder = "results/inference_ensemble"
if not os.path.isdir(results_folder):
    os.makedirs(results_folder, mode = 0o777)

for set_num, setup in enumerate(settings):
    print("###################################", set_num, "###################################")
    setup_properties = {
            "sigma": setup[0],
            "scale": setup[1],
            "num_warmup": setup[2],
            "num_samples": setup[3],
            "num_chains": setup[4]
            }
    results_file = results_folder  + "/setup_" + str(setup_properties["sigma"]) + "_" + str(setup_properties["scale"]) + "_" + str(setup_properties["num_warmup"]) + "_" + str(setup_properties["num_samples"]) + "_" + str(setup_properties["num_chains"]) + ".txt"
    mcmc = MCMC(numpyro.infer.NUTS(model, 
                                   forward_mode_differentiation=True),
                                   num_samples=setup_properties["num_samples"],
                                   num_warmup=setup_properties["num_warmup"],
                                   num_chains=setup_properties["num_chains"])
    mcmc.run(jax.random.PRNGKey(3450), 
             P_obs, setup_properties["sigma"], setup_properties["scale"], R_scale)
    mcmc.print_summary()
    R = jnp.mean(mcmc.get_samples()["theta"])

    #print(mcmc.get_samples())
    file = open(results_file, "a")  
    file.write(str(R_scale) + " " + str(R) + "  " + str(R1) + "\n")
    file.close()