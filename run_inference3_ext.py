from src.model import configSimulation, simulationLoopUnsafe
from numpyro.infer.reparam import TransformReparam
import os
from jax.config import config
import sys
from functools import partial
from jax import jit
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC

os.chdir(os.path.dirname(__file__))
config.update("jax_enable_x64", True)

numpyro.set_host_device_count(8)

config_filename = ""
if len(sys.argv) == 1:
    # base cases
    #config_filename = "test/single-artery/single-artery.yml"
    #config_filename = "test/tapering/tapering.yml"
    #config_filename = "test/conjunction/conjunction.yml"
    config_filename = "test/bifurcation/bifurcation.yml"
    #config_filename = "test/aspirator/aspirator.yml"

    # openBF-hub 
    #config_filename = "test/adan56/adan56.yml"

    # vascularmodels.com
    #modelname = "0007_H_AO_H"
    #modelname = "0029_H_ABAO_H"
    #modelname = "0053_H_CERE_H"
    #config_filename = "test/" + modelname + "/" + modelname + ".yml"
else:
    config_filename = "test/" + sys.argv[1] + "/" + sys.argv[1] + ".yml"


# compute baseline to fit
verbose = True
(N, B, J, 
 sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
 timepoints, conv_toll, Ccfl, edges, input_data, 
 rho, nodes, 
 starts, ends,
 indices1, indices2,
 vessel_names, cardiac_T) = configSimulation(config_filename, verbose)

simLoopJit = partial(jit, static_argnums=(0, 1, 15))(simulationLoopUnsafe)
sim_dat_new, _, P_obs  = simLoopJit(N, B,
                    sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, 
                    Ccfl, edges, input_data, 
                    rho, nodes, 
                    starts, ends,
                    indices1, indices2, 120000)

# store original Windkessel parameter R1_real and initial guess
R1_vessel_index = 1
R1_const_index = 7
R1_real = sim_dat_const[R1_const_index,ends[R1_vessel_index]]
R1_init = 1e8

# wrapper that takes value for R1 and returns simulated pressure
def simLoopWrapper(R1, sim_dat):

    R1 = R1*jnp.ones(ends[R1_vessel_index]-starts[R1_vessel_index]+4)
    sim_dat_const_new = jnp.array(sim_dat_const)
    sim_dat_const_new = sim_dat_const_new.at[R1_const_index,starts[R1_vessel_index]-2:
                                             ends[R1_vessel_index]+2].set(R1)

    sim_dat, _, P = simulationLoopUnsafe(N, B,
                    sim_dat, sim_dat_aux, sim_dat_const_new, sim_dat_const_aux, 
                    Ccfl, edges, input_data, 
                    rho, nodes, 
                    starts, ends,
                    indices1, indices2, 50000)
    return sim_dat

simLoopWrapperJit = jit(simLoopWrapper)


# define custom log-likelihood
def logp(P_obs, R1, sigma, sim_dat):
    P_new = simLoopWrapperJit(R1, sim_dat)
    L = jnp.mean(jax.scipy.stats.norm.pdf((P_obs - P_new).flatten(), loc = 0, scale=sigma))
    jax.debug.print("R1 = {x}", x = R1)
    jax.debug.print("L = {x}", x = L)
    jax.debug.print("{x}", x=mcmc._args[2])
    return L, sim_dat

# define model for numpyro 
def model(P_obs, R1_init, sim_dat):
    with numpyro.handlers.reparam(config={'theta': TransformReparam()}):
        R1_dist = numpyro.sample(
            'theta',
            dist.TransformedDistribution(dist.Normal(),
                                         dist.transforms.AffineTransform(0, R1_init)))
    log_density, sim_dat = logp(P_obs, R1_dist, 1e-5, sim_dat) 
    mcmc._args = (P_obs, R1_init, sim_dat)
    numpyro.factor("custom_logp", log_density)

#run inference
mcmc = MCMC(numpyro.infer.NUTS(model, forward_mode_differentiation=True), 
            num_samples=100, num_warmup=10, num_chains=1, )
mcmc.run(jax.random.PRNGKey(3450), sim_dat_new, R1_init, sim_dat,)

#output results
mcmc.print_summary()
R1_est = jnp.mean(mcmc.get_samples()["theta"])
print(R1_real)
print(R1_est) 
    