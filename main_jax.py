from src.program_test import runSimulation_opt
import os

import numpyro


#numpyro.set_platform("cpu")
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
#numpyro.set_host_device_count(8)
from jax.config import config
import jax
import os
import sys
#config.update("jax_debug_nans", True)
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32' # Use 8 CPU devices
os.chdir(os.path.dirname(__file__))

#jax.devices("cpu")[0]
config.update("jax_enable_x64", True)

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


#config.update('jax_disable_jit', True)
#with jax.checking_leaks():
#jax.distributed.initialize(num_processes=32)
runSimulation_opt(config_filename, verbose=True)
#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
