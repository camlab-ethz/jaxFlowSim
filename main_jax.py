from src.program import runSimulation_opt
import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices
from jax.config import config
import jax
import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=32' # Use 8 CPU devices

#jax.devices("gpu")[0]
config.update("jax_enable_x64", True)
#input_filename = "test/single-artery/single-artery.yml"
input_filename = "test/tapering/tapering.yml"
#input_filename = "test/conjunction/conjunction.yml"
#input_filename = "test/bifurcation/bifurcation.yml"
#input_filename = "test/aspirator/aspirator.yml"
#input_filename = "test/adan56/adan56.yml"
#conv_ceil = 0.001

#config.update('jax_disable_jit', True)
#with jax.checking_leaks():
#jax.distributed.initialize(num_processes=32)
runSimulation_opt(input_filename, verbose=True)
#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
