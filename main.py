from src.program import runSimulation
from jax import config
 
config.update("jax_enable_x64", True)
#input_filename = "test/single-artery/single-artery.yml"
#input_filename = "test/tapering/tapering.yml"
#input_filename = "test/conjunction/conjunction.yml"
#input_filename = "test/bifurcation/bifurcation.yml"
input_filename = "test/aspirator/aspirator.yml"
#input_filename = "test/adan56/adan56.yml"
conv_ceil = 0.001

#config.update('jax_disable_jit', True)

runSimulation(input_filename, verbose=True, out_files=True, conv_ceil=False)
