from src.inference import runSimulation_opt
import os
from jax.config import config
import os
import sys

os.chdir(os.path.dirname(__file__))
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


runSimulation_opt(config_filename, verbose=True)
