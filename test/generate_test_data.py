import jax
import sys

sys.path.append("/home/diego/studies/uni/thesis_maths/jaxFlowSim")
from src.model import run_simulation, run_simulation_unsafe
import time
import os
from functools import partial
from jax import block_until_ready, jit
import numpy as np

os.chdir(os.path.dirname(__file__) + "/..")

test_data_path = "test/test_data"
if not os.path.exists(test_data_path):
    os.mkdir(test_data_path)

jax.config.update("jax_enable_x64", True)

modelnames = [
    "single-artery",
    "tapering",
    "conjunction",
    "bifurcation",
    "aspirator",
    "adan56",
    "0007_H_AO_H",
    "0029_H_ABAO_H",
    "0053_H_CERE_H",
]

for modelname in modelnames:
    config_filename = "test/" + modelname + "/" + modelname + ".yml"
    verbose = True
    sim_dat, t, P = run_simulation(config_filename, verbose)

    np.savetxt("test/test_data/" + modelname + "_sim_dat.dat", sim_dat)
    np.savetxt("test/test_data/" + modelname + "_t.dat", t)
    np.savetxt("test/test_data/" + modelname + "_P.dat", P)

for modelname in modelnames:
    config_filename = "test/" + modelname + "/" + modelname + ".yml"
    verbose = True
    sim_dat, t, P = run_simulation_unsafe(config_filename, verbose)

    np.savetxt("test/test_data/" + modelname + "_sim_dat_unsafe.dat", sim_dat)
    np.savetxt("test/test_data/" + modelname + "_t_unsafe.dat", t)
    np.savetxt("test/test_data/" + modelname + "_P_unsafe.dat", P)
