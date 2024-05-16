import unittest
import jax
import sys
sys.path.append('/home/diego/studies/uni/thesis_maths/jaxFlowSim')
print('Updated sys.path:', sys.path)
from src.model import runSimulation, runSimulationUnsafe
import time
import os
from functools import partial
from jax import block_until_ready, jit
import numpy as np

os.chdir(os.path.dirname(__file__)+"/..")

jax.config.update("jax_enable_x64", True)

class TestModels(unittest.TestCase):

    def test_models(self):

        modelnames = ["single-artery", 
                           "tapering",
                           "conjunction",
                           "bifurcation",
                           "aspirator",
                           "adan56",
                           "0007_H_AO_H",
                           "0029_H_ABAO_H",
                           "0053_H_CERE_H"]

        for modelname in modelnames:
            config_filename = "test/" + modelname + "/" + modelname + ".yml"

            verbose = True
            sim_dat, t, P  = runSimulation(config_filename, verbose)

            P_base = np.loadtxt("test/test_data/" + modelname + "_P.dat")
            sim_dat_base = np.loadtxt("test/test_data/" + modelname + "_sim_dat.dat")
            t_base = np.loadtxt("test/test_data/" + modelname + "_t.dat")

            #np.testing.assert_almost_equal(P, P_base)
            #np.testing.assert_almost_equal(sim_dat,sim_dat_base)
            #np.testing.assert_almost_equal(t, t_base)

    def test_models_unsafe(self):

        modelnames = ["single-artery", 
                           "tapering",
                           "conjunction",
                           "bifurcation",
                           "aspirator",
                           "adan56",
                           "0007_H_AO_H",
                           "0029_H_ABAO_H",
                           "0053_H_CERE_H"]

        for modelname in modelnames:
            config_filename = "test/" + modelname + "/" + modelname + ".yml"

            verbose = True
            sim_dat, t, P  = runSimulationUnsafe(config_filename, verbose)

            P_base = np.loadtxt("test/test_data/" + modelname + "_P_unsafe.dat")
            sim_dat_base = np.loadtxt("test/test_data/" + modelname + "_sim_dat_unsafe.dat")
            t_base = np.loadtxt("test/test_data/" + modelname + "_t_unsafe.dat")

            #np.testing.assert_almost_equal(P, P_base)
            #np.testing.assert_almost_equal(sim_dat,sim_dat_base)
            #np.testing.assert_almost_equal(t, t_base)


if __name__ == '__main__':
    unittest.main()