"""
Unit tests for vascular network simulation models.

This module verifies that the "safe" and "unsafe" simulation runners produce
bitwise‐identical outputs to previously generated baseline data. It also
reports progress and timing information for each model to aid diagnostics.

Test cases
----------
TestModels.test_models
    Runs `run_simulation` (with convergence checks) for each example network,
    compares the resulting (sim_dat, t, P) arrays to baseline files, and
    prints timing info.

TestModels.test_models_unsafe
    Runs `run_simulation_unsafe` (without convergence checks) for each network,
    compares outputs to baseline files, and prints timing info.

Usage
-----
    python path/to/test_models.py [path/to/test_data]

Baseline data directory: `test/test_data/`
"""

import os
import sys
import time
import unittest

import jax
import numpy as np

# -----------------------------------------------------------------------------
# Project setup: ensure we can import the simulation code and enable 64‐bit JAX
# -----------------------------------------------------------------------------
sys.path.insert(0, sys.path[0] + "/..")
jax.config.update("jax_enable_x64", True)

# Change working directory to base directory of repository
os.chdir(os.path.dirname(__file__) + "/..")
from src.model import run_simulation, run_simulation_unsafe


class TestModels(unittest.TestCase):
    """
    Test suite for validating simulation outputs against stored baseline data.
    """

    def setUp(self):
        """
        Common setup for each test: define model list and baseline data folder.
        """
        self.models = [
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
        if len(sys.argv) > 1:
            self.baseline_dir = sys.argv[1]
        else:
            self.baseline_dir = "test/test_data"
        # Ensure baseline directory exists
        self.assertTrue(
            os.path.isdir(self.baseline_dir),
            f"Baseline data directory not found: {self.baseline_dir}",
        )

    def test_models(self):
        """
        For each model, run the 'safe' simulation (with convergence checks),
        compare outputs to baseline, and report timing.
        """
        print("\n=== Testing SAFE simulations ===")
        for model in self.models:
            with self.subTest(model=model):
                config_file = f"test/{model}/{model}.yml"
                print(f"Running safe simulation for '{model}'...", end="", flush=True)
                start = time.perf_counter()
                sim_dat, t, P = run_simulation(config_file)
                duration = time.perf_counter() - start
                print(f" done in {duration:.2f} s")

                # Load baseline arrays
                P_ref = np.loadtxt(f"{self.baseline_dir}/{model}_P.dat")
                sim_dat_ref = np.loadtxt(f"{self.baseline_dir}/{model}_sim_dat.dat")
                t_ref = np.loadtxt(f"{self.baseline_dir}/{model}_t.dat")

                # Assertions: use almost_equal to allow floating‐point tolerance
                np.testing.assert_almost_equal(
                    P, P_ref, err_msg=f"P mismatch for {model}"
                )
                np.testing.assert_almost_equal(
                    sim_dat, sim_dat_ref, err_msg=f"sim_dat mismatch for {model}"
                )
                np.testing.assert_almost_equal(
                    t, t_ref, err_msg=f"t mismatch for {model}"
                )

    def test_models_unsafe(self):
        """
        For each model, run the 'unsafe' simulation (no convergence checks),
        compare outputs to baseline, and report timing.
        """
        print("\n=== Testing UNSAFE simulations ===")
        for model in self.models:
            with self.subTest(model=model):
                config_file = f"test/{model}/{model}.yml"
                print(f"Running unsafe simulation for '{model}'...", end="", flush=True)
                start = time.perf_counter()
                sim_dat, t, P = run_simulation_unsafe(config_file)
                duration = time.perf_counter() - start
                print(f" done in {duration:.2f} s")

                # Load baseline arrays for unsafe runs
                P_ref = np.loadtxt(f"{self.baseline_dir}/{model}_P_unsafe.dat")
                sim_dat_ref = np.loadtxt(
                    f"{self.baseline_dir}/{model}_sim_dat_unsafe.dat"
                )
                t_ref = np.loadtxt(f"{self.baseline_dir}/{model}_t_unsafe.dat")

                np.testing.assert_almost_equal(
                    P, P_ref, err_msg=f"P (unsafe) mismatch for {model}"
                )
                np.testing.assert_almost_equal(
                    sim_dat,
                    sim_dat_ref,
                    err_msg=f"sim_dat (unsafe) mismatch for {model}",
                )
                np.testing.assert_almost_equal(
                    t, t_ref, err_msg=f"t (unsafe) mismatch for {model}"
                )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(
            "Usage: python path/to/test_models.py [path/to/test_data]\n"
            "Baseline data directory defaults to 'test/test_data/' if not specified."
        )
        sys.exit(0)
    elif len(sys.argv) > 1:
        unittest.main(argv=sys.argv[1:])
    else:
        unittest.main()
