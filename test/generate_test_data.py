"""
Test Data Generator for Vascular Network Simulations

This script runs a suite of example vascular network models (both with and
without convergence checks) to generate reference output data for testing
and validation. For each model it:

1. Loads the simulation configuration.
2. Runs the "safe" simulation (`run_simulation`) and records wall time.
3. Saves the resulting state, timepoints, and pressure arrays to disk.
4. Runs the "unsafe" simulation (`run_simulation_unsafe`) and records wall time.
5. Saves those outputs as well.
6. Prints progress and timing information to keep the user informed.

Usage:
    python generate_test_data.py

Dependencies:
- JAX (jax, jax.numpy)
- NumPy
- src.model.run_simulation
- src.model.run_simulation_unsafe
"""

import os
import sys
import time

import jax
import numpy as np

# -----------------------------------------------------------------------------
# Project setup: enable 64-bit precision and ensure `src` is on the path
# -----------------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, sys.path[0] + "/..")
from src.model import run_simulation, run_simulation_unsafe

# Change working directory to project root (one level up from this script)
os.chdir(os.path.dirname(__file__) + "/..")


# -----------------------------------------------------------------------------
# Output directory for test data
# -----------------------------------------------------------------------------
TEST_DATA_DIR = "test/test_data"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# List of example model names (each corresponds to test/{name}/{name}.yml)
# -----------------------------------------------------------------------------
MODEL_NAMES = [
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


# -----------------------------------------------------------------------------
# Helper function to save arrays with informative filenames
# -----------------------------------------------------------------------------
def save_array(model: str, suffix: str, arr: np.ndarray):
    """
    Save a NumPy array to disk under TEST_DATA_DIR with naming convention:
        {model}_{suffix}.dat
    """
    filename = f"{model}_{suffix}.dat"
    path = os.path.join(TEST_DATA_DIR, filename)
    np.savetxt(path, arr)
    print(f"    → Saved '{filename}' ({arr.shape}, dtype={arr.dtype})")


# -----------------------------------------------------------------------------
# Main loop: run safe and unsafe simulations for each model
# -----------------------------------------------------------------------------
total_models = len(MODEL_NAMES)
print(f"Starting test data generation for {total_models} models...\n")

for idx, model in enumerate(MODEL_NAMES, start=1):
    config_file = f"test/{model}/{model}.yml"
    print(f"[{idx}/{total_models}] Model '{model}'")

    # -- Safe simulation --
    print("  Running safe simulation...", end="", flush=True)
    start_time = time.perf_counter()
    sim_dat, t, P = run_simulation(config_file)
    elapsed = time.perf_counter() - start_time
    print(f" done in {elapsed:.2f}s")

    # Save safe outputs
    save_array(model, "sim_dat", np.array(sim_dat))
    save_array(model, "t", np.array(t))
    save_array(model, "P", np.array(P))

    # -- Unsafe simulation --
    print("  Running unsafe simulation...", end="", flush=True)
    start_time = time.perf_counter()
    sim_dat_u, t_u, P_u = run_simulation_unsafe(config_file)
    elapsed = time.perf_counter() - start_time
    print(f" done in {elapsed:.2f}s")

    # Save unsafe outputs
    save_array(model, "sim_dat_unsafe", np.array(sim_dat_u))
    save_array(model, "t_unsafe", np.array(t_u))
    save_array(model, "P_unsafe", np.array(P_u))

    print("")  # blank line between models

print("All test data generated successfully.")
