"""
Potential Surface Analysis Demo for Vascular Network Simulations

This script performs a “potential surface” exploration by varying a key model
parameter (a Windkessel resistance) and measuring its effect on the simulation
output. We compute both the loss (normalized pressure deviation) and its
gradient with respect to the parameter scale, over a specified range.

Workflow
--------
1. **Configure** the simulation environment and load network & blood properties.
2. **Run** a baseline (unsafe) simulation to obtain reference pressure data.
3. **Define** a loss function that rescales one parameter in the cardiac model,
   reruns the simulation, and measures deviation from the baseline.
4. **Determine** the subset of parameter scales to evaluate based on command‐line arguments (for distributed runs)
5. **Vectorize** and JIT‐compile both the loss and its gradient over a grid of
   parameter scaling factors.
6. **Plot** the resulting potential surface (loss vs. scale) and its gradient.

Usage
-----
    python potential_surface_demo.py [slice_index num_slices]

- *slice_index* (int, optional): zero‐based index of the current task slice.
- *num_slices* (int, optional): number of equally‐sized slices to split the
  TOTAL_NUM_POINTS across for distributed evaluation.

Dependencies
------------
- JAX (jax, jax.numpy, jacfwd, jit, vmap)
- Matplotlib
- src.model.config_simulation, simulation_loop_unsafe

Constants
---------
CONFIG_FILENAME : str
    Path to the YAML file defining the bifurcation network and solver settings.
RESULTS_FOLDER : str
    Directory to save any outputs (if expanded in future).
NUM_TIME_STEPS : int
    Number of time‐steps to run in the simulation loop.
TOTAL_NUM_POINTS : int
    Number of parameter scales to sample in the potential surface.
"""

import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, vmap
import matplotlib.pyplot as plt
import scienceplots  # prettier scientific plot styles

# -----------------------------------------------------------------------------
# Project setup: ensure we can import the simulation code and enable 64‐bit JAX
# -----------------------------------------------------------------------------
sys.path.insert(0, sys.path[0] + "/..")
jax.config.update("jax_enable_x64", True)
os.chdir(os.path.dirname(__file__) + "/..")

from src.model import config_simulation, simulation_loop_unsafe

# Use the "science" style for publication‐ready figures
plt.style.use("science")

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
CONFIG_FILE = "test/bifurcation/bifurcation.yml"  # YAML defining network & solver
NUM_TIME_STEPS = 50_000  # total steps per simulation run
TOTAL_NUM_POINTS = 10  # number of scale factors to sample
SCALE_MIN, SCALE_MAX = 0.8, 1.2  # range of resistance scales
RESULTS_DIR = "results/potential_surface"  # where to save outputs

# -----------------------------------------------------------------------------
# 1) Load and configure the simulation environment
# -----------------------------------------------------------------------------
(
    N,
    B,
    J,
    sim_dat,
    sim_dat_aux,
    sim_dat_const,
    sim_dat_const_aux,
    timepoints,
    conv_tol,
    Ccfl,
    input_data,
    rho,
    masks,
    strides,
    edges,
    vessel_names,
    cardiac_period,
) = config_simulation(CONFIG_FILE)

# For numerical stability, pick a moderate CFL number
Ccfl = 0.5

# JIT‐compile the "unsafe" simulation loop; treat N, B, and time‐slice as static
SIM_LOOP = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)

# -----------------------------------------------------------------------------
# 2) Run baseline simulation to get reference pressures
# -----------------------------------------------------------------------------
_, _, P_ref = SIM_LOOP(
    N,
    B,
    sim_dat,
    sim_dat_aux,
    sim_dat_const,
    sim_dat_const_aux,
    Ccfl,
    input_data,
    rho,
    masks,
    strides,
    edges,
    NUM_TIME_STEPS,
)


# -----------------------------------------------------------------------------
# 3) Define our loss function
# -----------------------------------------------------------------------------
# We'll scale the Windkessel resistance at (vessel_index, param_index)
VESSEL_INDEX = 1
PARAM_INDEX = 4
R_base = sim_dat_const_aux[VESSEL_INDEX, PARAM_INDEX]

# Convert numpy arrays to JAX arrays for loss wrapper
sim_dat = jnp.asarray(sim_dat)
sim_dat_aux = jnp.asarray(sim_dat_aux)
sim_dat_const = jnp.asarray(sim_dat_const)
sim_dat_const_aux = jnp.asarray(sim_dat_const_aux)
Ccfl = jnp.float64(Ccfl)
input_data = jnp.asarray(input_data)
rho = jnp.asarray(rho)
masks = jnp.asarray(masks)
strides = jnp.asarray(strides)
edges = jnp.asarray(edges)


def loss(scale: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the normalized mean‐squared pressure error when scaling R.

    Parameters
    ----------
    scale : jnp.ndarray, shape=()
        Multiplicative factor applied to the baseline resistance.

    Returns
    -------
    jnp.ndarray, shape=()
        Mean of (‖P_scaled − P_ref‖² / ‖P_ref‖²) across all time points and vessels.
    """
    # 1) Modify the auxiliary constant array for the chosen resistance
    R_new = scale * R_base
    sim_const_aux_new = sim_dat_const_aux.at[VESSEL_INDEX, PARAM_INDEX].set(R_new)

    # 2) Rerun the simulation with the scaled resistance
    _, _, P_mod = simulation_loop_unsafe(
        N,
        B,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_const_aux_new,
        Ccfl,
        input_data,
        rho,
        masks,
        strides,
        edges,
        NUM_TIME_STEPS,
    )

    # 3) Compute normalized L2 error: mean of sum((P_mod − P_ref)²) / sum(P_ref²)
    numerator = jnp.sum((P_mod - P_ref) ** 2, axis=0)
    denominator = jnp.sum(P_ref**2, axis=0)
    return jnp.mean(numerator / denominator)


# -----------------------------------------------------------------------------
# 4) Determine which subset of scales to evaluate (for distributed runs)
# -----------------------------------------------------------------------------
all_scales = jnp.linspace(SCALE_MIN, SCALE_MAX, TOTAL_NUM_POINTS)

if len(sys.argv) >= 3:
    idx = int(sys.argv[1])
    n_slices = int(sys.argv[2])
    per_slice = TOTAL_NUM_POINTS // n_slices
    start = idx * per_slice
    end = start + per_slice
    scales = all_scales[start:end]
    SAVEFILENAME = "potential_surface_" + str(idx) + ".eps"
else:
    scales = all_scales
    SAVEFILENAME = "potential_surface.eps"

# -----------------------------------------------------------------------------
# 5) Vectorized & JIT‐compiled gradient and loss evaluations
# -----------------------------------------------------------------------------
print("Computing gradients d(loss)/d(scale)...")
grad_fn = jit(vmap(jacfwd(loss)))
grad_vals = grad_fn(scales)
jax.block_until_ready(grad_vals)
print("Gradients computed.")

print("Computing loss values...")
loss_fn = jit(vmap(loss))
loss_vals = loss_fn(scales)
jax.block_until_ready(loss_vals)
print("Loss values computed.")

# -----------------------------------------------------------------------------
# 6) Plot the potential surface and its derivative
# -----------------------------------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Loss vs. scale on primary y‐axis
ax1.plot(scales, loss_vals, marker="o", label="Loss", color="C0")
ax1.set_xlabel("Resistance scale factor")
ax1.set_ylabel("Normalized Loss")

# Gradient vs. scale on secondary y‐axis
ax2.plot(scales, grad_vals, linestyle="--", label="Gradient", color="C1")
ax2.set_ylabel("d(Loss)/d(Scale)")

plt.title("Potential Surface Analysis for Windkessel Resistance")
fig.legend(loc="upper right")
plt.tight_layout()

# Save as high‐resolution PNG
out_path = os.path.join(RESULTS_DIR, SAVEFILENAME)
fig.savefig(out_path)
print(f"Figure saved to {out_path}")

plt.show()
plt.close(fig)
