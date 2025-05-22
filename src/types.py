"""
Type aliases for JAX arrays and Python scalars in the vascular network simulator.

This module centralizes all shape‐annotated type aliases using jaxtyping and
native Python types, covering:

- Simulation state arrays (SimDat, SimDatAux, SimDatConst, etc.)
- Time and pressure snapshot arrays
- Scalars, fixed‐length vectors, and index types
- Junction solver return shapes
- Input, mask, stride, and graph connectivity structures
- Composite tuple types for simulation step arguments

Using these aliases ensures consistency, readability, and static shape checking
across the codebase.
"""

import numpy as np
from jaxtyping import Array, Float, Integer, Bool
from typing import Callable, ParamSpec, TypeVar, TypeAlias

# -----------------------------------------------------------------------------
# Core simulation state arrays
# -----------------------------------------------------------------------------
# TODO: remove use of np.ndarray and use jax types instead or make them more concise by using NDarray and e.g. np.float64
"""State variables [u, q, a, c, p] stacked for K total nodes (including ghost cells)."""
SimDat: TypeAlias = Float[Array, "5 K"]  # type: ignore

"""NumPy array equivalent for SimDat (used in configuration/setup)."""
StaticSimDat: TypeAlias = np.ndarray

"""1D array of length K for a single state variable (e.g., velocity or area)."""
SimDatSingle: TypeAlias = Float[Array, " K"]  # type: ignore

"""NumPy array equivalent for SimDatSingle (used in configuration/setup)."""
StaticSimDatSingle: TypeAlias = np.ndarray
"""1D interior array excluding one ghost cell at each end (length K-1)."""
SimDatSingleReduced: TypeAlias = Float[Array, " K-1"]  # type: ignore

"""2×K array for paired variables (e.g., [u, q] combined)."""
SimDatDouble: TypeAlias = Float[Array, "2 K"]  # type: ignore

# -----------------------------------------------------------------------------
# Auxiliary and constant parameters
# -----------------------------------------------------------------------------
"""Auxiliary per‐vessel state variables, shape (N vessels, 3 parameters)."""
SimDatAux: TypeAlias = Float[Array, "N 3"]  # type: ignore

"""1D array of length N for storing per‐vessel norms or diagnostics."""
SimDatAuxSingle: TypeAlias = Float[Array, " N"]  # type: ignore

"""NumPy array equivalent for SimDatAux."""
StaticSimDatAux: TypeAlias = np.ndarray

"""Per‐node constant parameters, shape (7 parameters × K nodes)."""
SimDatConst: TypeAlias = Float[Array, "7 K"]  # type: ignore

"""NumPy array equivalent for SimDatConst."""
StaticSimDatConst: TypeAlias = np.ndarray

"""Per‐vessel constant parameters, shape (N vessels × 7 parameters)."""
SimDatConstAux: TypeAlias = Float[Array, "N 7"]  # type: ignore

"""NumPy array equivalent for SimDatConstAux."""
StaticSimDatConstAux: TypeAlias = np.ndarray

# -----------------------------------------------------------------------------
# Time, pressure, and snapshot arrays
# -----------------------------------------------------------------------------
"""1D array of J timepoints for pressure sampling over one cardiac cycle."""
Timepoints: TypeAlias = Float[Array, " J"]  # type: ignore

"""NumPy array equivalent for Timepoints."""
StaticTimepoints: TypeAlias = np.ndarray

"""Recorded times at I snapshots during the simulation."""
TimepointsReturn: TypeAlias = Float[Array, " I"]

"""NumPy array equivalent for TimepointsReturn."""
StaticTimepointsReturn: TypeAlias = np.ndarray

# TODO: implicitly implement M=5*N
"""Pressure snapshots: I timepoints × M = 5*K sampled values."""
PressureReturn: TypeAlias = Float[Array, "I M"]

"""Pressure time series for a single spatial sample across I timepoints."""
PressureReturnSingle: TypeAlias = Float[Array, " I"]

"""NumPy array equivalent for PressureReturn."""
StaticPressureReturn: TypeAlias = np.ndarray

# -----------------------------------------------------------------------------
# Scalar and small fixed‐length vector types
# -----------------------------------------------------------------------------
"""Scalar floating‐point (JAX array of shape ())."""
ScalarFloat: TypeAlias = Float[Array, ""]

"""Scalar integer (JAX array of shape ())."""
ScalarInt: TypeAlias = Integer[Array, ""]

"""Native Python float or NumPy scalar."""
StaticScalarFloat: TypeAlias = float | np.floating

"""Native Python int or NumPy scalar."""
StaticScalarInt: TypeAlias = int | np.int64

"""Scalar boolean (JAX array of shape ())."""
ScalarBool: TypeAlias = Bool[Array, ""]

"""Native Python boolean."""
StaticBool: TypeAlias = bool

"""Length‐2 float vector (e.g., [u1, u2])."""
PairFloat: TypeAlias = Float[Array, "2"]

"""Length‐3 float vector (e.g., junction variables for three vessels)."""
TripleFloat: TypeAlias = Float[Array, "3"]

"""Length‐4 float vector."""
QuadFloat: TypeAlias = Float[Array, "4"]

"""Length‐6 float vector."""
HexaFloat: TypeAlias = Float[Array, "6"]


# -----------------------------------------------------------------------------
# Jacobian and solver return shapes
# -----------------------------------------------------------------------------
"""4×4 Jacobian matrix for conjunction problems."""
SmallJacobian: TypeAlias = Float[Array, "4 4"]

"""6×6 Jacobian matrix for bifurcation/anastomosis problems."""
LargeJacobian: TypeAlias = Float[Array, "6 6"]

"""Return array of length 10 from conjunction solver [u1,u2,q1,q2,a1,a2,c1,c2,p1,p2]."""
DoubleJunctionReturn: TypeAlias = Float[Array, "10"]
"""Return array of length 15 from bifurcation/anastomosis solver [u1,u2,u3, q1,q2,q3, a1,a2,a3, c1,c2,c3, p1,p2,p3]."""
TripleJunctionReturn: TypeAlias = Float[Array, "15"]

# -----------------------------------------------------------------------------
# I/O, mask, stride, and graph connectivity types
# -----------------------------------------------------------------------------
# TODO: implicitly implement S=2*N
"""External boundary data array, shape (S signals × H history length)."""
InputData: TypeAlias = Float[Array, "S H"]  # type: ignore

"""NumPy array equivalent for InputData."""
StaticInputData: TypeAlias = np.ndarray

"""NumPy array for single‐signal input data."""
StaticInputDataSingle: TypeAlias = np.ndarray

"""Boolean masks (2 rows) for domain interior vs. boundaries over K nodes."""
Masks: TypeAlias = Integer[Array, "2 K"]  # type: ignore

"""Masks including two ghost cells on each side (total K+4)."""
MasksPadded: TypeAlias = Integer[Array, "2 K+4"]  # type: ignore

"""NumPy array equivalent for Masks."""
StaticMasks: TypeAlias = np.ndarray

"""Sample index strides per vessel: [start, end, offset1, offset2, offset3]."""
Strides: TypeAlias = Integer[Array, "N 5"]  # type: ignore

"""Reduced strides per vessel: [start, end] for interior updates."""
StridesReduced: TypeAlias = Integer[Array, "N 2"]  # type: ignore

"""NumPy array equivalent for Strides."""
StaticStrides: TypeAlias = np.ndarray

"""Connectivity matrix encoding graph topology and branch indices."""
Edges: TypeAlias = Integer[Array, "N 10"]  # type: ignore

"""NumPy array equivalent for Edges."""
StaticEdges: TypeAlias = np.ndarray

"""Single string type (e.g., file path or vessel name)."""
String: TypeAlias = str

"""List of strings (e.g., vessel name list)."""
Strings: TypeAlias = list[str]

"""Generic dict type."""
Dict: TypeAlias = dict

"""List of dicts (e.g., JSON/YAML objects)."""
Dicts: TypeAlias = list[dict]

"""Generic callable type alias (e.g., for functions that return a scalar or take scalar inputs)."""
P = ParamSpec("P")
R = TypeVar("R")
ScalarCallable = Callable[P, R]

# -----------------------------------------------------------------------------
# Composite tuple types for simulation step arguments
# -----------------------------------------------------------------------------
"""
Arguments passed through each safe simulation step:
(sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux,
 t, snapshot_idx, timepoints, passed_cycles,
 dt, p_t, p_l, t_t, conv_tol, ccfl,
 edges, input_data, rho)
"""
SimulationStepArgs = tuple[
    SimDat,
    SimDatAux,
    SimDatConst,
    SimDatConstAux,
    ScalarFloat,
    ScalarInt,
    Timepoints,
    ScalarInt,
    ScalarFloat,
    PressureReturn,
    PressureReturn,
    TimepointsReturn,
    ScalarFloat,
    ScalarFloat,
    Edges,
    InputData,
    ScalarFloat,
]

"""
Arguments passed through each unsafe simulation step:
(sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux,
 dt, t, t_t, edges, input_data, rho, p_t)
"""
SimulationStepArgsUnsafe: TypeAlias = tuple[
    SimDat,
    SimDatAux,
    SimDatConst,
    SimDatConstAux,
    ScalarFloat,
    ScalarFloat,
    TimepointsReturn,
    Edges,
    InputData,
    ScalarFloat,
    PressureReturn,
]
