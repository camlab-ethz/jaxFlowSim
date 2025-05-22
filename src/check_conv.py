"""
Convergence diagnostics for vascular network simulations using JAX.

This module provides utilities to measure and verify convergence of iterative
pressure solutions. It computes pointwise L₂ norms of the difference between
target and learned pressure fields, extracts the worst-case error, reports it
in physiological units (mmHg), and checks against a user-defined tolerance.

Available functions
-------------------
calc_norms(n, p_target, p_learned)
    Compute an array of L₂ norms of pressure differences at each spatial point.
compute_conv_error(n, p_target, p_learned)
    Return the maximum L₂ norm as the overall convergence error.
print_conv_error(err)
    Convert a convergence error from Pascals to mmHg and display it.
check_conv(err, conv_tol)
    Determine whether the error (in mmHg) meets the convergence criterion.

Dependencies
------------
- jax.lax               Control-flow primitives for efficient looping in JAX.
- jax.numpy (jnp)       Array operations and numerical routines.
- jaxtyping.jaxtyped     Static type annotations for JAX arrays.
- beartype.beartype      Runtime type enforcement of function signatures.

Type aliases (from src.types)
------------------------------
- StaticScalarInt  : Integer index for loop bounds.
- PressureReturn   : 2D array of pressure values, shape (num_signals, num_points*5).
- SimDatAuxSingle  : 1D array to store computed norms, length = num_points.
- SimDatSingle     : Alias for a single-sample data array (same as SimDatAuxSingle).
- ScalarFloat      : Scalar floating-point value.
- ScalarBool       : Scalar boolean value.
"""

from jax import lax
import jax.numpy as jnp
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

from src.types import (
    PressureReturn,
    SimDatAuxSingle,
    SimDatSingle,
    StaticScalarInt,
    ScalarFloat,
    ScalarBool,
)


@jaxtyped(typechecker=typechecker)
def calc_norms(
    n: StaticScalarInt, p_t: PressureReturn, p_l: PressureReturn
) -> SimDatAuxSingle:
    """
    Compute L₂ norms of pointwise pressure differences.

    For each of the `n` points, extract the pressure signal at the third
    channel (index 2) from both target and learned datasets, compute the
    difference vector, and calculate its Euclidean norm.

    Parameters
    ----------
    n : StaticScalarInt
        Number of spatial points to evaluate.
    p_target : PressureReturn
        Reference pressure data array of shape (num_signals, n*5).
    p_learned : PressureReturn
        Learned pressure data array (same shape as `p_target`).

    Returns
    -------
    SimDatAuxSingle
        1D array of length `n`, where each entry is the L₂ norm
        of the difference at that point.
    """
    # Initialize an array of zeros to hold the norms for each point
    norms: SimDatSingle = jnp.zeros(n)

    def body_fun(i, norms):
        # Extract the pressure trace for point i from both datasets
        err = p_l[:, i * 5 + 2] - p_t[:, i * 5 + 2]
        norms = norms.at[i].set(jnp.sqrt(jnp.sum(err**2)))
        # Compute Euclidean (L2) norm and store
        return norms

    # Loop over all points in a JAX-compatible manner
    norms = lax.fori_loop(0, n, body_fun, norms)
    return norms


@jaxtyped(typechecker=typechecker)
def compute_conv_error(
    n: StaticScalarInt, p_t: PressureReturn, p_l: PressureReturn
) -> ScalarFloat:
    """
    Determine the maximum convergence error across all spatial points.

    Computes the L₂ norm at each point via `calc_norms`, then returns
    the maximum norm as the overall convergence metric.

    Parameters
    ----------
    n : StaticScalarInt
        Number of spatial points to evaluate.
    p_target : PressureReturn
        Reference pressure data array.
    p_learned : PressureReturn
        Learned pressure data array.

    Returns
    -------
    ScalarFloat
        The maximum L₂ norm across all `n` points (in Pascals).
    """
    current_norms: PressureReturn = calc_norms(n, p_t, p_l)
    # Maximum norm indicates the worst-case pressure discrepancy
    maxnorm: ScalarFloat = jnp.max(current_norms)
    return maxnorm


@jaxtyped(typechecker=typechecker)
def print_conv_error(err: ScalarFloat):
    """
    Convert and display the convergence error in mmHg.

    Parameters
    ----------
    err : ScalarFloat
        Convergence error in Pascals.

    Returns
    -------
    None
    """
    mmhg = err / 133.332  # 1 mmHg ≈ 133.332 Pa
    # Print or log the error in physiological units
    # debug.print("error norm = {x} mmHg", x=err)


@jaxtyped(typechecker=typechecker)
def check_conv(err: ScalarFloat, conv_toll: ScalarFloat) -> ScalarBool:
    """
    Verify if the convergence error meets the specified tolerance.

    Converts the error from Pascals to mmHg, then checks whether it is
    less than or equal to the tolerance threshold.

    Parameters
    ----------
    err : ScalarFloat
        Convergence error in Pascals.
    conv_tol : ScalarFloat
        Allowed error tolerance in mmHg.

    Returns
    -------
    ScalarBool
        `True` if error (in mmHg) ≤ `conv_tol`, else `False`.
    """
    return err / 133.332 <= conv_toll
