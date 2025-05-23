"""
IO utilities for saving pressure snapshots in vascular network simulations using JAX.

This module provides functionality to extract and store temporary pressure
values at selected spatial locations (strides) along each vessel. It is used
to create compact representations of the full pressure field for diagnostics,
checkpointing, or convergence checks.

Functions
---------
save_temp_data(n, strides, p)
    Sample and return pressure values at five key points along each of the `n`
    vessels, according to the provided stride indices.

Dependencies
------------
- `jax.numpy` (as jnp)        : Array operations and numerical routines.
- `jax.lax`                   : JAX-compatible control-flow primitives.
- `jaxtyping.jaxtyped`        : Static type annotations for JAX arrays.
- `beartype.beartype`         : Runtime type checking of function signatures.

Type Aliases (from `src.types`)
-------------------------------
- `StaticScalarInt`    : Integer type for fixed-size loop bounds.
- `Strides`            : Integer array of shape (n, 5), where each row
                        encodes [start, end, offset1, offset2, offset3].
- `SimDatSingle`       : 1D float array of length n, used internally for norms.
- `PressureReturnSingle`: 1D float array of length 5*n, holding sampled pressures.
"""

import jax.numpy as jnp
from jax import lax
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

from src.types import (
    PressureReturnSingle,
    SimDatSingle,
    StaticScalarInt,
    Strides,
)


@jaxtyped(typechecker=typechecker)
def save_temp_data(
    n: StaticScalarInt, strides: Strides, p: SimDatSingle
) -> PressureReturnSingle:
    """
    Sample pressure at five designated points along each vessel.

    For each vessel i in range(n), this function:
      1. Reads the starting pressure at index `start = strides[i,0]`.
      2. Reads three intermediate pressures at offsets
         `start + strides[i,2]`, `start + strides[i,3]`, and
         `start + strides[i,4]`.
      3. Reads the ending pressure at index `end = strides[i,1] - 1`.
    The sampled pressures are concatenated into a single 1D array `p_t` of
    length 5*n, with vessel i’s values stored at positions
    `[5*i, 5*i+1, 5*i+2, 5*i+3, 5*i+4]`.

    Parameters
    ----------
    n : StaticScalarInt
        Total number of vessels (rows in `strides`).
    strides : Strides
        Integer array of shape (n, 5). Each row encodes:
          - strides[i,0]: start index for vessel i
          - strides[i,1]: end index (exclusive) for vessel i
          - strides[i,2..4]: offsets for the three intermediate sample points
    p : SimDatSingle
        1D array of raw pressure data for all spatial nodes; length must
        exceed max(strides[:,1]).

    Returns
    -------
    PressureReturnSingle
        1D array of sampled pressures with shape (5*n,). The entries are
        ordered by vessel, then by sample location.

    Example
    -------
    >>> # For 2 vessels, with strides = [[0, 10, 2, 5, 8], [10, 20, 3, 6, 9]]
    >>> # and p a length-20 array,
    >>> # save_temp_data(2, strides, p) returns:
    >>> # [p[0], p[2], p[5], p[8], p[9],  p[10], p[13], p[16], p[19], p[19]]

    """
    # Initialize output array: 5 samples per vessel
    p_t = jnp.zeros(5 * n)

    def assign_pressure(i: int, args):
        p_t_acc, strides_acc = args
        # Unpack stride information for vessel i
        start = strides_acc[i, 0]
        end = strides_acc[i, 1]
        # Sample pressures at start, three intermediates, and end
        p_t_acc = p_t_acc.at[i * 5 + 0].set(p[start])
        p_t_acc = p_t_acc.at[i * 5 + 1].set(p[start + strides_acc[i, 2]])
        p_t_acc = p_t_acc.at[i * 5 + 2].set(p[start + strides_acc[i, 3]])
        p_t_acc = p_t_acc.at[i * 5 + 3].set(p[start + strides_acc[i, 4]])
        p_t_acc = p_t_acc.at[i * 5 + 4].set(p[end - 1])
        return (p_t_acc, strides_acc)

    # Loop over vessels in a JAX-friendly way
    p_t_final, _ = lax.fori_loop(0, n, assign_pressure, (p_t, strides))
    return p_t_final
