"""
This module provides a function to save temporary pressure data in a vascular model using JAX.

It includes a function to:
- Save temporary pressure data at specified strides (`save_temp_data`).

The module makes use of the following imported utilities:
- `jax.numpy` for numerical operations and array handling.
- `jax.lax` for control flow operations.
- `jaxtyping` and `beartype` for type checking and ensuring type safety in the functions.
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
    Saves temporary pressure data at specified strides.

    Parameters:
    n (int): Number of vessels.
    strides (Integer[Array, "..."]): Array containing the stride information for each vessel.
    p (Float[Array, "..."]): Array containing the pressure data.

    Returns:
    Float[Array, ""]: Array of temporary pressure data.
    """
    p_t = jnp.zeros(5 * n)

    def assign_pressure(i, args):
        p_t, strides = args
        start = strides[i, 0]
        end = strides[i, 1]
        p_t = p_t.at[i * 5].set(p[start])
        p_t = p_t.at[i * 5 + 1].set(p[start + strides[i, 2]])
        p_t = p_t.at[i * 5 + 2].set(p[start + strides[i, 3]])
        p_t = p_t.at[i * 5 + 3].set(p[start + strides[i, 4]])
        p_t = p_t.at[i * 5 + 4].set(p[end - 1])
        return (p_t, strides)

    p_t, _ = lax.fori_loop(0, n, assign_pressure, (p_t, strides))
    return p_t
