"""
This module provides functions to calculate norms, compute convergence errors, print error messages, and check for convergence in a vascular network model using JAX.

It includes functions to:
- Calculate norms between two sets of pressure data (`calc_norms`).
- Compute the maximum convergence error (`compute_conv_error`).
- Print the convergence error (`print_conv_error`).
- Check if the convergence error is within a specified tolerance (`check_conv`).

The module makes use of the following imported utilities:
- `jax.lax` for control flow operations.
- `jax.numpy` for numerical operations and array handling.
- `jaxtyping` and `beartype` for type checking and ensuring type safety in the functions.
"""

from jax import lax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped, Bool
from beartype import beartype as typechecker

from src.types import StaticScalarInt, ScalarFloat


@jaxtyped(typechecker=typechecker)
def calc_norms(
    n: StaticScalarInt, p_t: Float[Array, "..."], p_l: Float[Array, "..."]
) -> Float[Array, "..."]:
    """
    Calculates the norms between two sets of pressure data.

    Parameters:
    n (Float[Array, ""]): Number of data points.
    p_t (Float[Array, ""]): Target pressure data.
    p_l (Float[Array, ""]): Learned pressure data.

    Returns:
    Float[Array, ""]: Array of norms for each data point.
    """
    norms = jnp.zeros(n)

    def body_fun(i, norms):
        err = p_l[:, i * 5 + 2] - p_t[:, i * 5 + 2]
        norms = norms.at[i].set(jnp.sqrt(jnp.sum(err**2)))
        return norms

    norms = lax.fori_loop(0, n, body_fun, norms)
    return norms


@jaxtyped(typechecker=typechecker)
def compute_conv_error(
    n: StaticScalarInt, p_t: Float[Array, "..."], p_l: Float[Array, "..."]
) -> ScalarFloat:
    """
    Computes the maximum convergence error between two sets of pressure data.

    Parameters:
    n (Float[Array, ""]): Number of data points.
    p_t (Float[ArrayTraced<ShapedArray(float64[100,5])>with<DynamicJaxprTrace(level=2/0)>, ""]): Target pressure data.
    p_l (Float[Array, ""]): Learned pressure data.

    Returns:
    Float[Array, ""]: Maximum convergence error.
    """
    current_norms = calc_norms(n, p_t, p_l)
    maxnorm = jnp.max(current_norms)
    return maxnorm


@jaxtyped(typechecker=typechecker)
def print_conv_error(err: Float[Array, ""]):
    """
    Prints the convergence error in mmHg.

    Parameters:
    err (Float[Array, ""]): Convergence error.

    Returns:
    None
    """
    err /= 133.332
    # debug.print("error norm = {x} mmHg", x=err)


@jaxtyped(typechecker=typechecker)
def check_conv(err: Float[Array, ""], conv_toll) -> Bool[Array, ""]:
    """
    Checks if the convergence error is within the specified tolerance.

    Parameters:
    err (Float[Array, ""]): Convergence error.
    conv_toll (Float[Array, ""]): Convergence tolerance.

    Returns:
    Float[Array, ""]: Boolean value indicating if the error is within the tolerance.
    """
    return err / 133.332 <= conv_toll
