"""
This module provides a function to solve nonlinear equations using the Newton-Raphson method.

It includes the following function:
- `newtonRaphson`: Solves the system of nonlinear equations by iteratively updating the solution using the Jacobian matrix and function evaluations.

The module makes use of the following imported utilities:
- `jax.numpy` for numerical operations and array handling.
- `jaxtyping` and `typeguard` for type checking and ensuring type safety in the functions.
"""

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def newtonRaphson(
    fun_f: Callable,
    j: Float[Array, ["..."]],
    u: Float[Array, ["..."]],
    a0s: Float[Array, ["..."]],
    betas: Float[Array, ["..."]],
) -> Float[Array, ["..."]]:
    """
    Solves a system of nonlinear equations using the Newton-Raphson method.

    Parameters:
    fun_f (Callable): Function that computes the system of equations.
    j (Float[Array, ["..."]]): Jacobian matrix of the system.
    u (Float[Array, ["..."]]): Initial guess for the solution vector.
    a0s (Float[Array, ["..."]]): Reference cross-sectional areas for the vessels.
    betas (Float[Array, ["..."]]): Stiffness coefficients for the vessels.

    Returns:
    Float[Array, ["..."]]: Updated solution vector after applying the Newton-Raphson method.
    """

    f = fun_f(u, a0s, betas)

    du = jnp.linalg.solve(j, -f)
    return u + du
