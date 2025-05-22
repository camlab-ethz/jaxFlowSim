"""
Newton–Raphson solver for nonlinear junction problems in vascular networks using JAX.

This module provides a single-step iterative solver implementing the Newton–Raphson
method for systems of nonlinear equations that arise in vascular junction models
(e.g., bifurcation, anastomosis, conjunction).

Main function
-------------
newton_raphson
    Perform one Newton–Raphson update: evaluate residuals, solve the linear
    correction, and return the updated solution.

Dependencies
------------
- jax.numpy (jnp): Array operations and linear algebra routines.
- jaxtyping.jaxtyped: Static type annotations for JAX arrays.
- beartype.beartype: Runtime type checking of function signatures.
- src.types: Domain-specific type aliases for state and Jacobian shapes.
"""

import jax.numpy as jnp
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

from src.types import (
    HexaFloat,
    QuadFloat,
    SmallJacobian,
    LargeJacobian,
    PairFloat,
    TripleFloat,
    ScalarCallable,
)


@jaxtyped(typechecker=typechecker)
def newton_raphson(
    fun_f: ScalarCallable[
        [QuadFloat | HexaFloat, PairFloat | TripleFloat, PairFloat | TripleFloat],
        QuadFloat | HexaFloat,
    ],
    jacobian: SmallJacobian | LargeJacobian,
    u: QuadFloat | HexaFloat,
    a0s: PairFloat | TripleFloat,
    betas: PairFloat | TripleFloat,
) -> QuadFloat | HexaFloat:
    """
    Perform a single Newton–Raphson iteration for a system of nonlinear equations.

    This function computes the residual vector f = fun_f(u, a0s, betas),
    solves the linear system J · Δu = –f for the update step Δu, and returns the
    next estimate u + Δu.

    Parameters
    ----------
    fun_f : Callable[[u, a0s, betas], f]
        Residual function evaluating the system at `u`. Should return an array of
        the same shape as `u`.
    jacobian : SmallJacobian or LargeJacobian
        Jacobian matrix J = ∂f/∂u evaluated at the current `u`. Shape must be
        (m, m) where m = len(u).
    u : QuadFloat or HexaFloat
        Current solution estimate vector of length m.
    a0s : PairFloat or TripleFloat
        Reference cross-sectional areas passed to `fun_f`.
    betas : PairFloat or TripleFloat
        Stiffness coefficients passed to `fun_f`.

    Returns
    -------
    QuadFloat or HexaFloat
        Updated solution estimate after applying the Newton step.

    Raises
    ------
    jnp.linalg.LinAlgError
        If the Jacobian matrix is singular or the linear solve fails.

    Notes
    -----
    - This function performs exactly one Newton–Raphson update. To converge
      to a root, call this function in a loop until the residual norm falls
      below a desired tolerance.
    - It is the caller's responsibility to ensure that `jacobian` corresponds
      to the derivative of `fun_f` at the provided `u`.
    """
    # 1) Evaluate the residual vector f(u)
    f = fun_f(u, a0s, betas)
    # 2) Solve the linear system J · Δu = –f for the Newton update Δu
    du = jnp.linalg.solve(jacobian, -f)
    # 3) Return the new estimate u + Δu
    return u + du
