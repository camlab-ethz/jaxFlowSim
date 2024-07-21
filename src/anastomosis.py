"""
This module provides functionality to solve the anastomosis problem in vascular networks using the Newton-Raphson method.

It includes functions to:
- Initialize and solve the anastomosis problem (`solve_anastomosis`).
- Calculate the Jacobian matrix specific to the anastomosis problem (`calculate_jacobian_anastomosis`).
- Evaluate the function vector for the current state of the anastomosis problem (`calculate_f_anastomosis`).
- Update the state of the system after solving the equations (`update_anastomosis`).

The module makes use of the following imported utilities:
- `newtonRaphson` from `src.newton` for solving the system of nonlinear equations.
- `pressure` and `waveSpeed` from `src.utils` for calculating pressure and wave speed in the vessels.
- `jax.numpy` for numerical operations and array handling.
- `jaxtyping` and `typeguard` for type checking and ensuring type safety in the functions.

Functions:
-----------
- solve_anastomosis(u1, u2, u3, a1, a2, a3, a01, a02, a03, beta1, beta2, beta3, gamma1, gamma2, gamma3, p_ext1, p_ext2, p_ext3): 
  Solves the anastomosis problem and returns the updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures.

- calculate_jacobian_anastomosis(u, k, a01, a02, a03, beta1, beta2, beta3):
  Calculates the Jacobian matrix needed for the Newton-Raphson method.

- calculate_f_anastomosis(u, a0s, betas):
  Computes the function vector for the current state of the anastomosis problem.

- update_anastomosis(u, a01, a02, a03, beta1, beta2, beta3, gamma1, gamma2, gamma3, p_ext1, p_ext2, p_ext3):
  Updates and returns the state of the system after solving the anastomosis equations.
"""

import jax.numpy as jnp
from src.newton import newtonRaphson
from src.utils import pressure, waveSpeed
from jaxtyping import Float, Array, jaxtyped, Scalar
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
def solve_anastomosis(
    u1: Scalar,
    u2: Scalar,
    u3: Scalar,
    a1: Scalar,
    a2: Scalar,
    a3: Scalar,
    a01: Scalar,
    a02: Scalar,
    a03: Scalar,
    beta1: Scalar,
    beta2: Scalar,
    beta3: Scalar,
    gamma1: Scalar,
    gamma2: Scalar,
    gamma3: Scalar,
    p_ext1: Scalar,
    p_ext2: Scalar,
    p_ext3: Scalar,
) -> tuple[
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
]:
    """
    Solves the anastomosis problem using the Newton-Raphson method.

    Parameters:
    u1, u2, u3 (Scalar): Initial velocities for vessels 1, 2, and 3.
    a1, a2, a3 (Scalar): Initial cross-sectional areas for vessels 1, 2, and 3.
    a01, a02, a03 (Scalar): Reference cross-sectional areas for vessels 1, 2, and 3.
    beta1, beta2, beta3 (Scalar): Stiffness coefficients for vessels 1, 2, and 3.
    gamma1, gamma2, gamma3 (Scalar): Admittance coefficients for vessels 1, 2, and 3.
    p_ext1, p_ext2, p_ext3 (Scalar): External pressures for vessels 1, 2, and 3.

    Returns:
    tuple: Updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures for vessels 1, 2, and 3.
    """
    u0 = jnp.array(
        (
            u1,
            u2,
            u3,
            jnp.sqrt(jnp.sqrt(a1)),
            jnp.sqrt(jnp.sqrt(a2)),
            jnp.sqrt(jnp.sqrt(a3)),
        )
    )

    k1 = jnp.sqrt(1.5 * gamma1)
    k2 = jnp.sqrt(1.5 * gamma2)
    k3 = jnp.sqrt(1.5 * gamma3)
    k = jnp.array([k1, k2, k3])

    j = calculate_jacobian_anastomosis(u0, k, a01, a02, a03, beta1, beta2, beta3)
    u = newtonRaphson(
        calculate_f_anastomosis,
        j,
        u0,
        jnp.array([a01, a02, a03]),
        jnp.array([beta1, beta2, beta3]),
    )

    return update_anastomosis(
        u,
        a01,
        a02,
        a03,
        beta1,
        beta2,
        beta3,
        gamma1,
        gamma2,
        gamma3,
        p_ext1,
        p_ext2,
        p_ext3,
    )


@jaxtyped(typechecker=typechecker)
def calculate_jacobian_anastomosis(
    u: Float[Array, " 6"],
    k: Float[Array, " 3"],
    a01: Scalar,
    a02: Scalar,
    a03: Scalar,
    beta1: Scalar,
    beta2: Scalar,
    beta3: Scalar,
) -> Float[Array, "6 6"]:
    """
    Calculates the Jacobian matrix for the anastomosis problem.

    Parameters:
    u (Float[Array, " 6"]): Current state vector.
    k (Float[Array, " 3"]): Array of constants derived from gamma values.
    a01, a02, a03 (Scalar): Reference cross-sectional areas for vessels 1, 2, and 3.
    beta1, beta2, beta3 (Scalar): Stiffness coefficients for vessels 1, 2, and 3.

    Returns:
    Float[Array, "6 6"]: Jacobian matrix.
    """
    u43 = u[3] * u[3] * u[3]
    u53 = u[4] * u[4] * u[4]
    u63 = u[5] * u[5] * u[5]

    j14 = 4.0 * k[0]
    j25 = 4.0 * k[1]
    j36 = -4.0 * k[2]

    j41 = u[3] * u43
    j42 = u[4] * u53
    j43 = -u[5] * u63
    j44 = 4.0 * u[0] * u43
    j45 = 4.0 * u[1] * u53
    j46 = -4.0 * u[2] * u63

    j54 = 2.0 * beta1 * u[3] * jnp.sqrt(1 / a01)
    j56 = -2.0 * beta3 * u[5] * jnp.sqrt(1 / a03)

    j65 = 2.0 * beta2 * u[4] * jnp.sqrt(1 / a02)
    j66 = -2.0 * beta3 * u[5] * jnp.sqrt(1 / a03)

    return jnp.array(
        [
            [1.0, 0.0, 0.0, j14, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, j25, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, j36],
            [j41, j42, j43, j44, j45, j46],
            [0.0, 0.0, 0.0, j54, 0.0, j56],
            [0.0, 0.0, 0.0, 0.0, j65, j66],
        ]
    )


@jaxtyped(typechecker=typechecker)
def calculate_f_anastomosis(
    u: Float[Array, " 6"], a0s: Float[Array, " 3"], betas: Float[Array, " 3"]
) -> Float[Array, " 6"]:
    """
    Calculates the function vector for the anastomosis problem.

    Parameters:
    u (Float[Array, " 6"]): Current state vector.
    a0s (Float[Array, " 3"]): Tuple of reference cross-sectional areas for vessels 1, 2, and 3.
    betas (Float[Array, " 3"]): Tuple of stiffness coefficients for vessels 1, 2, and 3.

    Returns:
    Float[Array, " 6"]: Function vector.
    """
    a01, a02, a03 = a0s
    beta1, beta2, beta3 = betas

    u42 = u[3] ** 2
    u52 = u[4] ** 2
    u62 = u[5] ** 2

    f1 = 0
    f2 = 0
    f3 = 0
    f4 = u[0] * u42**2 + u[1] * u52**2 - u[2] * u62**2

    f5 = beta1 * (u42 * jnp.sqrt(1 / a01) - 1.0) - (
        beta3 * (u62 * jnp.sqrt(1 / a03) - 1.0)
    )
    f6 = beta2 * (u52 * jnp.sqrt(1 / a02) - 1.0) - (
        beta3 * (u62 * jnp.sqrt(1 / a03) - 1.0)
    )

    return jnp.array([f1, f2, f3, f4, f5, f6])


@jaxtyped(typechecker=typechecker)
def update_anastomosis(
    u: Float[Array, " 6"],
    a01: Scalar,
    a02: Scalar,
    a03: Scalar,
    beta1: Scalar,
    beta2: Scalar,
    beta3: Scalar,
    gamma1: Scalar,
    gamma2: Scalar,
    gamma3: Scalar,
    p_ext1: Scalar,
    p_ext2: Scalar,
    p_ext3: Scalar,
) -> tuple[
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
    Scalar,
]:
    """
    Updates the state of the anastomosis problem based on the current state vector.

    Parameters:
    u (Float[Array, " 6"]): Current state vector.
    a01, a02, a03 (Scalar): Reference cross-sectional areas for vessels 1, 2, and 3.
    beta1, beta2, beta3 (Scalar): Stiffness coefficients for vessels 1, 2, and 3.
    gamma1, gamma2, gamma3 (Scalar): Admittance coefficients for vessels 1, 2, and 3.
    p_ext1, p_ext2, p_ext3 (Scalar): External pressures for vessels 1, 2, and 3.

    Returns:
    tuple: Updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures for vessels 1, 2, and 3.
    """

    u1 = u[0]
    u2 = u[1]
    u3 = u[2]

    a1 = u[3] ** 4
    a2 = u[4] ** 4
    a3 = u[5] ** 4

    q1 = u1 * a1
    q2 = u2 * a2
    q3 = u3 * a3

    c1 = waveSpeed(a1, gamma1)
    c2 = waveSpeed(a2, gamma2)
    c3 = waveSpeed(a3, gamma3)

    p1 = pressure(a1, a01, beta1, p_ext1)
    p2 = pressure(a2, a02, beta2, p_ext2)
    p3 = pressure(a3, a03, beta3, p_ext3)

    return u1, u2, u3, q1, q2, q3, a1, a2, a3, c1, c2, c3, p1, p2, p3
