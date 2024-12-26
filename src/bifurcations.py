"""
This module provides functionality to solve the bifurcation problem in vascular networks using the Newton-Raphson method.

It includes functions to:
- Initialize and solve the bifurcation problem (`solve_bifurcation`).
- Calculate the Jacobian matrix specific to the bifurcation problem (`calculate_jacobian_bifurcation`).
- Evaluate the function vector for the current state of the bifurcation problem (`calculate_f_bifurcation`).
- Update the state of the system after solving the equations (`update_bifurcation`).

The module makes use of the following imported utilities:
- `newtonRaphson` from `src.newton` for solving the system of nonlinear equations.
- `pressure` and `waveSpeed` from `src.utils` for calculating pressure and wave speed in the vessels.
- `jax.numpy` for numerical operations and array handling.
- `jaxtyping` and `typeguard` for type checking and ensuring type safety in the functions.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from src.newton import newton_raphson
from src.utils import pressure, wave_speed


@jaxtyped(typechecker=typechecker)
def solve_bifurcation(
    us: Float[Array, " 3"],
    a: Float[Array, " 3"],
    a0s: Float[Array, " 3"],
    betas: Float[Array, " 3"],
    gammas: Float[Array, " 3"],
    p_exts: Float[Array, " 3"],
) -> Float[Array, " 15"]:
    """
    Solves the bifurcation problem using the Newton-Raphson method.

    Parameters:
    us (Float[Array, "3"]): Initial velocities for vessels 1, 2, and 3.
    a (Float[Array, "3"]): Initial cross-sectional areas for vessels 1, 2, and 3.
    a0s (Float[Array, "3"]): Reference cross-sectional areas for vessels 1, 2, and 3.
    betas (Float[Array, "3"]): Stiffness coefficients for vessels 1, 2, and 3.
    gammas (Float[Array, "3"]): Admittance coefficients for vessels 1, 2, and 3.
    p_exts (Float[Array, "3"]): External pressures for vessels 1, 2, and 3.

    Returns:
    Float[Array, "15"]: Updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures for vessels 1, 2, and 3.
    """
    u0 = jnp.concatenate([us, jnp.sqrt(jnp.sqrt(a))])

    k = jnp.sqrt(1.5 * gammas)

    j = calculate_jacobian_bifurcation(u0, k, a0s, betas)  # pylint: disable=E1111
    u = newton_raphson(calculate_f_bifurcation, j, u0, a0s, betas)

    return update_bifurcation(u, a0s, betas, gammas, p_exts)


@jaxtyped(typechecker=typechecker)
def calculate_jacobian_bifurcation(
    u0: Float[Array, " 6"],
    k: Float[Array, " 3"],
    a0s: Float[Array, " 3"],
    betas: Float[Array, " 3"],
) -> Float[Array, "6 6"]:
    """
    Calculates the Jacobian matrix for the bifurcation problem.

    Parameters:
    u0 (Float[Array, "6"]): Initial guess for the solution vector.
    k (Float[Array, "3"]): Array of k parameters.
    a0s (Float[Array, "3"]): Reference cross-sectional areas for vessels 1, 2, and 3.
    betas (Float[Array, "3"]): Stiffness coefficients for vessels 1, 2, and 3.

    Returns:
    Float[Array, "6 6"]: Jacobian matrix.
    """
    u43 = u0[3] ** 3
    u53 = u0[4] ** 3
    u63 = u0[5] ** 3

    j14 = 4.0 * k[0]
    j25 = -4.0 * k[1]
    j36 = -4.0 * k[2]

    j41 = u0[3] * u43
    j42 = -u0[4] * u53
    j43 = -u0[5] * u63
    j44 = 4.0 * u0[0] * u43
    j45 = -4.0 * u0[1] * u53
    j46 = -4.0 * u0[2] * u63

    j54 = 2.0 * betas[0] * u0[3] * 1 / jnp.sqrt(a0s[0])
    j55 = -2.0 * betas[1] * u0[4] * 1 / jnp.sqrt(a0s[1])

    j64 = 2.0 * betas[0] * u0[3] * 1 / jnp.sqrt(a0s[0])
    j66 = -2.0 * betas[2] * u0[5] * 1 / jnp.sqrt(a0s[2])

    return jnp.array(
        [
            [1.0, 0.0, 0.0, j14, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, j25, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, j36],
            [j41, j42, j43, j44, j45, j46],
            [0.0, 0.0, 0.0, j54, j55, 0.0],
            [0.0, 0.0, 0.0, j64, 0.0, j66],
        ]
    )


@jaxtyped(typechecker=typechecker)
def calculate_f_bifurcation(
    u0s: Float[Array, " 6"], a0s: Float[Array, " 3"], betas: Float[Array, " 3"]
) -> Float[Array, " 6"]:
    """
    Evaluates the function vector for the current state of the bifurcation problem.

    Parameters:
    u0s (Float[Array, "6"]): Solution vector.
    a0s (Float[Array, "3"]): Reference cross-sectional areas for vessels 1, 2, and 3.
    betas (Float[Array, "3"]): Stiffness coefficients for vessels 1, 2, and 3.

    Returns:
    Float[Array, "6"]: Function values for the bifurcation problem.
    """

    u42 = u0s[3] * u0s[3]
    u52 = u0s[4] * u0s[4]
    u62 = u0s[5] * u0s[5]

    f1 = 0  # U[0] + 4.0 * k[0] * U[3] - W[0]
    f2 = 0  # U[1] - 4.0 * k[1] * U[4] - W[1]
    f3 = 0  # U[2] - 4.0 * k[2] * U[5] - W[2]
    f4 = u0s[0] * (u42 * u42) - u0s[1] * (u52 * u52) - u0s[2] * (u62 * u62)

    f5 = betas[0] * (u42 * jnp.sqrt(1 / a0s[0]) - 1.0) - (
        betas[1] * (u52 * jnp.sqrt(1 / a0s[1]) - 1.0)
    )
    f6 = betas[0] * (u42 * jnp.sqrt(1 / a0s[0]) - 1.0) - (
        betas[2] * (u62 * jnp.sqrt(1 / a0s[2]) - 1.0)
    )

    return jnp.array([f1, f2, f3, f4, f5, f6])


@jaxtyped(typechecker=typechecker)
def update_bifurcation(
    u0s: Float[Array, " 6"],
    a0s: Float[Array, " 3"],
    betas: Float[Array, " 3"],
    gammas: Float[Array, " 3"],
    p_exts: Float[Array, " 3"],
) -> Float[Array, " 15"]:
    """
    Updates the state of the bifurcation problem based on the current state vector.

    Parameters:
    u0s (Float[Array, "6"]): Solution vector.
    a0s (Float[Array, "3"]): Reference cross-sectional areas for vessels 1, 2, and 3.
    betas (Float[Array, "3"]): Stiffness coefficients for vessels 1, 2, and 3.
    gammas (Float[Array, "3"]): Admittance coefficients for vessels 1, 2, and 3.
    p_exts (Float[Array, "3"]): External pressures for vessels 1, 2, and 3.

    Returns:
    Float[Array, "15"]: Updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures for vessels 1, 2, and 3.
    """

    a = u0s[3:] * u0s[3:] * u0s[3:] * u0s[3:]

    qs = u0s[:3] * a

    ps = pressure(a, a0s, betas, p_exts)

    cs = wave_speed(a, gammas)

    return jnp.concatenate([u0s[:3], qs, a, cs, ps])
