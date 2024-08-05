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
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker

from src.newton import newton_raphson
from src.utils import pressure, wave_speed


@jaxtyped(typechecker=typechecker)
def solve_anastomosis(
    us: Float[Array, " 3"],
    a: Float[Array, " 3"],
    a0s: Float[Array, " 3"],
    betas: Float[Array, " 3"],
    gammas: Float[Array, " 3"],
    p_exts: Float[Array, " 3"],
) -> Float[Array, " 15"]:
    """
    Solves the anastomosis problem using the Newton-Raphson method.

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
    u0 = jnp.concatenate(
        [
            us,
            jnp.sqrt(jnp.sqrt(a)),
        ]
    )

    k = jnp.sqrt(1.5 * gammas)

    j = calculate_jacobian_anastomosis(u0, k, a0s, betas)  # pylint: disable=E1111
    u = newton_raphson(
        calculate_f_anastomosis,
        j,
        u0,
        a0s,
        betas,
    )

    return update_anastomosis(
        u,
        a0s,
        betas,
        gammas,
        p_exts,
    )


@jaxtyped(typechecker=typechecker)
def calculate_jacobian_anastomosis(
    u0: Float[Array, " 6"],
    k: Float[Array, " 3"],
    a0s: Float[Array, " 3"],
    betas: Float[Array, " 3"],
) -> Float[Array, "6 6"]:
    """
    Calculates the Jacobian matrix for the anastomosis problem.

    Parameters:
    u (Float[Array, "6"]): Current state vector.
    k (Float[Array, "3"]): Array of constants derived from gamma values.
    a0s (Float[Array, "3"]): Reference cross-sectional areas for vessels 1, 2, and 3.
    betas (Float[Array, "3"]): Stiffness coefficients for vessels 1, 2, and 3.

    Returns:
    Float[Array, "6 6"]: Jacobian matrix.
    """
    u43 = u0[3] * u0[3] * u0[3]
    u53 = u0[4] * u0[4] * u0[4]
    u63 = u0[5] * u0[5] * u0[5]

    j14 = 4.0 * k[0]
    j25 = 4.0 * k[1]
    j36 = -4.0 * k[2]

    j41 = u0[3] * u43
    j42 = u0[4] * u53
    j43 = -u0[5] * u63
    j44 = 4.0 * u0[0] * u43
    j45 = 4.0 * u0[1] * u53
    j46 = -4.0 * u0[2] * u63

    beta1, beta2, beta3 = betas
    a01, a02, a03 = a0s
    j54 = 2.0 * beta1 * u0[3] * jnp.sqrt(1 / a01)
    j56 = -2.0 * beta3 * u0[5] * jnp.sqrt(1 / a03)

    j65 = 2.0 * beta2 * u0[4] * jnp.sqrt(1 / a02)
    j66 = -2.0 * beta3 * u0[5] * jnp.sqrt(1 / a03)

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
    u (Float[Array, "6"]): Current state vector.
    a0s (Float[Array, "3"]): Reference cross-sectional areas for vessels 1, 2, and 3.
    betas (Float[Array, "3"]): Stiffness coefficients for vessels 1, 2, and 3.

    Returns:
    Float[Array, "6"]: Function vector.
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
    a0s: Float[Array, " 3"],
    betas: Float[Array, " 3"],
    gammas: Float[Array, " 3"],
    p_exts: Float[Array, " 3"],
) -> Float[Array, " 15"]:
    """
    Updates the state of the anastomosis problem based on the current state vector.

    Parameters:
    u (Float[Array, "6"]): Current state vector.
    a0s (Float[Array, "3"]): Reference cross-sectional areas for vessels 1, 2, and 3.
    betas (Float[Array, "3"]): Stiffness coefficients for vessels 1, 2, and 3.
    gammas (Float[Array, "3"]): Admittance coefficients for vessels 1, 2, and 3.
    p_exts (Float[Array, "3"]): External pressures for vessels 1, 2, and 3.

    Returns:
    Float[Array, "15"]: Updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures for vessels 1, 2, and 3.
    """

    a = u[3:] ** 4

    qs = u[:3] * a

    cs = wave_speed(a, gammas)

    ps = pressure(a, a0s, betas, p_exts)

    return jnp.concatenate([u[:3], qs, a, cs, ps])
