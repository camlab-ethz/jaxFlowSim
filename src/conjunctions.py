"""
This module provides functions to solve the conjunction problem in a vascular network model using JAX.

It includes functions to:
- Initialize and solve the conjunction problem (`solve_conjunction`).
- Calculate the Jacobian matrix specific to the conjunction problem (`calculate_jacobian_conjunction`).
- Evaluate the function vector for the current state of the conjunction problem (`calculate_f_conjunction`).
- Update the state of the system after solving the equations (`update_conjunction`).

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
def solve_conjunction(
    us: Float[Array, " 2"],
    a: Float[Array, " 2"],
    a0s: Float[Array, " 2"],
    betas: Float[Array, " 2"],
    gammas: Float[Array, " 2"],
    p_exts: Float[Array, " 2"],
    rho,
) -> Float[Array, " 10"]:
    """
    Solves the conjunction problem using the Newton-Raphson method.

    Parameters:
    us (Float[Array, "2"]): Initial velocities for vessels 1 and 2.
    a (Float[Array, "2"]): Initial cross-sectional areas for vessels 1 and 2.
    a0s (Float[Array, "2"]): Reference cross-sectional areas for vessels 1 and 2.
    betas (Float[Array, "2"]): Stiffness coefficients for vessels 1 and 2.
    gammas (Float[Array, "2"]): Admittance coefficients for vessels 1 and 2.
    p_exts (Float[Array, "2"]): External pressures for vessels 1 and 2.
    rho (Float): Blood density.

    Returns:
    Float[Array, "10"]: Updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures for vessels 1 and 2.
    """
    u0 = jnp.concatenate((us, jnp.sqrt(jnp.sqrt(a))))

    k = jnp.append(jnp.sqrt(1.5 * gammas), rho)

    j = calculate_jacobian_conjunction(u0, k, a0s, betas)
    u = newton_raphson(calculate_f_conjunction, j, u0, a0s, betas)

    return update_conjunction(u, a0s, betas, gammas, p_exts)


@jaxtyped(typechecker=typechecker)
def calculate_jacobian_conjunction(
    u0: Float[Array, " 4"],
    k: Float[Array, " 3"],
    a0s: Float[Array, " 2"],
    betas: Float[Array, " 2"],
) -> Float[Array, "4 4"]:
    """
    Calculates the Jacobian matrix for the conjunction problem.

    Parameters:
    u0 (Float[Array, "4"]): Initial guess for the solution vector.
    k (Float[Array, "3"]): Array of k parameters.
    a0s (Float[Array, "2"]): Reference cross-sectional areas for vessels 1 and 2.
    betas (Float[Array, "2"]): Stiffness coefficients for vessels 1 and 2.

    Returns:
    Float[Array, "4 4"]: Jacobian matrix.
    """
    u33 = u0[2] * u0[2] * u0[2]
    u43 = u0[3] * u0[3] * u0[3]

    j13 = 4.0 * k[0]
    j24 = -4.0 * k[1]

    j31 = u33 * u0[2]
    j32 = -u43 * u0[3]
    j33 = 4.0 * u0[0] * u33
    j34 = -4.0 * u0[1] * u43

    j43 = 2.0 * betas[0] * u0[2] * jnp.sqrt(1 / a0s[0])
    j44 = -2.0 * betas[1] * u0[3] * jnp.sqrt(1 / a0s[1])

    return jnp.array(
        [
            [1.0, 0.0, j13, 0.0],
            [0.0, 1.0, 0.0, j24],
            [j31, j32, j33, j34],
            [0.0, 0.0, j43, j44],
        ]
    )


@jaxtyped(typechecker=typechecker)
def calculate_f_conjunction(
    u0: Float[Array, " 4"], a0s: Float[Array, " 2"], betas: Float[Array, " 2"]
) -> Float[Array, " 4"]:
    """
    Evaluates the function vector for the current state of the conjunction problem.

    Parameters:
    u0 (Float[Array, "4"]): Solution vector.
    a0s (Float[Array, "2"]): Reference cross-sectional areas for vessels 1 and 2.
    betas (Float[Array, "2"]): Stiffness coefficients for vessels 1 and 2.

    Returns:
    Float[Array, "4"]: Function values for the conjunction problem.
    """

    a01, a02 = a0s
    (
        beta1,
        beta2,
    ) = betas

    u32 = u0[2] * u0[2]
    u42 = u0[3] * u0[3]

    f1 = 0
    f2 = 0
    f3 = u0[0] * u32 * u32 - u0[1] * u42 * u42

    f4 = beta1 * (u32 * jnp.sqrt(1 / a01) - 1.0) - +beta2 * (
        u42 * jnp.sqrt(1 / a02) - 1.0
    )

    return jnp.array([f1, f2, f3, f4], dtype=jnp.float64)


@jaxtyped(typechecker=typechecker)
def update_conjunction(
    u0: Float[Array, " 4"],
    a0s: Float[Array, " 2"],
    betas: Float[Array, " 2"],
    gammas: Float[Array, " 2"],
    p_exts: Float[Array, " 2"],
) -> Float[Array, " 10"]:
    """
    Updates the state of the conjunction problem based on the current state vector.

    Parameters:
    u0 (Float[Array, "4"]): Solution vector.
    a0s (Float[Array, "2"]): Reference cross-sectional areas for vessels 1 and 2.
    betas (Float[Array, "2"]): Stiffness coefficients for vessels 1 and 2.
    gammas (Float[Array, "2"]): Admittance coefficients for vessels 1 and 2.
    p_exts (Float[Array, "2"]): External pressures for vessels 1 and 2.

    Returns:
    Float[Array, "10"]: Updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures for vessels 1 and 2.
    """

    a = jnp.array([u0[2] * u0[2] * u0[2] * u0[2], u0[3] * u0[3] * u0[3] * u0[3]])
    qs = u0[:2] * a

    ps = pressure(a, a0s, betas, p_exts)

    cs = wave_speed(a, gammas)

    return jnp.array([*u0[:2], *qs, *a, *cs, *ps])
