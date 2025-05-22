"""
Module for solving vascular conjunction (two-vessel junction) problems using the Newton–Raphson method.

This module provides a complete workflow for assembling, solving, and post-processing
the nonlinear system governing two-vessel conjunction flows.

Functions
---------
solve_conjunction(us, a, a0s, betas, gammas, p_exts, rho)
    Initialize the state vector (including density) and solve for velocities and areas at the conjunction.
calculate_jacobian_conjunction(u0, k, a0s, betas)
    Assemble the 4×4 Jacobian matrix of the nonlinear conjunction equations.
calculate_f_conjunction(u, a0s, betas)
    Evaluate the nonlinear residual vector for the current state.
update_conjunction(u, a0s, betas, gammas, p_exts)
    Compute updated velocities, volumetric flows, cross-sectional areas,
    wave speeds, and pressures from the solved state.

Dependencies
------------
- `newton_raphson` from `src.newton`   : Custom Newton–Raphson solver for nonlinear systems.
- `pressure`, `wave_speed` from `src.utils` : Utility functions for hemodynamic calculations.
- `jax.numpy` (jnp)                   : Array operations and numeric routines.
- `jaxtyping.jaxtyped`                : Static typing of JAX arrays.
- `beartype.beartype`                 : Runtime type enforcement.

Type Aliases
------------
- `PairFloat`         : JAX array of shape (2,) for vessel parameters.
- `QuadFloat`         : JAX array of shape (4,) representing the state vector.
- `SmallJacobian`     : JAX array of shape (4,4) for the Jacobian matrix.
- `DoubleJunctionReturn` : Concatenated result array [u, q, A, c, p] of length 10.
"""

import jax.numpy as jnp
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

from src.newton import newton_raphson
from src.utils import pressure, wave_speed
from src.types import (
    PairFloat,
    TripleFloat,
    QuadFloat,
    DoubleJunctionReturn,
    SmallJacobian,
)


@jaxtyped(typechecker=typechecker)
def solve_conjunction(
    us: PairFloat,
    a: PairFloat,
    a0s: PairFloat,
    betas: PairFloat,
    gammas: PairFloat,
    p_exts: PairFloat,
    rho,
) -> DoubleJunctionReturn:
    """
    Solves the conjunction problem using the Newton-Raphson method.

    Parameters:
    us (PairFloat): Initial velocities for vessels 1 and 2.
    a (PairFloat): Initial cross-sectional areas for vessels 1 and 2.
    a0s (PairFloat): Reference cross-sectional areas for vessels 1 and 2.
    betas (PairFloat): Stiffness coefficients for vessels 1 and 2.
    gammas (PairFloat): Admittance coefficients for vessels 1 and 2.
    p_exts (PairFloat): External pressures for vessels 1 and 2.
    rho (Float): Blood density.

    Returns:
    DoubleJunctionReturn: Updated values of velocities, flow rates, cross-sectional areas, wave speeds, and pressures for vessels 1 and 2.
    """
    u0 = jnp.concatenate((us, jnp.sqrt(jnp.sqrt(a))))

    # include density as third k-entry
    k = jnp.append(jnp.sqrt(1.5 * gammas), rho)

    # assemble Jacobian and solve nonlinear system
    j = calculate_jacobian_conjunction(u0, k, a0s, betas)
    u = newton_raphson(calculate_f_conjunction, j, u0, a0s, betas)

    # post-process to physical variables
    return update_conjunction(u, a0s, betas, gammas, p_exts)


@jaxtyped(typechecker=typechecker)
def calculate_jacobian_conjunction(
    u0: QuadFloat,
    k: TripleFloat,
    a0s: PairFloat,
    betas: PairFloat,
) -> SmallJacobian:
    """
    Compute the Jacobian matrix of the conjunction residuals.

    Parameters
    ----------
    u0 : QuadFloat
        State vector [u1, u2, A1^(1/4), A2^(1/4)].
    k : array_like
        Coefficients derived from admittance and density [k1, k2, rho].
    a0s : PairFloat
        Reference areas for vessels (unstressed state).
    betas : PairFloat
        Wall stiffness coefficients.

    Returns
    -------
    SmallJacobian
        4×4 Jacobian matrix of partial derivatives.
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
    u0: QuadFloat, a0s: PairFloat, betas: PairFloat
) -> QuadFloat:
    """
    Calculates the Jacobian matrix for the conjunction problem.

    Parameters:
    u0 (QuadFloat): Initial guess for the solution vector.
    k (TripleFloat): Array of k parameters.
    a0s (PairFloat): Reference cross-sectional areas for vessels 1 and 2.
    betas (PairFloat): Stiffness coefficients for vessels 1 and 2.

    Returns:
    SmallJacobian: Jacobian matrix.
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
    u0: QuadFloat,
    a0s: PairFloat,
    betas: PairFloat,
    gammas: PairFloat,
    p_exts: PairFloat,
) -> DoubleJunctionReturn:
    """
    Post-process solved state to physical output variables.

    Converts the optimized state vector into velocities, flows, areas,
    wave speeds, and pressures for each vessel.

    Parameters
    ----------
    u : QuadFloat
        Solved state vector [u1, u2, A1^(1/4), A2^(1/4)].
    a0s : PairFloat
        Reference (unstressed) cross-sectional areas.
    betas : PairFloat
        Wall stiffness coefficients.
    gammas : PairFloat
        Admittance (compliance) coefficients.
    p_exts : PairFloat
        External pressures for each vessel.

    Returns
    -------
    DoubleJunctionReturn
        Concatenated array [u, q, A, c, p] for vessels 1 and 2.
    """

    a = jnp.array([u0[2] * u0[2] * u0[2] * u0[2], u0[3] * u0[3] * u0[3] * u0[3]])
    qs = u0[:2] * a

    ps = pressure(a, a0s, betas, p_exts)

    cs = wave_speed(a, gammas)

    return jnp.array([*u0[:2], *qs, *a, *cs, *ps])
