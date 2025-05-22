"""
Module for solving vascular bifurcation junction problems using the Newton–Raphson method.

This module provides a complete workflow for assembling, solving, and post-processing
the nonlinear system governing three-vessel bifurcation flows.

Functions
---------
solve_bifurcation(us, a, a0s, betas, gammas, p_exts)
    Initialize the state vector and solve for velocities and areas at the bifurcation.
calculate_jacobian_bifurcation(u0, k, a0s, betas)
    Assemble the 6×6 Jacobian matrix of the nonlinear bifurcation equations.
calculate_f_bifurcation(u, a0s, betas)
    Evaluate the nonlinear residual vector for the current state.
update_bifurcation(u, a0s, betas, gammas, p_exts)
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
- `TripleFloat`       : JAX array of shape (3,) for vessel parameters.
- `HexaFloat`         : JAX array of shape (6,) representing the state vector.
- `LargeJacobian`     : JAX array of shape (6, 6) for the Jacobian matrix.
- `TripleJunctionReturn` : Concatenated result array [u, q, A, c, p] of length 15.

"""

import jax.numpy as jnp
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

from src.newton import newton_raphson
from src.utils import pressure, wave_speed
from src.types import LargeJacobian, TripleFloat, HexaFloat, TripleJunctionReturn


@jaxtyped(typechecker=typechecker)
def solve_bifurcation(
    us: TripleFloat,
    a: TripleFloat,
    a0s: TripleFloat,
    betas: TripleFloat,
    gammas: TripleFloat,
    p_exts: TripleFloat,
) -> TripleJunctionReturn:
    """
    Solve the three-vessel bifurcation junction problem using Newton–Raphson.

    Parameters
    ----------
    us : TripleFloat
        Initial velocities for vessels 1, 2, and 3.
    a : TripleFloat
        Initial cross-sectional areas for vessels 1, 2, and 3.
    a0s : TripleFloat
        Reference (unstressed) areas for vessels 1, 2, and 3.
    betas : TripleFloat
        Wall stiffness coefficients for vessels 1, 2, and 3.
    gammas : TripleFloat
        Admittance (compliance) coefficients for vessels 1, 2, and 3.
    p_exts : TripleFloat
        External pressures for vessels 1, 2, and 3.

    Returns
    -------
    TripleJunctionReturn
        Concatenation of:
        - Velocities (u1, u2, u3)
        - Flows (q1, q2, q3)
        - Areas (A1, A2, A3)
        - Wave speeds (c1, c2, c3)
        - Pressures (p1, p2, p3)
    """
    u0 = jnp.concatenate([us, jnp.sqrt(jnp.sqrt(a))])

    k = jnp.sqrt(1.5 * gammas)

    j = calculate_jacobian_bifurcation(u0, k, a0s, betas)  # pylint: disable=E1111
    u = newton_raphson(calculate_f_bifurcation, j, u0, a0s, betas)

    return update_bifurcation(u, a0s, betas, gammas, p_exts)


@jaxtyped(typechecker=typechecker)
def calculate_jacobian_bifurcation(
    u0: HexaFloat,
    k: TripleFloat,
    a0s: TripleFloat,
    betas: TripleFloat,
) -> LargeJacobian:
    """
    Compute the Jacobian matrix of the bifurcation residuals.

    The Jacobian is a 6×6 matrix combining continuity and wall-law equations
    for a three-vessel junction.

    Parameters
    ----------
    u0 : HexaFloat
        State vector [u1, u2, u3, A1^(1/4), A2^(1/4), A3^(1/4)].
    k : TripleFloat
        Coefficient array derived from admittance values.
    a0s : TripleFloat
        Reference areas for vessels (unstressed state).
    betas : TripleFloat
        Wall stiffness coefficients.

    Returns
    -------
    LargeJacobian
        6×6 Jacobian matrix of partial derivatives.
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
    u0s: HexaFloat, a0s: TripleFloat, betas: TripleFloat
) -> HexaFloat:
    """
    Evaluate the residual vector of the bifurcation equations.

    Residuals combine mass continuity at the junction and wall-law balances.

    Parameters
    ----------
    u : HexaFloat
        Current state vector [u1, u2, u3, A1^(1/4), A2^(1/4), A3^(1/4)].
    a0s : TripleFloat
        Reference (unstressed) cross-sectional areas.
    betas : TripleFloat
        Wall stiffness coefficients.

    Returns
    -------
    HexaFloat
        6-element residual vector [f1, f2, f3, f4, f5, f6].
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
    u0s: HexaFloat,
    a0s: TripleFloat,
    betas: TripleFloat,
    gammas: TripleFloat,
    p_exts: TripleFloat,
) -> TripleJunctionReturn:
    """
    Post-process solved state to physical output variables.

    Converts the optimized state vector into velocities, flows, areas,
    wave speeds, and pressures for each vessel.

    Parameters
    ----------
    u : HexaFloat
        Solved state vector [u1, u2, u3, A1^(1/4), A2^(1/4), A3^(1/4)].
    a0s : TripleFloat
        Reference (unstressed) cross-sectional areas.
    betas : TripleFloat
        Wall stiffness coefficients.
    gammas : TripleFloat
        Admittance (compliance) coefficients.
    p_exts : TripleFloat
        External pressures for each vessel.

    Returns
    -------
    TripleJunctionReturn
        Concatenated array [u, q, A, c, p] for vessels 1–3.
    """

    a = u0s[3:] * u0s[3:] * u0s[3:] * u0s[3:]

    qs = u0s[:3] * a

    ps = pressure(a, a0s, betas, p_exts)

    cs = wave_speed(a, gammas)

    return jnp.concatenate([u0s[:3], qs, a, cs, ps])
