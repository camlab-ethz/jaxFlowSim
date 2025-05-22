"""
Module for solving vascular anastomosis junction problems using the Newton–Raphson method.

This module provides a complete workflow for assembling, solving, and post-processing
the nonlinear system governing three-vessel anastomosis (junction) flows.

Functions
---------
solve_anastomosis(us, a, a0s, betas, gammas, p_exts)
    Initialize the state vector and solve for velocities and areas at the junction.
calculate_jacobian_anastomosis(u0, k, a0s, betas)
    Assemble the 6×6 Jacobian matrix of the nonlinear anastomosis equations.
calculate_f_anastomosis(u, a0s, betas)
    Evaluate the nonlinear residual vector for the current state.
update_anastomosis(u, a0s, betas, gammas, p_exts)
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
def solve_anastomosis(
    us: TripleFloat,
    a: TripleFloat,
    a0s: TripleFloat,
    betas: TripleFloat,
    gammas: TripleFloat,
    p_exts: TripleFloat,
) -> TripleJunctionReturn:
    """
    Solve the three-vessel anastomosis junction problem using Newton–Raphson.

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
    u0 = jnp.concatenate(
        [
            us,
            jnp.sqrt(jnp.sqrt(a)),  # initial sqrt-area state variables
        ]
    )

    k = jnp.sqrt(1.5 * gammas)

    # Assemble Jacobian and solve nonlinear system
    j = calculate_jacobian_anastomosis(u0, k, a0s, betas)
    u = newton_raphson(
        calculate_f_anastomosis,
        j,
        u0,
        a0s,
        betas,
    )

    # Post-process solution into physical quantities
    return update_anastomosis(
        u,
        a0s,
        betas,
        gammas,
        p_exts,
    )


@jaxtyped(typechecker=typechecker)
def calculate_jacobian_anastomosis(
    u0: HexaFloat,
    k: TripleFloat,
    a0s: TripleFloat,
    betas: TripleFloat,
) -> LargeJacobian:
    """
    Compute the Jacobian matrix of the anastomosis residuals.

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
    u: HexaFloat, a0s: TripleFloat, betas: TripleFloat
) -> HexaFloat:
    """
    Evaluate the residual vector of the anastomosis equations.

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
    a01, a02, a03 = a0s
    beta1, beta2, beta3 = betas

    u42 = u[3] ** 2
    u52 = u[4] ** 2
    u62 = u[5] ** 2

    f1 = 0.0
    f2 = 0.0
    f3 = 0.0
    f4 = u[0] * u42**2 + u[1] * u52**2 - u[2] * u62**2

    f5 = beta1 * (u42 * jnp.sqrt(1 / a01) - 1.0) - beta3 * (
        u62 * jnp.sqrt(1 / a03) - 1.0
    )
    f6 = beta2 * (u52 * jnp.sqrt(1 / a02) - 1.0) - beta3 * (
        u62 * jnp.sqrt(1 / a03) - 1.0
    )

    return jnp.array([f1, f2, f3, f4, f5, f6])


@jaxtyped(typechecker=typechecker)
def update_anastomosis(
    u: HexaFloat,
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
    # Recover cross-sectional areas
    a = u[3:] ** 4

    # Volumetric flows
    qs = u[:3] * a

    # Wave speeds via utility function
    cs = wave_speed(a, gammas)

    # Pressures via wall law
    ps = pressure(a, a0s, betas, p_exts)

    return jnp.concatenate([u[:3], qs, a, cs, ps])
