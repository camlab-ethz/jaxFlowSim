"""
This module provides functionality to set boundary conditions for inlet and outlet in a vascular network model using JAX.

It includes functions to:
- Set the inlet boundary condition (`set_inlet_bc`).
- Set the outlet boundary condition (`set_outlet_bc`).
- Calculate the input from provided data (`input_from_data`).
- Ensure inlet compatibility (`inlet_compatibility`).
- Ensure outlet compatibility (`outlet_compatibility`).
- Handle three-element Windkessel model for the outlet (`three_element_windkessel`).
- Calculate Riemann invariants and their inverse (`riemann_invariants`, `inverse_riemann_invariants`).
- Calculate area from pressure (`areaFromPressure`).

The module makes use of the following imported utilities:
- `pressure` from `src.utils` for calculating pressure in the vessels.
- `jax.numpy` and `jax.lax` for numerical operations and control flow.
- `jaxtyping` and `beartype` for type checking and ensuring type safety in the functions.
"""

import jax.numpy as jnp
from jax import lax
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

from src.utils import pressure
from src.types import (
    HexaFloat,
    InputData,
    QuadFloat,
    ScalarFloat,
    PairFloat,
    TripleFloat,
)


@jaxtyped(typechecker=typechecker)
def set_inlet_bc(
    inlet: ScalarFloat,
    us: PairFloat,
    a: ScalarFloat,
    cs: PairFloat,
    t: ScalarFloat,
    dt: ScalarFloat,
    input_data: InputData,
    cardiac_t: ScalarFloat,
    inv_dx: ScalarFloat,
    a0: ScalarFloat,
    beta: ScalarFloat,
    p_ext: ScalarFloat,
):
    """
    Sets the inlet boundary condition.

    Parameters:
    inlet (ScalarFloat): Indicator for inlet type (1 for flow, 0 for pressure).
    us (PairFloat): Velocities at the inlet.
    a (ScalarFloat): Cross-sectional area at the inlet.
    cs (PairFloat): Wave speeds at the inlet.
    t (ScalarFloat): Current time.
    dt (ScalarFloat): Time step.
    input_data (InputData): Input data for boundary conditions.
    cardiac_t (ScalarFloat): Cardiac cycle time.
    inv_dx (ScalarFloat): Inverse spatial step size.
    a0 (ScalarFloat): Reference cross-sectional area.
    beta (ScalarFloat): Stiffness coefficient.
    p_ext (ScalarFloat): External pressure.

    Returns:
    ScalarFloat: Updated boundary conditions.
    """
    q0, p0 = lax.cond(
        inlet == 1,
        lambda: (input_from_data(t, input_data.transpose(), cardiac_t), 0.0),
        lambda: (0.0, input_from_data(t, input_data.transpose(), cardiac_t)),
    )
    return inlet_compatibility(inlet, us, q0, a, cs, p0, dt, inv_dx, a0, beta, p_ext)


@jaxtyped(typechecker=typechecker)
def input_from_data(
    t: ScalarFloat, input_data: InputData, cardiac_t: ScalarFloat
) -> ScalarFloat:
    """
    Extracts input values from provided data.

    Parameters:
    t (ScalarFloat): Current time.
    input_data (InputData): Input data array.
    cardiac_t (ScalarFloat): Cardiac cycle time.

    Returns:
    ScalarFloat: Interpolated input value.
    """
    idt = input_data[:, 0]
    idt1 = idt
    idt1 = idt1.at[:-1].set(idt1[1:])
    idq = input_data[:, 1]

    t_hat = t // cardiac_t
    t -= t_hat * cardiac_t

    idx = (
        jnp.where(
            (t >= idt) & (t <= idt1), jnp.arange(0, idt.size, 1), jnp.zeros(idt.size)
        )
        .sum()
        .astype(int)
    )

    qu = idq[idx] + (t - idt[idx]) * (idq[idx + 1] - idq[idx]) / (
        idt[idx + 1] - idt[idx]
    )

    return qu


@jaxtyped(typechecker=typechecker)
def inlet_compatibility(
    inlet: ScalarFloat,
    us: PairFloat,
    q0: ScalarFloat,
    a: ScalarFloat,
    cs: PairFloat,
    p0: ScalarFloat,
    dt: ScalarFloat,
    inv_dx: ScalarFloat,
    a0: ScalarFloat,
    beta: ScalarFloat,
    p_ext: ScalarFloat,
) -> PairFloat:
    """
    Ensures inlet compatibility by calculating updated boundary conditions.

    Parameters:
    inlet (ScalarFloat): Indicator for inlet type (1 for flow, 0 for pressure).
    us (PairFloat): Velocities at the inlet.
    q0 (ScalarFloat): Initial flow rate at the inlet.
    a (ScalarFloat): Cross-sectional area at the inlet.
    cs (PairFloat): Wave speeds at the inlet.
    p0 (ScalarFloat): Initial pressure at the inlet.
    dt (ScalarFloat): Time step.
    inv_dx (ScalarFloat): Inverse spatial step size.
    a0 (ScalarFloat): Reference cross-sectional area.
    beta (ScalarFloat): Stiffness coefficient.
    p_ext (ScalarFloat): External pressure.

    Returns:
    PairFloat: Updated boundary conditions.
    """
    w11, w21 = riemann_invariants(us[0], cs[0])
    w12, _ = riemann_invariants(us[1], cs[1])

    w11 += (w12 - w11) * (cs[0] - us[0]) * dt * inv_dx
    w21 = 2.0 * q0 / a - w11

    u0, _ = inverse_riemann_invariants(w11, w21)

    return lax.cond(
        inlet == 1,
        lambda: jnp.array([q0, q0 / u0], dtype=jnp.float64),
        lambda: jnp.array(
            [
                u0 * areaFromPressure(p0, a0, beta, p_ext),
                areaFromPressure(p0, a0, beta, p_ext),
            ],
            dtype=jnp.float64,
        ),
    )


@jaxtyped(typechecker=typechecker)
def riemann_invariants(u: ScalarFloat, c: ScalarFloat) -> PairFloat:
    """
    Calculates the Riemann invariants for given velocity and wave speed.

    Parameters:
    u (ScalarFloat): Velocity.
    c (ScalarFloat): Wave speed.

    Returns:
    PairFloat: Riemann invariants (w1, w2).
    """
    w1 = u - 4.0 * c
    w2 = u + 4.0 * c

    return jnp.array([w1, w2])


@jaxtyped(typechecker=typechecker)
def inverse_riemann_invariants(w1: ScalarFloat, w2: ScalarFloat) -> PairFloat:
    """
    Calculates velocity and wave speed from Riemann invariants.

    Parameters:
    w1 (ScalarFloat): First Riemann invariant.
    w2 (ScalarFloat): Second Riemann invariant.

    Returns:
    PairFloat: Velocity and wave speed.
    """
    u = 0.5 * (w1 + w2)
    c = (w2 - w1) * 0.125

    return jnp.array([u, c])


@jaxtyped(typechecker=typechecker)
def areaFromPressure(
    p: ScalarFloat,
    a0: ScalarFloat,
    beta: ScalarFloat,
    p_ext: ScalarFloat,
) -> ScalarFloat:
    """
    Calculates cross-sectional area from pressure using reference area and stiffness coefficient.

    Parameters:
    p (ScalarFloat): Pressure.
    a0 (ScalarFloat): Reference cross-sectional area.
    beta (ScalarFloat): Stiffness coefficient.
    p_ext (ScalarFloat): External pressure.

    Returns:
    ScalarFloat: Cross-sectional area.
    """
    return a0 * ((p - p_ext) / beta + 1.0) * ((p - p_ext) / beta + 1.0)


@jaxtyped(typechecker=typechecker)
def set_outlet_bc(
    dt: ScalarFloat,
    us: PairFloat,
    q1: ScalarFloat,
    a1: ScalarFloat,
    cs: PairFloat,
    ps: TripleFloat,
    pc: ScalarFloat,
    ws: PairFloat,
    a0: ScalarFloat,
    beta: ScalarFloat,
    gamma: ScalarFloat,
    dx: ScalarFloat,
    p_ext: ScalarFloat,
    outlet: ScalarFloat,
    wks: QuadFloat,
) -> HexaFloat:
    """
    Sets the outlet boundary condition.

    Parameters:
    dt (ScalarFloat): Time step.
    us (PairFloat): Velocities at the outlet.
    q1 (ScalarFloat): Initial flow rate at the outlet.
    a1 (ScalarFloat): Cross-sectional area at the outlet.
    cs (PairFloat): Wave speeds at the outlet.
    ps (TripleFloat): Pressures at the outlet.
    pc (ScalarFloat): Compliance pressure.
    ws (PairFloat): Riemann invariants at the outlet.
    a0 (ScalarFloat): Reference cross-sectional area.
    beta (ScalarFloat): Stiffness coefficient.
    gamma (ScalarFloat): Admittance coefficient.
    dx (ScalarFloat): Spatial step size.
    p_ext (ScalarFloat): External pressure.
    outlet (ScalarFloat): Indicator for outlet type.
    wks (QuadFloat): Windkessel model parameters.

    Returns:
    HexaFloat: Updated boundary conditions for the outlet.
    """

    def outlet_compatibility_wrapper():
        p1_out = 2.0 * ps[1] - ps[2]
        u1_out, q1_out, c1_out = outlet_compatibility(us, a1, cs, ws, dt, dx, wks[0])
        return jnp.array([u1_out, q1_out, a1, c1_out, p1_out, pc], dtype=jnp.float64)

    def three_element_windkessel_wrapper():
        u1_out, a1_out, pc_out = three_element_windkessel(
            dt, us[0], a1, pc, wks[1:], beta, gamma, a0, p_ext
        )
        return jnp.array([u1_out, q1, a1_out, cs[0], ps[0], pc_out], dtype=jnp.float64)

    return lax.cond(
        outlet == 1,
        outlet_compatibility_wrapper,
        three_element_windkessel_wrapper,
    )


@jaxtyped(typechecker=typechecker)
def outlet_compatibility(
    us: PairFloat,
    a1: ScalarFloat,
    cs: PairFloat,
    ws: PairFloat,
    dt: ScalarFloat,
    dx: ScalarFloat,
    rt: ScalarFloat,
) -> TripleFloat:
    """
    Ensures outlet compatibility by calculating updated boundary conditions.

    Parameters:
    us (PairFloat): Velocities at the outlet.
    a1 (ScalarFloat): Cross-sectional area at the outlet.
    cs (PairFloat): Wave speeds at the outlet.
    ws (PairFloat): Riemann invariants at the outlet.
    dt (ScalarFloat): Time step.
    dx (ScalarFloat): Spatial step size.
    rt (ScalarFloat): Resistance term.

    Returns:
    TripleFloat: Updated velocity, flow rate, and wave speed at the outlet.
    """
    _, w2m1 = riemann_invariants(us[1], cs[1])
    w1m, w2m = riemann_invariants(us[0], cs[0])

    w2m += (w2m1 - w2m) * (us[0] + cs[0]) * dt / dx
    w1m = ws[0] - rt * (w2m - ws[2])

    u1, c1 = inverse_riemann_invariants(w1m, w2m)
    q1 = a1 * u1

    return jnp.array([u1, q1, c1])


@jaxtyped(typechecker=typechecker)
def three_element_windkessel(
    dt: ScalarFloat,
    u1: ScalarFloat,
    a1: ScalarFloat,
    pc: ScalarFloat,
    wks: TripleFloat,
    beta: ScalarFloat,
    gamma: ScalarFloat,
    a0: ScalarFloat,
    p_ext: ScalarFloat,
) -> TripleFloat:
    """
    Handles the three-element Windkessel model for the outlet.

    Parameters:
    dt (ScalarFloat): Time step.
    u1 (ScalarFloat): Velocity at the outlet.
    a1 (ScalarFloat): Cross-sectional area at the outlet.
    pc (ScalarFloat): Compliance pressure.
    wks (TripleFloat): Windkessel model parameters.
    beta (ScalarFloat): Stiffness coefficient.
    gamma (ScalarFloat): Admittance coefficient.
    a0 (ScalarFloat): Reference cross-sectional area.
    p_ext (ScalarFloat): External pressure.

    Returns:
    TripleFloat: Updated velocity, cross-sectional area, and compliance pressure at the outlet.
    """
    p_out = 0.0

    al = a1
    ul = u1
    pc += dt / wks[2] * (al * ul - (pc - p_out) / wks[1])

    a = al
    ssal = jnp.sqrt(jnp.sqrt(al))
    sgamma = 2 * jnp.sqrt(6 * gamma)
    sa0 = jnp.sqrt(a0)
    ba0 = beta / sa0

    def fun(a):
        return (
            a * wks[0] * (ul + sgamma * (ssal - jnp.sqrt(jnp.sqrt(a))))
            - (p_ext + ba0 * (jnp.sqrt(a) - sa0))
            + pc
        )

    def dfun(a):
        return wks[0] * (
            ul + sgamma * (ssal - 1.25 * jnp.sqrt(jnp.sqrt(a)))
        ) - ba0 * 0.5 / jnp.sqrt(a)

    def newton_solver(x0):
        xn = x0 - fun(x0) / dfun(x0)

        return xn

    a = newton_solver(a)

    us = (pressure(a, a0, beta, p_ext) - p_out) / (a * wks[0])

    a1 = a
    u1 = us

    return jnp.array([u1, a1, pc])
