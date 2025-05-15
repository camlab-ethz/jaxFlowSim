"""
This module provides functionality for simulating blood flow in vascular networks.

It includes functions to:
- Compute the time step size based on the Courant–Friedrichs–Lewy (CFL) condition (`compute_dt`).
- Solve the model equations for the vascular network (`solve_model`).
- Apply the Monotonic Upstream-centered Scheme for Conservation Laws (MUSCL) method for numerical flux calculation (`muscl`).
- Compute fluxes (`compute_flux`).
- Apply the superbee flux limiter (`super_bee`).
- Compute limiters for numerical fluxes (`compute_limiter` and `compute_limiter_idx`).

The module makes use of the following imported utilities:
- `jax.numpy` for numerical operations and array handling.
- `jax` for just-in-time compilation and loop constructs.
- `jaxtyping` and `beartype` for type checking and ensuring type safety in the functions.
- Various utility functions and solvers for boundary conditions and flow solutions from the `src` package.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap
from jaxtyping import Array, Float, jaxtyped, Integer
from beartype import beartype as typechecker

from src.anastomosis import solve_anastomosis
from src.bifurcations import solve_bifurcation
from src.boundary_conditions import set_inlet_bc, set_outlet_bc
from src.conjunctions import solve_conjunction
from src.types import (
    Edges,
    InputData,
    Masks,
    MasksPadded,
    ScalarFloat,
    ScalarInt,
    SimDat,
    SimDatAux,
    SimDatConst,
    SimDatConstAux,
    SimDatDouble,
    SimDatSingle,
    SimDatSingleReduced,
    StaticScalarFloat,
    StaticScalarInt,
    Strides,
    StridesReduced,
)
from src.utils import pressure_sa, wave_speed_sa


@jaxtyped(typechecker=typechecker)
def compute_dt(
    ccfl: ScalarFloat,
    u: SimDatSingle,
    c: SimDatSingle,
    dx: SimDatSingle,
) -> ScalarFloat:
    """
    Computes the time step size based on the Courant–Friedrichs–Lewy (CFL) condition.

    Parameters:
    ccfl (Float[Array, ""]): CFL number.
    u (Float[Array, "..."]): Velocity array.
    c (Float[Array, "..."]): Wave speed array.
    dx (Float[Array, "..."]): Spatial step size array.

    Returns:
    Float[Array, ""]: Computed time step size.
    """
    smax = vmap(lambda a, b: jnp.abs(a + b))(u, c)
    vessel_dt = vmap(lambda a, b: a * ccfl / b)(dx, smax)
    dt = jnp.min(vessel_dt)
    return dt


@partial(jit, static_argnums=(0, 1))
@jaxtyped(typechecker=typechecker)
def solve_model(
    n: int,
    b: StaticScalarInt,
    t: ScalarFloat,
    dt: ScalarFloat,
    # TODO: correctly tpye input_data
    input_data: Float[Array, "..."],
    rho: ScalarFloat,
    sim_dat: SimDat,
    sim_dat_aux: SimDatAux,
    sim_dat_const: SimDatConst,
    sim_dat_const_aux: SimDatConstAux,
    masks: Masks,
    strides: StridesReduced,
    edges: Edges,
) -> tuple[SimDat, SimDatAux]:
    """
    Solves the model equations for the vascular network.

    Parameters:
    n (int): Number of vessels.
    b (int): Boundary size.
    t (Float[Array, ""]): Current time.
    dt (Float[Array, ""]): Time step size.
    input_data (Float[Array, "..."]): Input data array.
    rho (Float[Array, ""]): Blood density.
    sim_dat (Float[Array, "..."]): Simulation data array.
    sim_dat_aux (Float[Array, "..."]): Auxiliary simulation data array.
    sim_dat_const (Float[Array, "..."]): Constant simulation data array.
    sim_dat_const_aux (Float[Array, "..."]): Auxiliary constant simulation data array.
    masks (Integer[Array, "..."]): Masks for boundary conditions.
    strides (Integer[Array, "..."]): Strides array.
    edges (Integer[Array, "..."]): Edges array.

    Returns:
    tuple[Float[Array, "..."], Float[Array, "..."]]: Updated simulation data and auxiliary simulation data.
    """

    inlet = sim_dat_const_aux[0, 1]
    us = jnp.array([sim_dat[0, b], sim_dat[0, b + 1]])
    a0 = sim_dat[2, b]
    cs = jnp.array([sim_dat[3, b], sim_dat[3, b + 1]])
    cardiac_t = sim_dat_const_aux[0, 0]
    dx = sim_dat_const[-1, b]
    a00 = sim_dat_const[0, b]
    beta0 = sim_dat_const[1, b]
    p_ext = sim_dat_const[4, b]

    sim_dat = sim_dat.at[1:3, 0 : b + 1].set(
        jnp.array(
            set_inlet_bc(
                inlet,
                us,
                a0,
                cs,
                t,
                dt,
                input_data,
                cardiac_t,
                1 / dx,
                a00,
                beta0,
                p_ext,
            )
        )[:, jnp.newaxis]
        * jnp.ones(b + 1)[jnp.newaxis, :]
    )

    sim_dat = sim_dat.at[:, b:-b].set(
        muscl(
            dt,
            sim_dat[1, b:-b],
            sim_dat[2, b:-b],
            sim_dat_const[0, b:-b],
            sim_dat_const[1, b:-b],
            sim_dat_const[2, b:-b],
            sim_dat_const[3, b:-b],
            sim_dat_const[-1, b:-b],
            sim_dat_const[4, b:-b],
            sim_dat_const[5, b:-b],
            masks,
        )
    )

    def set_outlet_or_junction(
        j: ScalarInt,
        dat: tuple[
            SimDat, SimDatAux, SimDatConst, SimDatConstAux, Edges, ScalarFloat, Strides
        ],
    ):
        (
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            edges,
            rho,
            strides,
        ) = dat
        end = strides[j, 1]

        def set_outlet_bc_wrapper(sim_dat: SimDat, sim_dat_aux: SimDatAux):
            us = jnp.array([sim_dat[0, end - 1], sim_dat[0, end - 2]])
            q1 = sim_dat[1, end - 1]
            a1 = sim_dat[2, end - 1]
            cs = jnp.array([sim_dat[3, end - 1], sim_dat[3, end - 2]])
            ps = jnp.array(
                [sim_dat[4, end - 1], sim_dat[4, end - 2], sim_dat[4, end - 3]]
            )
            pc = sim_dat_aux[j, 2]
            wks = jnp.array(
                [
                    sim_dat_const_aux[j, 3],
                    sim_dat_const_aux[j, 4],
                    sim_dat_const_aux[j, 5],
                    sim_dat_const_aux[j, 6],
                ]
            )
            u, q, a, c, pl, pc = set_outlet_bc(
                dt,
                us,
                q1,
                a1,
                cs,
                ps,
                pc,
                sim_dat_aux[j, :2],
                sim_dat_const[0, end - 1],
                sim_dat_const[1, end - 1],
                sim_dat_const[2, end - 1],
                sim_dat_const[-1, end - 1],
                sim_dat_const[4, end - 1],
                sim_dat_const_aux[j, 2],
                wks,
            )
            temp = jnp.array((u, q, a, c, pl))
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, end - 1),
            )
            sim_dat_aux = sim_dat_aux.at[j, 2].set(pc)
            return sim_dat, sim_dat_aux

        (sim_dat, sim_dat_aux) = lax.cond(
            sim_dat_const_aux[j, 2] != 0,
            set_outlet_bc_wrapper,
            lambda x, y: (x, y),
            sim_dat,
            sim_dat_aux,
        )

        def solve_bifurcation_wrapper(sim_dat: SimDat):
            d1_i = edges[j, 4]
            d2_i = edges[j, 5]
            d1_i_start = strides[d1_i, 0]
            d2_i_start = strides[d2_i, 0]
            us = jnp.array(
                [sim_dat[0, end - 1], sim_dat[0, d1_i_start], sim_dat[0, d2_i_start]]
            )
            a = jnp.array(
                [sim_dat[2, end - 1], sim_dat[2, d1_i_start], sim_dat[2, d2_i_start]]
            )
            a0s = jnp.array(
                [
                    sim_dat_const[0, end - 1],
                    sim_dat_const[0, d1_i_start],
                    sim_dat_const[0, d2_i_start],
                ]
            )
            betas = jnp.array(
                [
                    sim_dat_const[1, end - 1],
                    sim_dat_const[1, d1_i_start],
                    sim_dat_const[1, d2_i_start],
                ]
            )
            gammas = jnp.array(
                [
                    sim_dat_const[2, end - 1],
                    sim_dat_const[2, d1_i_start],
                    sim_dat_const[2, d2_i_start],
                ]
            )
            p_exts = jnp.array(
                [
                    sim_dat_const[4, end - 1],
                    sim_dat_const[4, d1_i_start],
                    sim_dat_const[4, d2_i_start],
                ]
            )
            (
                u1,
                u2,
                u3,
                q1,
                q2,
                q3,
                a1,
                a2,
                a3,
                c1,
                c2,
                c3,
                p1,
                p2,
                p3,
            ) = solve_bifurcation(  # pylint: disable=E1111
                us, a, a0s, betas, gammas, p_exts
            )  # pylint: disable=E1111  # pylint: disable=E1111
            temp1 = jnp.array((u1, q1, a1, c1, p1))
            temp2 = jnp.array((u2, q2, a2, c2, p2))
            temp3 = jnp.array((u3, q3, a3, c3, p3))
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp1[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, end - 1),
            )
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp2[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, d1_i_start - b),
            )
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp3[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, d2_i_start - b),
            )
            return sim_dat

        sim_dat = lax.cond(
            (sim_dat_const_aux[j, 2] == 0) * (edges[j, 3] == 2),
            solve_bifurcation_wrapper,
            lambda x: x,
            sim_dat,
        )

        def solve_conjunction_wrapper(sim_dat: SimDat, rho: ScalarFloat):
            d_i = edges[j, 7]
            d_i_start = strides[d_i, 0]
            us = jnp.array([sim_dat[0, end - 1], sim_dat[0, d_i_start]])
            a = jnp.array([sim_dat[2, end - 1], sim_dat[2, d_i_start]])
            a0s = jnp.array(
                [
                    sim_dat_const[0, end - 1],
                    sim_dat_const[0, d_i_start],
                ]
            )
            betas = jnp.array(
                [
                    sim_dat_const[1, end - 1],
                    sim_dat_const[1, d_i_start],
                ]
            )
            gammas = jnp.array(
                [
                    sim_dat_const[2, end - 1],
                    sim_dat_const[2, d_i_start],
                ]
            )
            p_exts = jnp.array(
                [
                    sim_dat_const[4, end - 1],
                    sim_dat_const[4, d_i_start],
                ]
            )
            (u1, u2, q1, q2, a1, a2, c1, c2, p1, p2) = solve_conjunction(
                us,
                a,
                a0s,
                betas,
                gammas,
                p_exts,
                rho,
            )
            temp1 = jnp.array((u1, q1, a1, c1, p1))
            temp2 = jnp.array((u2, q2, a2, c2, p2))
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp1[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, end - 1),
            )
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp2[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, d_i_start - b),
            )
            return sim_dat

        sim_dat = lax.cond(
            (sim_dat_const_aux[j, 2] == 0) * (edges[j, 3] != 2) * (edges[j, 6] == 1),
            solve_conjunction_wrapper,
            lambda x, _: x,
            sim_dat,
            rho,
        )

        def solve_anastomosis_wrapper(sim_dat: SimDat):
            p1_i = edges[j, 7]
            p2_i = edges[j, 8]
            d = edges[j, 9]
            p1_i_end = strides[p1_i, 1]
            d_start = strides[d, 0]
            us = jnp.array(
                [
                    sim_dat[0, end - 1],
                    sim_dat[0, p1_i_end - 1],
                    sim_dat[0, d_start],
                ]
            )
            qs = jnp.array(
                [
                    sim_dat[1, end - 1],
                    sim_dat[1, p1_i_end - 1],
                    sim_dat[1, d_start],
                ]
            )
            a = jnp.array(
                [
                    sim_dat[2, end - 1],
                    sim_dat[2, p1_i_end - 1],
                    sim_dat[2, d_start],
                ]
            )
            cs = jnp.array(
                [
                    sim_dat[3, end - 1],
                    sim_dat[3, p1_i_end - 1],
                    sim_dat[3, d_start],
                ]
            )
            ps = jnp.array(
                [
                    sim_dat[4, end - 1],
                    sim_dat[4, p1_i_end - 1],
                    sim_dat[4, d_start],
                ]
            )
            a0s = jnp.array(
                [
                    sim_dat_const[0, end - 1],
                    sim_dat_const[0, p1_i_end - 1],
                    sim_dat_const[0, d_start],
                ]
            )
            betas = jnp.array(
                [
                    sim_dat_const[1, end - 1],
                    sim_dat_const[1, p1_i_end - 1],
                    sim_dat_const[1, d_start],
                ]
            )
            gammas = jnp.array(
                [
                    sim_dat_const[2, end - 1],
                    sim_dat_const[2, p1_i_end - 1],
                    sim_dat_const[2, d_start],
                ]
            )
            p_exts = jnp.array(
                [
                    sim_dat_const[4, end - 1],
                    sim_dat_const[4, p1_i_end - 1],
                    sim_dat_const[4, d_start],
                ]
            )
            u1, u2, u3, q1, q2, q3, a1, a2, a3, c1, c2, c3, p1, p2, p3 = lax.cond(
                jnp.maximum(p1_i, p2_i) == j,
                lambda: solve_anastomosis(
                    us,
                    a,
                    a0s,
                    betas,
                    gammas,
                    p_exts,
                ),
                lambda: jnp.concatenate(
                    [
                        us,
                        qs,
                        a,
                        cs,
                        ps,
                    ]
                ),
            )
            temp1 = jnp.array((u1, q1, a1, c1, p1))
            temp2 = jnp.array((u2, q2, a2, c2, p2))
            temp3 = jnp.array((u3, q3, a3, c3, p3))
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp1[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, end - 1),
            )
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp2[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, p1_i_end - 1),
            )
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp3[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, d_start - b),
            )
            return sim_dat

        sim_dat = lax.cond(
            (sim_dat_const_aux[j, 2] == 0) * (edges[j, 3] != 2) * (edges[j, 6] == 2),
            solve_anastomosis_wrapper,
            lambda x: x,
            sim_dat,
        )

        return (
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux,
            edges,
            rho,
            strides,
        )

    (sim_dat, sim_dat_aux, _, _, _, _, _) = lax.fori_loop(
        0,
        n,
        set_outlet_or_junction,
        (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, rho, strides),
    )

    return sim_dat, sim_dat_aux


@jaxtyped(typechecker=typechecker)
def muscl(
    dt: ScalarFloat,
    q: SimDatSingle,
    a: SimDatSingle,
    a0: SimDatSingle,
    beta: SimDatSingle,
    gamma: SimDatSingle,
    wall_e: SimDatSingle,
    dx: SimDatSingle,
    p_ext: SimDatSingle,
    visc_t: SimDatSingle,
    masks: MasksPadded,
) -> SimDat:
    """
    Applies the Monotonic Upstream-centered Scheme for Conservation Laws (MUSCL) method for numerical flux calculation.

    Parameters:
    dt (Float[Array, ""]): Time step size.
    q (Float[Array, "..."]): Flow rate array.
    a (Float[Array, "..."]): Cross-sectional area array.
    a0 (Float[Array, "..."]): Reference cross-sectional area array.
    beta (Float[Array, "..."]): Stiffness coefficient array.
    gamma (Float[Array, "..."]): Admittance coefficient array.
    wall_e (Float[Array, "..."]): Wall elasticity array.
    dx (Float[Array, "..."]): Spatial step size array.
    p_ext (Float[Array, "..."]): External pressure array.
    visc_t (Float[Array, "..."]): Viscosity term array.
    masks (Integer[Array, "..."]): Masks for boundary conditions.

    Returns:
    Float[Array, "..."]: Updated simulation data array.
    """
    k = len(q) + 2

    s_a0 = vmap(jnp.sqrt)(a0)
    s_inv_a0 = vmap(lambda a: 1 / jnp.sqrt(a))(a0)
    half_dx = 0.5 * dx
    inv_dx = 1 / dx
    gamma_ghost = jnp.zeros(k)
    gamma_ghost = gamma_ghost.at[1:-1].set(gamma)
    gamma_ghost = gamma_ghost.at[0].set(gamma[0])
    gamma_ghost = gamma_ghost.at[-1].set(gamma[-1])
    va = jnp.empty(k)
    vq = jnp.empty(k)
    va = va.at[0].set(a[0])
    va = va.at[-1].set(a[-1])

    vq = vq.at[0].set(q[0])
    vq = vq.at[-1].set(q[-1])
    va = va.at[1:-1].set(a)
    vq = vq.at[1:-1].set(q)

    inv_dx_temp = jnp.concatenate((inv_dx, jnp.array([inv_dx[-1]])))
    limiter_a = compute_limiter(va, inv_dx_temp)
    limiter_q = compute_limiter(vq, inv_dx_temp)
    half_dx_temp = jnp.concatenate(
        (jnp.array([half_dx[0]]), half_dx, jnp.array([half_dx[-1]]))
    )
    slope_a_half_dx = vmap(lambda a, b: a * b)(limiter_a, half_dx_temp)
    slope_q_half_dx = vmap(lambda a, b: a * b)(limiter_q, half_dx_temp)

    al = vmap(lambda a, b: a + b)(va, slope_a_half_dx)
    ar = vmap(lambda a, b: a - b)(va, slope_a_half_dx)
    ql = vmap(lambda a, b: a + b)(vq, slope_q_half_dx)
    qr = vmap(lambda a, b: a - b)(vq, slope_q_half_dx)

    fl = jnp.array(vmap(compute_flux)(gamma_ghost, al, ql))
    fr = jnp.array(vmap(compute_flux)(gamma_ghost, ar, qr))

    dx_dt = dx / dt

    inv_dx_dt = dt / dx

    flux = jnp.empty((2, k - 1))
    dx_dt_temp = jnp.empty(k - 1)
    dx_dt_temp = dx_dt_temp.at[0:-1].set(dx_dt)
    dx_dt_temp = dx_dt_temp.at[-1].set(dx_dt[-1])
    flux = flux.at[0, :].set(
        vmap(lambda a, b, c, d, e: 0.5 * (a + b - e * (c - d)))(
            fr[0, 1:], fl[0, 0:-1], ar[1:], al[0:-1], dx_dt_temp
        )
    )
    flux = flux.at[1, :].set(
        vmap(lambda a, b, c, d, e: 0.5 * (a + b - e * (c - d)))(
            fr[1, 1:], fl[1, 0:-1], qr[1:], ql[0:-1], dx_dt_temp
        )
    )

    u_star = jnp.empty((2, k))
    u_star = u_star.at[0, 1:-1].set(
        vmap(lambda a, b, c, d: a + d * (b - c))(
            va[1:-1], flux[0, 0:-1], flux[0, 1:], inv_dx_dt
        )
    )
    u_star = u_star.at[1, 1:-1].set(
        vmap(lambda a, b, c, d: a + d * (b - c))(
            vq[1:-1], flux[1, 0:-1], flux[1, 1:], inv_dx_dt
        )
    )

    u_star1 = jnp.zeros((2, k + 2))
    u_star1 = u_star1.at[:, 0:-2].set(u_star)
    u_star2 = jnp.zeros((2, k + 2))
    u_star2 = u_star1.at[:, 1:-1].set(u_star)
    u_star3 = jnp.zeros((2, k + 2))
    u_star3 = u_star1.at[:, 2:].set(u_star)
    u_star2 = jnp.where(masks[0, :], u_star1, u_star2)
    u_star2 = jnp.where(masks[1, :], u_star3, u_star2)
    u_star = u_star2[:, 1:-1]

    limiter_a = compute_limiter_idx(u_star, 0, inv_dx_temp)
    limiter_q = compute_limiter_idx(u_star, 1, inv_dx_temp)
    slopes_a = vmap(lambda a, b: a * b)(limiter_a, half_dx_temp)
    slopes_q = vmap(lambda a, b: a * b)(limiter_q, half_dx_temp)

    al = vmap(lambda a, b: a + b)(u_star[0, :], slopes_a)
    ar = vmap(lambda a, b: a - b)(u_star[0, :], slopes_a)
    ql = vmap(lambda a, b: a + b)(u_star[1, :], slopes_q)
    qr = vmap(lambda a, b: a - b)(u_star[1, :], slopes_q)

    fl = jnp.array(vmap(compute_flux)(gamma_ghost, al, ql))
    fr = jnp.array(vmap(compute_flux)(gamma_ghost, ar, qr))

    flux = jnp.empty((2, k - 1))
    flux = flux.at[0, :].set(
        vmap(lambda a, b, c, d, e: 0.5 * (a + b - e * (c - d)))(
            fr[0, 1:], fl[0, 0:-1], ar[1:], al[0:-1], dx_dt_temp
        )
    )
    flux = flux.at[1, :].set(
        vmap(lambda a, b, c, d, e: 0.5 * (a + b - e * (c - d)))(
            fr[1, 1:], fl[1, 0:-1], qr[1:], ql[0:-1], dx_dt_temp
        )
    )

    a = vmap(lambda a, b, c, d, e: 0.5 * (a + b + e * (c - d)))(
        a[:], u_star[0, 1:-1], flux[0, 0:-1], flux[0, 1:], inv_dx_dt
    )
    q = vmap(lambda a, b, c, d, e: 0.5 * (a + b + e * (c - d)))(
        q[:], u_star[1, 1:-1], flux[1, 0:-1], flux[1, 1:], inv_dx_dt
    )

    s_a = vmap(jnp.sqrt)(a)
    q = vmap(lambda a, b, c, d, e: a - dt * (visc_t[0] * a / b + c * (d - e) * b))(
        q, a, wall_e, s_a, s_a0
    )
    p = vmap(lambda a, b, c, d: pressure_sa(a * b, c, d))(s_a, s_inv_a0, beta, p_ext)
    c = vmap(wave_speed_sa)(s_a, gamma)
    u = vmap(lambda a, b: a / b)(q, a)

    return jnp.stack((u, q, a, c, p))


@jaxtyped(typechecker=typechecker)
def compute_flux(
    gamma_ghost: ScalarFloat, a: ScalarFloat, q: ScalarFloat
) -> tuple[ScalarFloat, ScalarFloat]:
    """
    Computes the fluxes.

    Parameters:
    gamma_ghost (Float[Array, "..."]): Ghost cell admittance coefficient.
    a (Float[Array, "..."]): Cross-sectional area array.
    q (Float[Array, "..."]): Flow rate array.

    Returns:
    tuple[Float[Array, "..."], Float[Array, "..."]]: Computed fluxes.
    """
    return q, q * q / a + gamma_ghost * a * jnp.sqrt(a)


@jaxtyped(typechecker=typechecker)
def max_mod(a: SimDatSingle, b: SimDatSingle) -> SimDatSingle:
    """
    Applies the max mod function.

    Parameters:
    a (Float[Array, "..."]): First array.
    b (Float[Array, "..."]): Second array.

    Returns:
    Float[Array, "..."]: Result of the max mod function.
    """
    return jnp.where(a > b, a, b)


@jaxtyped(typechecker=typechecker)
def min_mod(a: SimDatSingle, b: SimDatSingle) -> SimDatSingle:
    """
    Applies the min mod function.

    Parameters:
    a (Float[Array, "..."]): First array.
    b (Float[Array, "..."]): Second array.

    Returns:
    Float[Array, "..."]: Result of the min mod function.
    """
    return jnp.where((a <= 0.0) | (b <= 0.0), 0.0, jnp.where(a < b, a, b))


@jaxtyped(typechecker=typechecker)
def super_bee(du: SimDatDouble) -> SimDatSingle:
    """
    Applies the superbee flux limiter.

    Parameters:
    du (Float[Array, "..."]): Array of differences.

    Returns:
    Float[Array, "..."]: Result of the superbee flux limiter.
    """
    return max_mod(min_mod(du[0, :], 2 * du[1, :]), min_mod(2 * du[0, :], du[1, :]))


@jaxtyped(typechecker=typechecker)
def compute_limiter(u: SimDatSingle, inv_dx: SimDatSingleReduced) -> SimDatSingle:
    """
    Computes the limiter for numerical fluxes.

    Parameters:
    u (Float[Array, "..."]): Array of values.
    inv_dx (Float[Array, "..."]): Array of inverse spatial step sizes.

    Returns:
    Float[Array, "..."]: Computed limiter values.
    """
    du = vmap(lambda a, b: a * b)(jnp.diff(u), inv_dx)
    return super_bee(
        jnp.stack(
            (
                jnp.concatenate((jnp.array([0.0]), du)),
                jnp.concatenate((du, jnp.array([0.0]))),
            )
        )
    )


@jaxtyped(typechecker=typechecker)
def compute_limiter_idx(
    u: SimDatDouble, idx: StaticScalarInt, inv_dx: SimDatSingleReduced
) -> SimDatSingle:
    """
    Computes the limiter for numerical fluxes at a specified index.

    Parameters:
    u (Float[Array, "..."]): Array of values.
    idx (int): Index for which to compute the limiter.
    inv_dx (Float[Array, "..."]): Array of inverse spatial step sizes.

    Returns:
    Float[Array, "..."]: Computed limiter values at the specified index.
    """
    du = vmap(lambda a, b: a * b)(jnp.diff(u[idx, :]), inv_dx)
    return super_bee(
        jnp.stack(
            (
                jnp.concatenate((jnp.array([0.0]), du)),
                jnp.concatenate((du, jnp.array([0.0]))),
            )
        )
    )
