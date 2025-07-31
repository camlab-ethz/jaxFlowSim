"""
Numerical solver for blood flow in vascular networks using JAX.

This module implements the core finite-volume scheme for 1D blood flow in
arterial networks. It provides functions to:

- compute_dt:       Determine the stable time step via the CFL condition.
- solve_model:      Advance the solution by one time step, applying inlet,
                    interior (MUSCL), and outlet/junction boundary updates.
- muscl:            Apply the MUSCL reconstruction and flux computation.
- compute_flux:     Compute physical fluxes for conserved variables.
- super_bee and helpers:
                    Apply the Superbee limiter to ensure monotonicity.
- compute_limiter / compute_limiter_idx:
                    Compute slope limiters for each cell or along a specified row.

Dependencies
------------
- jax.numpy (jnp)           : Array operations and linear algebra.
- jax (jit, lax, vmap)      : JIT compilation and loop/vectorization primitives.
- jaxtyping.jaxtyped        : Static typing for JAX arrays.
- beartype.beartype          : Runtime type checking.
- src.anastomosis
- src.bifurcations
- src.boundary_conditions
- src.conjunctions
- src.utils: pressure_sa, wave_speed_sa
- src.types: domain-specific array shapes and type aliases
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap
from jaxtyping import jaxtyped
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
    Compute stable time step via the CFL condition.

    Uses the maximum wave speed \\|u\\| + c on each vessel segment to ensure
    numerical stability: dt = min(ccfl * dx / max_wave_speed).

    Parameters
    ----------
    ccfl
        Courant–Friedrichs–Lewy number (dimensionless).
    u
        Velocity array for each segment (m/s).
    c
        Wave speed array for each segment (m/s).
    dx
        Spatial step size for each segment (m).

    Returns
    -------
    dt : ScalarFloat
        Time step size (s) satisfying the CFL constraint.
    """

    smax = vmap(lambda a, b: jnp.abs(a + b))(u, c)
    # Compute candidate dt per segment
    vessel_dt = vmap(lambda a, b: a * ccfl / b)(dx, smax)
    # Compute local max wave speed per segment
    dt = jnp.min(vessel_dt)
    return dt


@partial(jit, static_argnums=(0, 1))
@jaxtyped(typechecker=typechecker)
def solve_model(
    n: StaticScalarInt,
    b: StaticScalarInt,
    t: ScalarFloat,
    dt: ScalarFloat,
    input_data: InputData,
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
    Advance the blood flow solution by one time step.

    1) Apply inlet boundary condition.
    2) Update interior cells using MUSCL scheme.
    3) For each vessel outlet or junction, apply the appropriate condition:
       - outlet BC
       - bifurcation
       - conjunction
       - anastomosis

    Parameters
    ----------
    n
        Number of vessels in the network.
    b
        Number of ghost or buffer cells per vessel.
    t
        Current simulation time (s).
    dt
        Time step size (s).
    input_data
        External boundary input signals (e.g., inlet flow waveform).
    rho
        Blood density (kg/m³).
    sim_dat
        State array [u, q, a, c, p] with shape (5, total_nodes).
    sim_dat_aux
        Auxiliary state array for boundary conditions.
    sim_dat_const
        Per-node constant parameters array.
    sim_dat_const_aux
        Per-vessel constant parameters array.
    masks
        Boolean masks indicating domain interior for flux limiting.
    strides
        Start/end indices for each vessel’s interior segment.
    edges
        Connectivity matrix defining graph topology.

    Returns
    -------
    sim_dat_updated
        Updated state array after one time step.
    sim_dat_aux_updated
        Updated auxiliary array for boundary conditions.
    """
    # --- 1) Inlet boundary condition at upstream end ---
    inlet = sim_dat_const_aux[0, 1]
    # Extract local inlet state: velocities, reference area, wave speeds
    us = jnp.array([sim_dat[0, b], sim_dat[0, b + 1]])
    a0 = sim_dat[2, b]
    cs = jnp.array([sim_dat[3, b], sim_dat[3, b + 1]])
    cardiac_t = sim_dat_const_aux[0, 0]
    dx = sim_dat_const[-1, b]
    a00 = sim_dat_const[0, b]
    beta0 = sim_dat_const[1, b]
    p_ext = sim_dat_const[4, b]

    # Compute inlet updates and broadcast to ghost cells
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

    # --- 2) Interior update via MUSCL scheme over valid nodes ---
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

    # --- 3) Outlet/junction updates for each vessel ---
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

        # Outlet boundary for terminal vessels
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
            # Compute outlet BC updates
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
            # Write back new state into ghost cells
            temp = jnp.array((u, q, a, c, pl))
            sim_dat = lax.dynamic_update_slice(
                sim_dat,
                temp[:, jnp.newaxis] * jnp.ones(b + 1)[jnp.newaxis, :],
                (0, end - 1),
            )
            # Update auxiliary pressure for next cycle
            sim_dat_aux = sim_dat_aux.at[j, 2].set(pc)
            return sim_dat, sim_dat_aux

        (sim_dat, sim_dat_aux) = lax.cond(
            sim_dat_const_aux[j, 2] != 0,
            set_outlet_bc_wrapper,
            lambda x, y: (x, y),
            sim_dat,
            sim_dat_aux,
        )

        # Bifurcation junction
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
            # Solve junction equations
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
            # Update state arrays for each branch
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

        # Conjunction junction
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

        # Anastomosis junction
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
            # Only solve if branch index matches
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

    # Loop over all vessels for junction handling
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
    Apply MUSCL scheme for high-resolution flux computation.

    1) Reconstruct left/right states with slope limiters.
    2) Compute numerical fluxes using Lax–Friedrichs type formula.
    3) Update conserved variables via finite-volume update.
    4) Apply boundary masks to prevent unphysical updates.

    Parameters
    ----------
    dt
        Time step (s).
    q
        Flow rate array for interior cells.
    a
        Cross-sectional area array.
    a0
        Reference area for computing pressure.
    beta, gamma, wall_e
        Vessel stiffness and wall elasticity parameters.
    dx
        Spatial step size.
    p_ext
        External pressure.
    visc_t
        Viscous term modifier.
    masks
        Ghost‐cell masks to preserve boundary values.

    Returns
    -------
    sim_dat_updated : SimDat
        Stack [u, q, a, c, p] after one MUSCL update on interior cells.
    """
    # Number of total cells including ghost points
    k = len(q) + 2

    s_a0 = vmap(jnp.sqrt)(a0)
    s_inv_a0 = vmap(lambda a: 1 / jnp.sqrt(a))(a0)
    half_dx = 0.5 * dx
    inv_dx = 1 / dx
    # Prepare ghost‐extended arrays for reconstruction
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

    # Compute limited slopes for area and flow
    inv_dx_temp = jnp.concatenate((inv_dx, jnp.array([inv_dx[-1]])))
    limiter_a = compute_limiter(va, inv_dx_temp)
    limiter_q = compute_limiter(vq, inv_dx_temp)

    # Reconstruct left and right states at cell faces
    half_dx_temp = jnp.concatenate(
        (jnp.array([half_dx[0]]), half_dx, jnp.array([half_dx[-1]]))
    )
    slope_a_half_dx = vmap(lambda a, b: a * b)(limiter_a, half_dx_temp)
    slope_q_half_dx = vmap(lambda a, b: a * b)(limiter_q, half_dx_temp)

    al = vmap(lambda a, b: a + b)(va, slope_a_half_dx)
    ar = vmap(lambda a, b: a - b)(va, slope_a_half_dx)
    ql = vmap(lambda a, b: a + b)(vq, slope_q_half_dx)
    qr = vmap(lambda a, b: a - b)(vq, slope_q_half_dx)

    # Compute physical fluxes f = [q, q^2/a + gamma a sqrt(a)]
    fl = jnp.array(vmap(compute_flux)(gamma_ghost, al, ql))
    fr = jnp.array(vmap(compute_flux)(gamma_ghost, ar, qr))

    # Compute numerical flux via Rusanov/Lax–Friedrichs splitting
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

    # Update conserved variables u* = u - dt/dx * (f_i+1/2 - f_i-1/2)
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
    # Apply masks to preserve boundary ghost cells
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

    # Final update: compute physical primitives
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
    Physical flux function for 1D blood flow.

    f1 = q
    f2 = q^2 / a + gamma * a * sqrt(a)

    Parameters
    ----------
    gamma_ghost
        Admittance coefficient (ghost cell).
    a
        Cross-sectional area.
    q
        Flow rate.

    Returns
    -------
    (f1, f2)
        Tuple of flux components for continuity and momentum equations.
    """

    return q, q * q / a + gamma_ghost * a * jnp.sqrt(a)


@jaxtyped(typechecker=typechecker)
def max_mod(a: SimDatSingle, b: SimDatSingle) -> SimDatSingle:
    """
    Maximum modulus limiter helper: max(a, b).

    Parameters
    ----------
    a, b
        Arrays of candidate slope ratios.

    Returns
    -------
    elementwise maximum of a and b.
    """
    return jnp.where(a > b, a, b)


@jaxtyped(typechecker=typechecker)
def min_mod(a: SimDatSingle, b: SimDatSingle) -> SimDatSingle:
    """
    Minmod limiter helper: returns 0 if signals change sign, else min(\\|a\\|,\\|b\\|).

    Parameters
    ----------
    a, b
        Arrays of forward/backward differences scaled by dx.

    Returns
    -------
    elementwise minmod of a and b.
    """
    return jnp.where((a <= 0.0) | (b <= 0.0), 0.0, jnp.where(a < b, a, b))


@jaxtyped(typechecker=typechecker)
def super_bee(du: SimDatDouble) -> SimDatSingle:
    """
    Superbee flux limiter combining min_mod and max_mod for sharp resolution.

    Parameters
    ----------
    du
        2×N array where du[0] = forward diff, du[1] = backward diff.

    Returns
    -------
    limited slopes for each cell interface.
    """
    return max_mod(min_mod(du[0, :], 2 * du[1, :]), min_mod(2 * du[0, :], du[1, :]))


@jaxtyped(typechecker=typechecker)
def compute_limiter(u: SimDatSingle, inv_dx: SimDatSingleReduced) -> SimDatSingle:
    """
    Compute slope limiter values for all cell faces.

    Parameters
    ----------
    u
        Extended array of primitive variable (with ghost points).
    inv_dx
        Array of inverse spatial steps for each interval.

    Returns
    -------
    limiter values for each cell face.
    """
    # Compute forward/backward scaled differences
    du = vmap(lambda a, b: a * b)(jnp.diff(u), inv_dx)
    # Stack for minmod/superbee
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
    Compute slope limiter for a specific row of a multi-variable array.

    Parameters
    ----------
    u
        2×N array of primitive variables (with ghost points).
    idx
        Row index (0 for q, 1 for a).
    inv_dx
        Inverse spatial steps.

    Returns
    -------
    limiter values for each cell face of the specified variable.
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
