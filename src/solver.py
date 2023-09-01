from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from src.anastomosis import solveAnastomosis
from src.boundary_conditions import setInletBC, setOutletBC
from src.junctions import joinVessels
from src.utils import pressureSA, waveSpeedSA
import src.initialise as ini


@jax.jit
def calculateDeltaT(vessels, Ccfl):
    dt = 1.0
    for i in range(len(vessels)):
        Smax = 0.0
        for j in range(ini.VCS[i].M):
            _lambda = jnp.abs(vessels[i].u[j] + vessels[i].c[j])
            Smax = jax.lax.cond(Smax < _lambda, lambda: _lambda, lambda: Smax)
        vessel_dt = ini.VCS[i].dx * Ccfl / Smax
        dt = jax.lax.cond(dt > vessel_dt, lambda: vessel_dt, lambda: dt)
    return dt

#@partial(jax.jit, static_argnums=1)
#def solveModel(vessels, edges, blood, dt, current_time):
#    for j in np.arange(0,edges.edges.shape[0],1):
#        i = edges.edges[j,0]-1
#        vessels[i] = solveVessel(vessels[i], blood, dt, current_time)
#
#        if vessels[i].outlet != "none":
#            vessels[i] = setOutletBC(dt, vessels[i])
#
#        elif edges.inlets[j,0] == 2:
#            d1_i = edges.inlets[j,1]
#            d2_i = edges.inlets[j,2]
#            vessels[i], vessels[d1_i], vessels[d2_i] = joinVessels(blood, vessels[i], vessels[d1_i], vessels[d2_i])
#
#        elif edges.outlets[j,0] == 1:
#            d_i = edges.outlets[j,1]
#            vessels[i], vessels[d_i] = joinVessels(blood, vessels[i], vessels[d_i])
#
#        elif edges.outlets[j,0] == 2:
#            p1_i = edges.outlets[j,1]
#            p2_i = edges.outlets[j,2]
#            d = edges.outlets[j,3]
#            vessels[i], vessels[p1_i], vessels[d] = jax.lax.cond(jnp.maximum(p1_i, p2_i) == i,
#                                                                lambda: solveAnastomosis(vessels[i], vessels[p1_i], vessels[d]),
#                                                                lambda: (vessels[i], vessels[p1_i], vessels[d]))
#                 
#    return vessels

@partial(jax.jit, static_argnums=1)
def solveModel(vessels, edges, blood, dt, current_time):
    for j in np.arange(0,edges.edges.shape[0],1):
        i = edges.edges[j,0]-1
        vessels[i].Q, vessels[i].A, vessels[i].P, vessels[i].c = solveVessel(vessels[i], i, blood, dt, current_time)
        vessels[i].u = vessels[i].Q / vessels[i].A

        if ini.VCS[i].outlet != "none":
            vessels[i] = setOutletBC(dt, vessels[i], i)

        elif edges.inlets[j,0] == 2:
            d1_i = edges.inlets[j,1]
            d2_i = edges.inlets[j,2]
            vessels[i], vessels[d1_i], vessels[d2_i] = joinVessels(blood, vessels[i], vessels[d1_i], vessels[d2_i])

        elif edges.outlets[j,0] == 1:
            d_i = edges.outlets[j,1]
            vessels[i], vessels[d_i] = joinVessels(blood, vessels[i], vessels[d_i])

        elif edges.outlets[j,0] == 2:
            p1_i = edges.outlets[j,1]
            p2_i = edges.outlets[j,2]
            d = edges.outlets[j,3]
            vessels[i], vessels[p1_i], vessels[d] = jax.lax.cond(jnp.maximum(p1_i, p2_i) == i,
                                                                lambda: solveAnastomosis(vessels[i], vessels[p1_i], vessels[d], i, p1_i, d),
                                                                lambda: (vessels[i], vessels[p1_i], vessels[d]))
                 
    return vessels


@partial(jax.jit, static_argnums=1)
def solveVessel(vessel, i, blood, dt, current_time):
    if ini.VCS[i].inlet:
        vessel = setInletBC(i, current_time, dt, vessel)

    return muscl(i, vessel.dU, vessel.U00A, vessel.UM1A, vessel.U00Q, 
                                vessel.UM1Q, vessel.A, vessel.Q, dt,  blood)


#@jax.jit
@partial(jax.jit, static_argnums=(0,8,))
def muscl(i, dU, U00A, UM1A, U00Q, UM1Q, A, Q, dt, b):
    #v = ini.VCS[i]
    M = ini.VCS[i].M
    vA = jnp.zeros(M+2)
    vQ = jnp.zeros(M+2)
    vA = vA.at[0].set(U00A)
    vA = vA.at[-1].set(UM1A)

    vQ = vQ.at[0].set(U00Q)
    vQ = vQ.at[-1].set(UM1Q)

    vA = vA.at[1:M+1].set(A)
    vQ = vQ.at[1:M+1].set(Q)

    slopesA = computeLimiter(M, vA, ini.VCS[i].invDx, dU)
    slopesQ = computeLimiter(M, vQ, ini.VCS[i].invDx, dU)

    slopeA_halfDx = slopesA * ini.VCS[i].halfDx
    slopeQ_halfDx = slopesQ * ini.VCS[i].halfDx

    Al = vA + slopeA_halfDx
    Ar = vA - slopeA_halfDx
    Ql = vQ + slopeQ_halfDx
    Qr = vQ - slopeQ_halfDx

    Fl = computeFlux(ini.VCS[i].gamma_ghost, Al, Ql, M)
    Fr = computeFlux(ini.VCS[i].gamma_ghost, Ar, Qr, M)

    dxDt = jnp.float64(ini.VCS[i].dx) / jnp.float64(dt)
    one = jnp.float64(1.0)
    
    invDxDt = one / jnp.float64(dxDt)

    flux = jnp.zeros((2,M+2))
    flux = flux.at[0,0:M+1].set(0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])))
    flux = flux.at[1,0:M+1].set(0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1])))

    uStar = jnp.zeros((2,M+2))
    uStar = uStar.at[0,1:M+1].set(vA[1:M+1] + invDxDt * (flux[0, 0:M] - flux[0, 1:M+1]))
    uStar = uStar.at[1,1:M+1].set(vQ[1:M+1] + invDxDt * (flux[1, 0:M] - flux[1, 1:M+1]))

    uStar = uStar.at[0,0].set(uStar[0,1])
    uStar = uStar.at[1,0].set(uStar[1,1])
    uStar = uStar.at[0,M+1].set(uStar[0,M])
    uStar = uStar.at[1,M+1].set(uStar[1,M])

    slopesA = computeLimiterIdx(M, uStar, 0, ini.VCS[i].invDx, dU)
    slopesQ = computeLimiterIdx(M, uStar, 1, ini.VCS[i].invDx, dU)

    Al = uStar[0,0:M+2] + slopesA * ini.VCS[i].halfDx
    Ar = uStar[0,0:M+2] - slopesA * ini.VCS[i].halfDx
    Ql = uStar[1,0:M+2] + slopesQ * ini.VCS[i].halfDx
    Qr = uStar[1,0:M+2] - slopesQ * ini.VCS[i].halfDx

    Fl = computeFlux(ini.VCS[i].gamma_ghost, Al, Ql, M)
    Fr = computeFlux(ini.VCS[i].gamma_ghost, Ar, Qr, M)

    flux = jnp.zeros((2,M+2))
    flux = flux.at[0,0:M+1].set(0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])))
    flux = flux.at[1,0:M+1].set(0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1])))

    #jax.debug.breakpoint()
    A = A.at[0:M].set(0.5*(A[0:M] + uStar[0,1:M+1] + invDxDt * (flux[0, 0:M] - flux[0, 1:M+1])))
    Q = Q.at[0:M].set(0.5*(Q[0:M] + uStar[1,1:M+1] + invDxDt * (flux[1, 0:M] - flux[1, 1:M+1])))

    s_A = jnp.sqrt(A)
    Si = - ini.VCS[i].viscT * Q / A - ini.VCS[i].wallE * (s_A - ini.VCS[i].s_A0) * A
    Q = Q + dt * Si

    P = pressureSA(s_A * ini.VCS[i].s_inv_A0, ini.VCS[i].beta, ini.VCS[i].Pext)
    c = waveSpeedSA(s_A, ini.VCS[i].gamma)

    #if (v.wallVa[0] != 0.0).astype(bool):
    #mask = v.wallVa != 0.0
    #    Td = 1.0 / dt + v.wallVb
    #    Tlu = -v.wallVa
    #    T = jax.scipy.linalg.solve_banded((1, 1), jnp.array([Tlu[:-1], Td, Tlu[1:]]), v.Q[mask])

    #    d = (1.0 / dt - v.wallVb) * v.Q[mask]
    #    d = d.at[0].set(v.wallVa[1:-1] * v.Q[mask[:-1]])
    #    d = d.at[-1].set(v.wallVa[1:-1] * v.Q[mask[1:]])
    #    d = d.at[jax.ops.index[1:-1]].set(v.wallVa[1:-1] * v.Q[mask[:-2]] + v.wallVa[:-2] * v.Q[mask[2:]])

    #    v.Q = v.Q.at[mask].set(jax.scipy.linalg.solve_banded((1, 1), jnp.array([Tlu[:-1], Td, Tlu[1:]]), d))

    #u = Q / A

    #jax.debug.breakpoint()
    #jax.debug.print("a")
    return Q, A, P, c

#@jax.jit
#def muscl(v, dt, b):
#    v.vA = v.vA.at[0].set(v.U00A)
#    v.vA = v.vA.at[-1].set(v.UM1A)
#
#    v.vQ = v.vQ.at[0].set(v.U00Q)
#    v.vQ = v.vQ.at[-1].set(v.UM1Q)
#
#    v.vA = v.vA.at[1:v.M+1].set(v.A)
#    v.vQ = v.vQ.at[1:v.M+1].set(v.Q)
#
#    v.slopesA = computeLimiter(v.M, v.vA, v.invDx, v.dU)
#    v.slopesQ = computeLimiter(v.M, v.vQ, v.invDx, v.dU)
#
#    slopeA_halfDx = v.slopesA * v.halfDx
#    slopeQ_halfDX = v.slopesQ * v.halfDx
#
#    v.Al = v.vA + slopeA_halfDx
#    v.Ar = v.vA - slopeA_halfDx
#    v.Ql = v.vQ + slopeQ_halfDX
#    v.Qr = v.vQ - slopeQ_halfDX
#
#    v.Fl = computeFlux(v.gamma_ghost, v.Al, v.Ql, v.Fl)
#    v.Fr = computeFlux(v.gamma_ghost, v.Ar, v.Qr, v.Fr)
#
#    dxDt = jnp.float64(v.dx) / jnp.float64(dt)
#    one = jnp.float64(1.0)
#    
#    invDxDt = one / jnp.float64(dxDt)
#
#    v.flux = jnp.zeros_like(v.flux)
#    v.flux = v.flux.at[0,0:v.M+1].set(0.5 * (v.Fr[0, 1:v.M+2] + v.Fl[0, 0:v.M+1] - dxDt * (v.Ar[1:v.M+2] - v.Al[0:v.M+1])))
#    v.flux = v.flux.at[1,0:v.M+1].set(0.5 * (v.Fr[1, 1:v.M+2] + v.Fl[1, 0:v.M+1] - dxDt * (v.Qr[1:v.M+2] - v.Ql[0:v.M+1])))
#
#    v.uStar = jnp.zeros_like(v.uStar)
#    v.uStar = v.uStar.at[0,1:v.M+1].set(v.vA[1:v.M+1] + invDxDt * (v.flux[0, 0:v.M] - v.flux[0, 1:v.M+1]))
#    v.uStar = v.uStar.at[1,1:v.M+1].set(v.vQ[1:v.M+1] + invDxDt * (v.flux[1, 0:v.M] - v.flux[1, 1:v.M+1]))
#
#    v.uStar = v.uStar.at[0,0].set(v.uStar[0,1])
#    v.uStar = v.uStar.at[1,0].set(v.uStar[1,1])
#    v.uStar = v.uStar.at[0,v.M+1].set(v.uStar[0,v.M])
#    v.uStar = v.uStar.at[1,v.M+1].set(v.uStar[1,v.M])
#
#    v.slopesA = computeLimiterIdx(v.M, v.uStar, 0, v.invDx, v.dU)
#    v.slopesQ = computeLimiterIdx(v.M, v.uStar, 1, v.invDx, v.dU)
#
#    v.Al = v.uStar[0,0:v.M+2] + v.slopesA * v.halfDx
#    v.Ar = v.uStar[0,0:v.M+2] - v.slopesA * v.halfDx
#    v.Ql = v.uStar[1,0:v.M+2] + v.slopesQ * v.halfDx
#    v.Qr = v.uStar[1,0:v.M+2] - v.slopesQ * v.halfDx
#
#    v.Fl = computeFlux(v.gamma_ghost, v.Al, v.Ql, v.Fl)
#    v.Fr = computeFlux(v.gamma_ghost, v.Ar, v.Qr, v.Fr)
#
#    v.flux = jnp.zeros_like(v.flux)
#    v.flux = v.flux.at[0,0:v.M+1].set(0.5 * (v.Fr[0, 1:v.M+2] + v.Fl[0, 0:v.M+1] - dxDt * (v.Ar[1:v.M+2] - v.Al[0:v.M+1])))
#    v.flux = v.flux.at[1,0:v.M+1].set(0.5 * (v.Fr[1, 1:v.M+2] + v.Fl[1, 0:v.M+1] - dxDt * (v.Qr[1:v.M+2] - v.Ql[0:v.M+1])))
#
#    v.A = v.A.at[0:v.M].set(0.5*(v.A[0:v.M] + v.uStar[0,1:v.M+1] + invDxDt * (v.flux[0, 0:v.M] - v.flux[0, 1:v.M+1])))
#    v.Q = v.Q.at[0:v.M].set(0.5*(v.Q[0:v.M] + v.uStar[1,1:v.M+1] + invDxDt * (v.flux[1, 0:v.M] - v.flux[1, 1:v.M+1])))
#
#    s_A = jnp.sqrt(v.A)
#    Si = - v.viscT * v.Q / v.A - v.wallE * (s_A - v.s_A0) * v.A
#    v.Q = v.Q + dt * Si
#
#    v.P = pressureSA(s_A * v.s_inv_A0, v.beta, v.Pext)
#    v.c = waveSpeedSA(s_A, v.gamma)
#
#    #if (v.wallVa[0] != 0.0).astype(bool):
#    #mask = v.wallVa != 0.0
#    #    Td = 1.0 / dt + v.wallVb
#    #    Tlu = -v.wallVa
#    #    T = jax.scipy.linalg.solve_banded((1, 1), jnp.array([Tlu[:-1], Td, Tlu[1:]]), v.Q[mask])
#
#    #    d = (1.0 / dt - v.wallVb) * v.Q[mask]
#    #    d = d.at[0].set(v.wallVa[1:-1] * v.Q[mask[:-1]])
#    #    d = d.at[-1].set(v.wallVa[1:-1] * v.Q[mask[1:]])
#    #    d = d.at[jax.ops.index[1:-1]].set(v.wallVa[1:-1] * v.Q[mask[:-2]] + v.wallVa[:-2] * v.Q[mask[2:]])
#
#    #    v.Q = v.Q.at[mask].set(jax.scipy.linalg.solve_banded((1, 1), jnp.array([Tlu[:-1], Td, Tlu[1:]]), d))
#
#    v.u = v.Q / v.A
#
#    return v

@partial(jax.jit, static_argnums=3)
def computeFlux(gamma_ghost, A, Q, M):
    Flux = jnp.zeros((2,M+2))
    Flux = Flux.at[0,:].set(Q)
    Flux = Flux.at[1,:].set(Q * Q / A + gamma_ghost * A * jnp.sqrt(A))

    return Flux

@jax.jit
def maxMod(a, b):
    return jnp.where(a > b, a, b)

@jax.jit
def minMod(a, b):
    return jnp.where((a <= 0.0) | (b <= 0.0), 0.0, jnp.where(a < b, a, b))

@jax.jit
def superBee(dU):
    s1 = minMod(dU[0, :], 2 * dU[1, :])
    s2 = minMod(2 * dU[0, :], dU[1, :])

    return maxMod(s1, s2)

@partial(jax.jit, static_argnums=0)
def computeLimiter(M, U, invDx, dU):
    dU = dU.at[0, 1:M+1].set((U[1:M+1] - U[:M]) * invDx)
    dU = dU.at[1, 0:M].set(dU[0, 1:M+1])
    
    return superBee(dU)

@partial(jax.jit, static_argnums=0)
def computeLimiterIdx(M, U, idx, invDx, dU):
    U = U[idx, :]
    dU = dU.at[0, 1:M+1].set((U[1:M+1] - U[:M]) * invDx)
    dU = dU.at[1, 0:M].set(dU[0, 1:M+1])
    
    return superBee(dU)