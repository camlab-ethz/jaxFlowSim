from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from src.anastomosis import solveAnastomosis
from src.conjunctions import solveConjunction
from src.boundary_conditions import setInletBC, setOutletBC
from src.junctions import joinVessels
from src.utils import pressureSA, waveSpeedSA
import src.initialise as ini


@jax.jit
def calculateDeltaT(u, c):
    dt = 1.0
    for i in range(ini.NUM_VESSELS):
        Smax = 0.0
        _lambda = jnp.abs(u[ini.MESH_SIZES[i]:ini.MESH_SIZES[i+1]] + c[ini.MESH_SIZES[i]:ini.MESH_SIZES[i+1]])
        Smax = jnp.max(_lambda)
        vessel_dt = ini.VCS[i].dx * ini.CCFL / Smax
        dt = jax.lax.cond(dt > vessel_dt, lambda: vessel_dt, lambda: dt)
    return dt



@jax.jit
def solveModel(sim_dat, sim_dat_aux, dt, t):
    for j in np.arange(0,ini.EDGES.edges.shape[0],1):
        i = ini.EDGES.edges[j,0]-1
        start = ini.MESH_SIZES[i]
        end = ini.MESH_SIZES[i+1]
        sim_dat = sim_dat.at[1:,start:end].set(solveVessel(i, sim_dat[0,start], sim_dat[0,start+1], 
                                 sim_dat[1,start:end], 
                                 sim_dat[2,start:end], 
                                 sim_dat[3,start], sim_dat[3,start+1], 
                                 sim_dat_aux[2,i], 
                                 sim_dat_aux[3,i], 
                                 sim_dat_aux[6,i], 
                                 sim_dat_aux[7,i],
                                 dt, t))
        sim_dat = sim_dat.at[0,start:end].set(sim_dat[1,start:end]/sim_dat[2,start:end])


        if ini.VCS[i].outlet != "none":
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,end-2]
            Q1 = sim_dat[1,end-1]
            A1 = sim_dat[2,end-1]
            c1 = sim_dat[3,end-1]
            c2 = sim_dat[3,end-2]
            P1 = sim_dat[4,end-1]
            P2 = sim_dat[4,end-2]
            P3 = sim_dat[4,end-3]
            Pc = sim_dat_aux[10,i]
            W1M0 = sim_dat_aux[0,i]
            W2M0 = sim_dat_aux[1,i]
            u, Q, A, c, P1, Pc = setOutletBC(i, u1, u2, Q1, A1, c1, c2, P1, P2, P3, Pc, W1M0, W2M0, dt)
            sim_dat = sim_dat.at[0,end-1].set(u)
            sim_dat = sim_dat.at[1,end-1].set(Q)
            sim_dat = sim_dat.at[2,end-1].set(A)
            sim_dat = sim_dat.at[3,end-1].set(c)
            sim_dat = sim_dat.at[4,end-1].set(P1)
            sim_dat_aux = sim_dat_aux.at[10,i].set(Pc)



        #elif ini.EDGES.inlets[j,0] == 2:
        #    d1_i = ini.EDGES.inlets[j,1]
        #    d2_i = ini.EDGES.inlets[j,2]
        #    vessels[i], vessels[d1_i], vessels[d2_i] = joinVessels(vessels[i], vessels[d1_i], vessels[d2_i])

        elif ini.EDGES.outlets[j,0] == 1:
            d_i = ini.EDGES.outlets[j,1]
            d_i_start = ini.MESH_SIZES[d_i]
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,d_i_start]
            Q1 = sim_dat[1,end-1]
            Q2 = sim_dat[1,d_i_start]
            A1 = sim_dat[2,end-1]
            A2 = sim_dat[2,d_i_start]
            c1 = sim_dat[3,end-1]
            c2 = sim_dat[3,d_i_start]
            P1 = sim_dat[4,end-1]
            P2 = sim_dat[4,d_i_start]
            u1, u2, Q1, Q2, A1, A2, c1, c2, P1, P2, = solveConjunction(i, d_i,
                                                                        u1, u2, 
                                                                        A1, A2)
            sim_dat = sim_dat.at[0,end-1].set(u1)
            sim_dat = sim_dat.at[0,d_i_start].set(u2)
            sim_dat = sim_dat.at[1,end-1].set(Q1)
            sim_dat = sim_dat.at[1,d_i_start].set(Q2)
            sim_dat = sim_dat.at[2,end-1].set(A1)
            sim_dat = sim_dat.at[2,d_i_start].set(A2)
            sim_dat = sim_dat.at[3,end-1].set(c1)
            sim_dat = sim_dat.at[3,d_i_start].set(c2)
            sim_dat = sim_dat.at[4,end-1].set(P1)
            sim_dat = sim_dat.at[4,d_i_start].set(P2)
            #jax.debug.breakpoint()

        elif ini.EDGES.outlets[j,0] == 2:                                           
            p1_i = ini.EDGES.outlets[j,1]
            p2_i = ini.EDGES.outlets[j,2]
            d = ini.EDGES.outlets[j,3]
            p1_i_end = ini.MESH_SIZES[p1_i+1]
            d_start = ini.MESH_SIZES[d]
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,p1_i_end-1]
            u3 = sim_dat[0,d_start]
            Q1 = sim_dat[1,end-1]
            Q2 = sim_dat[1,p1_i_end-1]
            Q3 = sim_dat[1,d_start]
            A1 = sim_dat[2,end-1]
            A2 = sim_dat[2,p1_i_end-1]
            A3 = sim_dat[2,d_start]
            c1 = sim_dat[3,end-1]
            c2 = sim_dat[3,p1_i_end-1]
            c3 = sim_dat[3,d_start]
            P1 = sim_dat[4,end-1]
            P2 = sim_dat[4,p1_i_end-1]
            P3 = sim_dat[4,d_start]
            u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3 = jax.lax.cond(
                jnp.maximum(p1_i, p2_i) == i, 
                lambda: solveAnastomosis(i, p1_i, d,
                                         u1, u2, u3, 
                                         A1, A2, A3,
                                        ), 
                lambda: (u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3))
            sim_dat = sim_dat.at[0,end-1].set(u1)
            sim_dat = sim_dat.at[0,p1_i_end-1].set(u2)
            sim_dat = sim_dat.at[0,d_start].set(u3)
            sim_dat = sim_dat.at[1,end-1].set(Q1)
            sim_dat = sim_dat.at[1,p1_i_end-1].set(Q2)
            sim_dat = sim_dat.at[1,d_start].set(Q3)
            sim_dat = sim_dat.at[2,end-1].set(A1)
            sim_dat = sim_dat.at[2,p1_i_end-1].set(A2)
            sim_dat = sim_dat.at[2,d_start].set(A3)
            sim_dat = sim_dat.at[3,end-1].set(c1)
            sim_dat = sim_dat.at[3,p1_i_end-1].set(c2)
            sim_dat = sim_dat.at[3,d_start].set(c3)
            sim_dat = sim_dat.at[4,end-1].set(P1)
            sim_dat = sim_dat.at[4,p1_i_end-1].set(P2)
            sim_dat = sim_dat.at[4,d_start].set(P3)
    
    return sim_dat, sim_dat_aux


@partial(jax.jit, static_argnums=0)
def solveVessel(i,u0, u1, Q, A, c0, c1, U00Q, U00A, UM1Q, UM1A, dt, t):
    if ini.VCS[i].inlet:
        Q0, A0 = setInletBC(i, u0, u1, A[0], c0, c1, t, dt)
        Q = Q.at[0].set(Q0)
        A = A.at[0].set(A0)

    return muscl(i, U00Q, U00A, 
                UM1Q, UM1A, Q, A, dt)


@partial(jax.jit, static_argnums=(0,))
def muscl(i, U00Q, U00A, UM1Q, UM1A, Q, A, dt):
    M = ini.VCS[i].M
    vA = jnp.empty(M+2, dtype=jnp.float64)
    vQ = jnp.empty(M+2, dtype=jnp.float64)
    vA = vA.at[0].set(U00A)
    vA = vA.at[-1].set(UM1A)

    vQ = vQ.at[0].set(U00Q)
    vQ = vQ.at[-1].set(UM1Q)

    vA = vA.at[1:M+1].set(A)
    vQ = vQ.at[1:M+1].set(Q)

    slopesA = computeLimiter(vA, ini.VCS[i].invDx)
    slopesQ = computeLimiter(vQ, ini.VCS[i].invDx)

    slopeA_halfDx = slopesA * ini.VCS[i].halfDx
    slopeQ_halfDx = slopesQ * ini.VCS[i].halfDx

    Al = vA + slopeA_halfDx
    Ar = vA - slopeA_halfDx
    Ql = vQ + slopeQ_halfDx
    Qr = vQ - slopeQ_halfDx

    Fl = computeFlux(ini.VCS[i].gamma_ghost, Al, Ql)
    Fr = computeFlux(ini.VCS[i].gamma_ghost, Ar, Qr)

    dxDt = jnp.float64(ini.VCS[i].dx) / jnp.float64(dt)
    one = jnp.float64(1.0)
    
    invDxDt = one / jnp.float64(dxDt)

    flux = jnp.empty((2,M+2), dtype=jnp.float64)
    flux = flux.at[0,0:M+1].set(0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])))
    flux = flux.at[1,0:M+1].set(0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1])))

    uStar = jnp.empty((2,M+2), dtype=jnp.float64)
    uStar = uStar.at[0,1:M+1].set(vA[1:M+1] + invDxDt * (flux[0, 0:M] - flux[0, 1:M+1]))
    uStar = uStar.at[1,1:M+1].set(vQ[1:M+1] + invDxDt * (flux[1, 0:M] - flux[1, 1:M+1]))

    uStar = uStar.at[0,0].set(uStar[0,1])
    uStar = uStar.at[1,0].set(uStar[1,1])
    uStar = uStar.at[0,M+1].set(uStar[0,M])
    uStar = uStar.at[1,M+1].set(uStar[1,M])

    slopesA = computeLimiterIdx(uStar, 0, ini.VCS[i].invDx)
    slopesQ = computeLimiterIdx(uStar, 1, ini.VCS[i].invDx)

    Al = uStar[0,0:M+2] + slopesA * ini.VCS[i].halfDx
    Ar = uStar[0,0:M+2] - slopesA * ini.VCS[i].halfDx
    Ql = uStar[1,0:M+2] + slopesQ * ini.VCS[i].halfDx
    Qr = uStar[1,0:M+2] - slopesQ * ini.VCS[i].halfDx

    Fl = computeFlux(ini.VCS[i].gamma_ghost, Al, Ql)
    Fr = computeFlux(ini.VCS[i].gamma_ghost, Ar, Qr)

    flux = jnp.empty((2,M+2), dtype=jnp.float64)
    flux = flux.at[0,0:M+1].set(0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])))
    flux = flux.at[1,0:M+1].set(0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1])))

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


    return Q, A, c, P 


@jax.jit
def computeFlux(gamma_ghost, A, Q):
    Flux = jnp.empty((2,A.size), dtype=jnp.float64)
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

@jax.jit
def computeLimiter(U, invDx):
    dU = jnp.empty((2, U.size), dtype=jnp.float64)
    dU = dU.at[0, 1:].set((U[1:] - U[:-1]) * invDx)
    dU = dU.at[1, 0:-1].set(dU[0, 1:])
    
    return superBee(dU)

@jax.jit
def computeLimiterIdx(U, idx, invDx):
    U = U[idx, :]
    dU = jnp.empty((2, U.size), dtype=jnp.float64)
    dU = dU.at[0, 1:].set((U[1:] - U[:-1]) * invDx)
    dU = dU.at[1, 0:-1].set(dU[0, 1:])
    
    return superBee(dU)