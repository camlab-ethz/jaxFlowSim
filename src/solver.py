from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from src.anastomosis import solveAnastomosis
from src.conjunctions import solveConjunction
from src.bifurcations import solveBifurcation
from src.boundary_conditions import setInletBC, setOutletBC
from src.utils import pressureSA, waveSpeedSA
import src.initialise as ini


@jax.jit
def calculateDeltaT(u, c):
    dt = 1.0
    start = 0
    for i in range(ini.NUM_VESSELS):
        end = (i+1)*ini.MESH_SIZE
        Smax = jnp.max(jnp.abs(u[start:end] + c[start:end]))
        vessel_dt = ini.DXS[i] * ini.CCFL / Smax
        dt = jax.lax.cond(dt > vessel_dt, lambda: vessel_dt, lambda: dt)
        start = end
    return dt



@jax.jit
def solveModel(sim_dat, sim_dat_aux, dt, t):

    def body_fun(j,dat):
        sim_dat, sim_dat_aux, edges, outlet, inlet, input_data, cardiac_T, dx, A0, beta, Pext, gamma, viscT, wallE, Rt, R1, R2, Cc = dat
        i = edges[j,0]-1
        M = ini.MESH_SIZE
        start = i*M
        end = (i+1)*M
        size = input_data.shape[1]
        test = solveVessel(inlet[i], sim_dat[0,start], sim_dat[0,start+1], 
                                 jax.lax.dynamic_slice(sim_dat, (1,start), (1,M)).reshape(M),
                                 jax.lax.dynamic_slice(sim_dat, (2,start), (1,M)).reshape(M),
                                 sim_dat[3,start], sim_dat[3,start+1], 
                                 sim_dat_aux[2,i], 
                                 sim_dat_aux[3,i], 
                                 sim_dat_aux[6,i], 
                                 sim_dat_aux[7,i],
                                 dt, t, jax.lax.dynamic_slice(input_data, (i,0), (2,size)), 
                                 cardiac_T[i], dx[i], 
                                 jax.lax.dynamic_slice(A0, (i,0), (1,M)).reshape(M), beta[i,0], 
                                 Pext[i], jax.lax.dynamic_slice(gamma, (i,0), (1,M)).reshape(M), 
                                 viscT[i], jax.lax.dynamic_slice(wallE, (i,0), (1,M)).reshape(M))
        sim_dat = jax.lax.dynamic_update_slice(sim_dat, test, (0,start))


        
        def setOutletBC_wrapper(sim_dat, sim_dat_aux):
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
            u, Q, A, c, P1, Pc = setOutletBC(outlet[i], u1, u2, Q1, A1, c1, c2, P1, P2, P3, Pc, W1M0, W2M0, dt, dx[i], Rt[i], Cc[i], R1[i], R2[i], beta[i], gamma[i], A0[i,M-1], Pext[i])
            sim_dat = sim_dat.at[0,end-1].set(u)
            sim_dat = sim_dat.at[1,end-1].set(Q)
            sim_dat = sim_dat.at[2,end-1].set(A)
            sim_dat = sim_dat.at[3,end-1].set(c)
            sim_dat = sim_dat.at[4,end-1].set(P1)
            sim_dat_aux = sim_dat_aux.at[10,i].set(Pc)
            return sim_dat, sim_dat_aux

        sim_dat, sim_dat_aux = jax.lax.cond(outlet[i] != 0,lambda x, y: setOutletBC_wrapper(x, y), lambda x, y: (x, y), sim_dat, sim_dat_aux)



        #if edges[j,3] == 2:
        #    d1_i = edges[j,4]
        #    d2_i = edges[j,5]
        #    d1_i_start = mesh_sizes[d1_i]
        #    d2_i_start = mesh_sizes[d2_i]
        #    u1 = sim_dat[0,end-1]
        #    u2 = sim_dat[0,d1_i_start]
        #    u3 = sim_dat[0,d2_i_start]
        #    Q1 = sim_dat[1,end-1]
        #    Q2 = sim_dat[1,d1_i_start]
        #    Q3 = sim_dat[1,d2_i_start]
        #    A1 = sim_dat[2,end-1]
        #    A2 = sim_dat[2,d1_i_start]
        #    A3 = sim_dat[2,d2_i_start]
        #    c1 = sim_dat[3,end-1]
        #    c2 = sim_dat[3,d1_i_start]
        #    c3 = sim_dat[3,d2_i_start]
        #    P1 = sim_dat[4,end-1]
        #    P2 = sim_dat[4,d1_i_start]
        #    P3 = sim_dat[4,d2_i_start]
        #    u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3 = solveBifurcation(i, d1_i, d2_i,
        #                                                                            u1, u2, u3, 
        #                                                                            A1, A2, A3)
        #    sim_dat = sim_dat.at[0,end-1].set(u1) 
        #    sim_dat = sim_dat.at[0,d1_i_start].set(u2)    
        #    sim_dat = sim_dat.at[0,d2_i_start].set(u3)
        #    sim_dat = sim_dat.at[1,end-1].set(Q1)
        #    sim_dat = sim_dat.at[1,d1_i_start].set(Q2)
        #    sim_dat = sim_dat.at[1,d2_i_start].set(Q3)
        #    sim_dat = sim_dat.at[2,end-1].set(A1)
        #    sim_dat = sim_dat.at[2,d1_i_start].set(A2)
        #    sim_dat = sim_dat.at[2,d2_i_start].set(A3)
        #    sim_dat = sim_dat.at[3,end-1].set(c1)
        #    sim_dat = sim_dat.at[3,d1_i_start].set(c2)
        #    sim_dat = sim_dat.at[3,d2_i_start].set(c3)
        #    sim_dat = sim_dat.at[4,end-1].set(P1)
        #    sim_dat = sim_dat.at[4,d1_i_start].set(P2)
        #    sim_dat = sim_dat.at[4,d2_i_start].set(P3)

        #elif edges[j,6] == 1:
        #    d_i = edges[j,7]
        #    d_i_start = mesh_sizes[d_i]
        #    u1 = sim_dat[0,end-1]
        #    u2 = sim_dat[0,d_i_start]
        #    A1 = sim_dat[2,end-1]
        #    A2 = sim_dat[2,d_i_start]
        #    u1, u2, Q1, Q2, A1, A2, c1, c2, P1, P2, = solveConjunction(i, d_i,
        #                                                                u1, u2, 
        #                                                                A1, A2)
        #    sim_dat = sim_dat.at[0,end-1].set(u1)
        #    sim_dat = sim_dat.at[0,d_i_start].set(u2)
        #    sim_dat = sim_dat.at[1,end-1].set(Q1)
        #    sim_dat = sim_dat.at[1,d_i_start].set(Q2)
        #    sim_dat = sim_dat.at[2,end-1].set(A1)
        #    sim_dat = sim_dat.at[2,d_i_start].set(A2)
        #    sim_dat = sim_dat.at[3,end-1].set(c1)
        #    sim_dat = sim_dat.at[3,d_i_start].set(c2)
        #    sim_dat = sim_dat.at[4,end-1].set(P1)
        #    sim_dat = sim_dat.at[4,d_i_start].set(P2)
        #    #jax.debug.breakpoint()

        #elif edges[j,6] == 2:                                           
        #    p1_i = edges[j,7]
        #    p2_i = edges[j,8]
        #    d = edges[j,9]
        #    p1_i_end = mesh_sizes[p1_i+1]
        #    d_start = mesh_sizes[d]
        #    u1 = sim_dat[0,end-1]
        #    u2 = sim_dat[0,p1_i_end-1]
        #    u3 = sim_dat[0,d_start]
        #    Q1 = sim_dat[1,end-1]
        #    Q2 = sim_dat[1,p1_i_end-1]
        #    Q3 = sim_dat[1,d_start]
        #    A1 = sim_dat[2,end-1]
        #    A2 = sim_dat[2,p1_i_end-1]
        #    A3 = sim_dat[2,d_start]
        #    c1 = sim_dat[3,end-1]
        #    c2 = sim_dat[3,p1_i_end-1]
        #    c3 = sim_dat[3,d_start]
        #    P1 = sim_dat[4,end-1]
        #    P2 = sim_dat[4,p1_i_end-1]
        #    P3 = sim_dat[4,d_start]
        #    u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3 = jax.lax.cond(
        #        jnp.maximum(p1_i, p2_i) == i, 
        #        lambda: solveAnastomosis(i, p1_i, d,
        #                                 u1, u2, u3, 
        #                                 A1, A2, A3,
        #                                ), 
        #        lambda: (u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3))
        #    sim_dat = sim_dat.at[0,end-1].set(u1)
        #    sim_dat = sim_dat.at[0,p1_i_end-1].set(u2)
        #    sim_dat = sim_dat.at[0,d_start].set(u3)
        #    sim_dat = sim_dat.at[1,end-1].set(Q1)
        #    sim_dat = sim_dat.at[1,p1_i_end-1].set(Q2)
        #    sim_dat = sim_dat.at[1,d_start].set(Q3)
        #    sim_dat = sim_dat.at[2,end-1].set(A1)
        #    sim_dat = sim_dat.at[2,p1_i_end-1].set(A2)
        #    sim_dat = sim_dat.at[2,d_start].set(A3)
        #    sim_dat = sim_dat.at[3,end-1].set(c1)
        #    sim_dat = sim_dat.at[3,p1_i_end-1].set(c2)
        #    sim_dat = sim_dat.at[3,d_start].set(c3)
        #    sim_dat = sim_dat.at[4,end-1].set(P1)
        #    sim_dat = sim_dat.at[4,p1_i_end-1].set(P2)
        #    sim_dat = sim_dat.at[4,d_start].set(P3)
        
        return sim_dat, sim_dat_aux, edges, outlet, inlet, input_data, cardiac_T, dx, A0, beta, Pext, gamma, viscT, wallE, Rt, R1, R2, Cc

    #for j in np.arange(0,,1):
    
    #def cond_fun(dat):
    #    _, _, j = dat
    #    return j < ini.EDGES.edges.shape[0]


    sim_dat, sim_dat_aux, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,  = jax.lax.fori_loop(0, ini.NUM_VESSELS, 
                      body_fun, (sim_dat, sim_dat_aux, ini.EDGES, 
                                 ini.OUTLET_TYPES, ini.INLET_TYPES, ini.INPUT_DATAS, ini.CARDIAC_TS, 
                                 ini.DXS, 
                                 ini.A0S, ini.BETAS, 
                                 ini.PEXTS, ini.GAMMAS, ini.VISCTS, ini.WALLES, 
                                 ini.RTS, ini.R1S, ini.R2S, ini.CCS))

    
    return sim_dat, sim_dat_aux


#@partial(jax.jit, static_argnums=0)
@jax.jit
def solveVessel(inlet, u0, u1, Q, A, 
                c0, c1, U00Q, U00A, UM1Q, UM1A, 
                dt, t, input_data, cardiac_T, 
                dx, A0, beta, Pext,
                gamma, viscT, wallE):
    _Q, _A = jax.lax.cond(inlet > 0, lambda: setInletBC(inlet, u0, u1, A[0], c0, c1, t, dt, input_data, cardiac_T, 1/dx, A0[0], beta, Pext), lambda: (Q[0],A[0]))
    Q = Q.at[0].set(_Q)
    A = A.at[0].set(_A)

    #if inlet > 0:
    #    Q0, A0 = setInletBC(i, u0, u1, A[0], c0, c1, t, dt)
    #    Q = Q.at[0].set(Q0)
    #    A = A.at[0].set(A0)

    return muscl(U00Q, U00A, 
                UM1Q, UM1A, Q.transpose(), A.transpose(), A0,
                dt, dx, beta, Pext, gamma, viscT, wallE)

#@partial(jax.jit, static_argnums=(0,))
@jax.jit
def muscl(U00Q, U00A, UM1Q, UM1A, Q, A, A0, dt, dx, beta, Pext, gamma, viscT, wallE):
    M = 242
    s_A0 = jnp.sqrt(A0)
    s_inv_A0 = 1/s_A0
    halfDx = 0.5*dx
    invDx = 1/dx
    gamma_ghost = jnp.zeros(M+2)
    gamma_ghost = gamma_ghost.at[1:M+1].set(gamma)
    gamma_ghost = gamma_ghost.at[0].set(gamma[0])
    gamma_ghost = gamma_ghost.at[-1].set(gamma[-1])
    vA = jnp.empty(M+2, dtype=jnp.float64)
    vQ = jnp.empty(M+2, dtype=jnp.float64)
    vA = vA.at[0].set(U00A)
    vA = vA.at[-1].set(UM1A)

    vQ = vQ.at[0].set(U00Q)
    vQ = vQ.at[-1].set(UM1Q)
    vA = vA.at[1:M+1].set(A)
    vQ = vQ.at[1:M+1].set(Q)
    #vA = jnp.concatenate((jnp.array([U00A],dtype=jnp.float64),A,jnp.array([UM1A],dtype=jnp.float64)))
    #vQ = jnp.concatenate((jnp.array([U00Q],dtype=jnp.float64),Q,jnp.array([UM1Q],dtype=jnp.float64)))

    slopeA_halfDx = computeLimiter(vA, invDx) * halfDx
    slopeQ_halfDx = computeLimiter(vQ, invDx) * halfDx

    #slopeA_halfDx = slopesA * ini.VCS[i].halfDx
    #slopeQ_halfDx = slopesQ * ini.VCS[i].halfDx

    Al = vA + slopeA_halfDx
    Ar = vA - slopeA_halfDx
    Ql = vQ + slopeQ_halfDx
    Qr = vQ - slopeQ_halfDx

    Fl = computeFlux(gamma_ghost, Al, Ql)
    Fr = computeFlux(gamma_ghost, Ar, Qr)

    dxDt = dx / dt
    
    invDxDt = dt / dx

    flux = jnp.empty((2,M+2), dtype=jnp.float64)
    flux = flux.at[0,0:M+1].set(0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])))
    flux = flux.at[1,0:M+1].set(0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1])))
    #flux = jnp.stack((0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])), 
    #                  0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1]))), dtype=jnp.float64)

    #uStar = jnp.empty((2,M+2), dtype=jnp.float64)
    #uStar = uStar.at[0,1:M+1].set(vA[1:M+1] - invDxDt * jnp.diff(flux[0,0:M+1]))
    #uStar = uStar.at[1,1:M+1].set(vQ[1:M+1] - invDxDt * jnp.diff(flux[1,0:M+1]))
    uStar1 = vA[1:M+1] - invDxDt * jnp.diff(flux[0,0:M+1])
    uStar2 = vQ[1:M+1] - invDxDt * jnp.diff(flux[1,0:M+1])
    uStar = jnp.stack((jnp.concatenate((jnp.array([uStar1[0]],dtype=jnp.float64),uStar1,jnp.array([uStar1[-1]],dtype=jnp.float64))), 
                       jnp.concatenate((jnp.array([uStar2[0]],dtype=jnp.float64),uStar2,jnp.array([uStar2[-1]],dtype=jnp.float64)))), dtype=jnp.float64)


    #uStar = uStar.at[0,0].set(uStar[0,1])
    #uStar = uStar.at[1,0].set(uStar[1,1])
    #uStar = uStar.at[0,M+1].set(uStar[0,M])
    #uStar = uStar.at[1,M+1].set(uStar[1,M])

    slopesA = computeLimiterIdx(uStar, 0, invDx) * halfDx
    slopesQ = computeLimiterIdx(uStar, 1, invDx) * halfDx

    Al = uStar[0,0:M+2] + slopesA
    Ar = uStar[0,0:M+2] - slopesA
    Ql = uStar[1,0:M+2] + slopesQ
    Qr = uStar[1,0:M+2] - slopesQ
    
    #Fl = jax.pmap(lambda A, Q: computeFlux_par(ini.VCS[i].gamma_ghost,A,Q))(Al, Ql)
    #Fr = jax.pmap(lambda A, Q: computeFlux_par(ini.VCS[i].gamma_ghost,A,Q))(Ar, Qr)
    
    #jax.debug.print("{x}", x = Fl)
    Fl = computeFlux(gamma_ghost, Al, Ql)
    Fr = computeFlux(gamma_ghost, Ar, Qr)

    flux = jnp.empty((2,M+2), dtype=jnp.float64)
    flux = flux.at[0,0:M+1].set(0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])))
    flux = flux.at[1,0:M+1].set(0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1])))
    #flux = jnp.stack((0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])), 
    #                 0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1]))), dtype=jnp.float64)


    #A = A.at[0:M].set(0.5*(A[0:M] + uStar[0,1:M+1] + invDxDt * (flux[0, 0:M] - flux[0, 1:M+1])))
    A = A.at[0:M].set(0.5*(A[0:M] + uStar[0,1:M+1] - invDxDt * jnp.diff(flux[0,0:M+1])))
    Q = Q.at[0:M].set(0.5*(Q[0:M] + uStar[1,1:M+1] - invDxDt * jnp.diff(flux[1,0:M+1])))

    s_A = jnp.sqrt(A)
    #Si = - ini.VCS[i].viscT * Q / A - ini.VCS[i].wallE * (s_A - ini.VCS[i].s_A0) * A
    Q = Q - dt * (viscT * Q / A + wallE * (s_A - s_A0) * A)

    P = pressureSA(s_A * s_inv_A0, beta, Pext)
    c = waveSpeedSA(s_A, gamma)

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

    u = Q/A
    return jnp.stack((u, Q, A, c, P ))


@jax.jit
def computeFlux(gamma_ghost, A, Q):
    #Flux = jnp.empty((2,A.size), dtype=jnp.float64)
    #Flux = Flux.at[0,:].set(Q)
    #Flux = Flux.at[1,:].set(Q * Q / A + gamma_ghost * A * jnp.sqrt(A))

    #return Flux
    return jnp.stack((Q, Q * Q / A + gamma_ghost * A * jnp.sqrt(A)), dtype=jnp.float64)

def computeFlux_par(gamma_ghost, A, Q):
    #Flux = jnp.empty((2,A.size), dtype=jnp.float64)
    #Flux = Flux.at[0,:].set(Q)
    #Flux = Flux.at[1,:].set(Q * Q / A + gamma_ghost * A * jnp.sqrt(A))

    #return Flux
    return Q, Q * Q / A + gamma_ghost * A * jnp.sqrt(A)


@jax.jit
def maxMod(a, b):
    return jnp.where(a > b, a, b)

@jax.jit
def minMod(a, b):
    return jnp.where((a <= 0.0) | (b <= 0.0), 0.0, jnp.where(a < b, a, b))

@jax.jit
def superBee(dU):
    #s1 = minMod(dU[0, :], 2 * dU[1, :])
    #s2 = minMod(2 * dU[0, :], dU[1, :])

    #return maxMod(s1, s2)
    return maxMod(minMod(dU[0, :], 2 * dU[1, :]), minMod(2 * dU[0, :], dU[1, :]))

@jax.jit
def computeLimiter(U, invDx):
    #dU = jnp.empty((2, U.size), dtype=jnp.float64)
    #dU = dU.at[0, 1:].set((U[1:] - U[:-1]) * invDx)
    #dU = dU.at[1, 0:-1].set(dU[0, 1:])
    dU = jnp.diff(U) * invDx
    #test = [[0,(U[1:] - U[:-1]) * invDx], 
    #       [0, (U[1:-1] - U[:-2]) * invDx, 0]]
    #jax.debug.breakpoint()
    return superBee(jnp.stack((jnp.concatenate((jnp.array([0.0]),dU), dtype=jnp.float64),jnp.concatenate((dU,jnp.array([0.0])), dtype=jnp.float64)), dtype=jnp.float64))
                                                                   


@jax.jit
def computeLimiterIdx(U, idx, invDx):
    #U = U[idx, :]
    dU = jnp.diff(U[idx, :]) * invDx
    #dU = jnp.empty((2, U.size), dtype=jnp.float64)
    #dU = dU.at[0, 1:].set((U[1:] - U[:-1]) * invDx)
    #dU = dU.at[1, 0:-1].set(dU[0, 1:])
    
    #return superBee(dU)
    return superBee(jnp.stack((jnp.concatenate((jnp.array([0.0]),dU), dtype=jnp.float64),jnp.concatenate((dU,jnp.array([0.0])), dtype=jnp.float64)), dtype=jnp.float64))