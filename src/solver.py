import jax.numpy as jnp
from functools import partial
from jax import lax, vmap, jit
from src.anastomosis import solveAnastomosis
from src.conjunctions import solveConjunction
from src.bifurcations import solveBifurcation
from src.boundary_conditions import setInletBC, setOutletBC
from src.utils import pressureSA, waveSpeedSA

def computeDt(Ccfl, u, c, dx):
    Smax = vmap(lambda a, b: jnp.abs(a+b))(u,c)
    vessel_dt = vmap(lambda a, b: a*Ccfl/b)(dx,Smax)
    dt = jnp.min(vessel_dt)
    return dt

@partial(jit, static_argnums=(0, 1))
def solveModel(N, B, starts, ends, indices1, indices2, t, dt, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, input_data, rho):

    inlet = sim_dat_const_aux[0,1] 
    u0 = sim_dat[0,B]
    u1 = sim_dat[0,B+1]
    A0 = sim_dat[2,B]
    c0 = sim_dat[3,B]
    c1 = sim_dat[3,B+1]
    cardiac_T = sim_dat_const_aux[0,0]
    dx = sim_dat_const[-1,B]
    A00 = sim_dat_const[0,B]
    beta0 = sim_dat_const[1,B]
    Pext = sim_dat_const[4,B]

    sim_dat = sim_dat.at[1:3,0:B+1].set(jnp.array(setInletBC(inlet, u0, u1, A0, 
                        c0, c1, t, dt, 
                        input_data, cardiac_T, 1/dx, A00, 
                        beta0, Pext))[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:])

    sim_dat = sim_dat.at[:,B:-B].set(muscl(dt, 
                  sim_dat[1,B:-B],
                  sim_dat[2,B:-B], 
                  sim_dat_const[0,B:-B], 
                  sim_dat_const[1,B:-B], 
                  sim_dat_const[2,B:-B], 
                  sim_dat_const[3,B:-B],
                  sim_dat_const[-1,B:-B],
                  sim_dat_const[4,B:-B],
                  sim_dat_const[5,B:-B],
                  indices1, indices2))

    def bodyFun(j, dat):
        (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, rho, starts, ends) = dat
        end = ends[j]

        def setOutletBCWrapper(sim_dat, sim_dat_aux):
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,end-2]
            Q1 = sim_dat[1,end-1]
            A1 = sim_dat[2,end-1]
            c1 = sim_dat[3,end-1]
            c2 = sim_dat[3,end-2]
            P1 = sim_dat[4,end-1]
            P2 = sim_dat[4,end-2]
            P3 = sim_dat[4,end-3]
            Pc = sim_dat_aux[j,2]
            W1M0 = sim_dat_aux[j,0]
            W2M0 = sim_dat_aux[j,1]
            u, Q, A, c, P1, Pc = setOutletBC(dt,
                                             u1, u2, Q1, A1, c1, c2, 
                                             P1, P2, P3, Pc, W1M0, W2M0,
                                             sim_dat_const[0,end-1],
                                             sim_dat_const[1,end-1],
                                             sim_dat_const[2,end-1],
                                             sim_dat_const[-1, end-1],
                                             sim_dat_const[4, end-1],
                                             sim_dat_const_aux[j, 2], 
                                             sim_dat_const[6, end-1],
                                             sim_dat_const[7, end-1],
                                             sim_dat_const[8, end-1],
                                             sim_dat_const[9, end-1])
            temp = jnp.array((u,Q,A,c,P1))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,end-1))
            sim_dat_aux = sim_dat_aux.at[j,2].set(Pc)
            return sim_dat, sim_dat_aux

        (sim_dat, 
         sim_dat_aux) = lax.cond(sim_dat_const_aux[j,2] != 0,
                                    lambda x, y: setOutletBCWrapper(x,y), 
                                    lambda x, y: (x,y), sim_dat, sim_dat_aux)

        def solveBifurcationWrapper(sim_dat):
            d1_i = edges[j,4]
            d2_i = edges[j,5]
            d1_i_start = starts[d1_i] 
            d2_i_start = starts[d2_i] 
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,d1_i_start]
            u3 = sim_dat[0,d2_i_start]
            A1 = sim_dat[2,end-1]
            A2 = sim_dat[2,d1_i_start]
            A3 = sim_dat[2,d2_i_start]
            (u1, u2, u3, 
             Q1, Q2, Q3, 
             A1, A2, A3, 
             c1, c2, c3, 
             P1, P2, P3) = solveBifurcation(u1, u2, u3, 
                                            A1, A2, A3,
                                            sim_dat_const[0,end-1],
                                            sim_dat_const[0,d1_i_start],
                                            sim_dat_const[0,d2_i_start],
                                            sim_dat_const[1,end-1],
                                            sim_dat_const[1,d1_i_start],
                                            sim_dat_const[1,d2_i_start],
                                            sim_dat_const[2,end-1],
                                            sim_dat_const[2,d1_i_start],
                                            sim_dat_const[2,d2_i_start],
                                            sim_dat_const[4, end-1],
                                            sim_dat_const[4, d1_i_start],
                                            sim_dat_const[4, d2_i_start],
                                            )
            temp1 = jnp.array((u1, Q1, A1, c1, P1))
            temp2 = jnp.array((u2, Q2, A2, c2, P2))
            temp3 = jnp.array((u3, Q3, A3, c3, P3))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp1[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,end-1))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp2[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,d1_i_start-B))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp3[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,d2_i_start-B))
            return sim_dat

        sim_dat = lax.cond((sim_dat_const_aux[j,2] == 0) * (edges[j,3] == 2),
                                    lambda x: solveBifurcationWrapper(x), 
                                    lambda x: x, sim_dat)

        def solveConjunctionWrapper(sim_dat, rho):
            d_i = edges[j,7]
            d_i_start = starts[d_i]
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,d_i_start]
            A1 = sim_dat[2,end-1]
            A2 = sim_dat[2,d_i_start]
            (u1, u2, Q1, Q2, 
             A1, A2, c1, c2, P1, P2) = solveConjunction(u1, u2, 
                                                        A1, A2,
                                                        sim_dat_const[0,end-1],
                                                        sim_dat_const[0,d_i_start],
                                                        sim_dat_const[1,end-1],
                                                        sim_dat_const[1,d_i_start],
                                                        sim_dat_const[2,end-1],
                                                        sim_dat_const[2,d_i_start],
                                                        sim_dat_const[4, end-1],
                                                        sim_dat_const[4, d_i_start],
                                                        rho)
            temp1 = jnp.array((u1, Q1, A1, c1, P1))
            temp2 = jnp.array((u2, Q2, A2, c2, P2))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp1[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,end-1))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp2[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,d_i_start-B))
            return sim_dat

        sim_dat = lax.cond((sim_dat_const_aux[j,2] == 0) * 
                               (edges[j,3] != 2) *
                               (edges[j,6] == 1),
                                lambda x, y: solveConjunctionWrapper(x, y), 
                                lambda x, y: x, sim_dat, rho)

        def solveAnastomosisWrapper(sim_dat):
            p1_i = edges[j,7]
            p2_i = edges[j,8]
            d = edges[j,9]
            p1_i_end = ends[p1_i]
            d_start = starts[d]
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
            u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3 = lax.cond(
                jnp.maximum(p1_i, p2_i) == j, 
                lambda: solveAnastomosis(u1, u2, u3, 
                                         A1, A2, A3,
                                         sim_dat_const[0,end-1],
                                         sim_dat_const[0,p1_i_end-1],
                                         sim_dat_const[0,d_start],
                                         sim_dat_const[1,end-1],
                                         sim_dat_const[1,p1_i_end-1],
                                         sim_dat_const[1,d_start],
                                         sim_dat_const[2,end-1],
                                         sim_dat_const[2,p1_i_end-1],
                                         sim_dat_const[2,d_start],
                                         sim_dat_const[4,end-1],
                                         sim_dat_const[4,p1_i_end-1],
                                         sim_dat_const[4,d_start],
                                        ), 
                lambda: (u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3))
            temp1 = jnp.array((u1, Q1, A1, c1, P1))
            temp2 = jnp.array((u2, Q2, A2, c2, P2))
            temp3 = jnp.array((u3, Q3, A3, c3, P3))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp1[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,end-1))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp2[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,p1_i_end-1))
            sim_dat = lax.dynamic_update_slice( 
                sim_dat, 
                temp3[:,jnp.newaxis]*jnp.ones(B+1)[jnp.newaxis,:],
                (0,d_start-B))
            return sim_dat
        
        sim_dat = lax.cond((sim_dat_const_aux[j,2] == 0) * 
                               (edges[j,3] != 2) *
                               (edges[j,6] == 2),
                                lambda x: solveAnastomosisWrapper(x), 
                                lambda x: x, sim_dat)


        return (sim_dat, sim_dat_aux, 
                sim_dat_const, sim_dat_const_aux, 
                edges, rho, starts, ends)

    (sim_dat, sim_dat_aux, _, 
     _, _, _, 
     _, _)  = lax.fori_loop(0, N, bodyFun, (sim_dat, sim_dat_aux, sim_dat_const, 
                                            sim_dat_const_aux, edges, rho, 
                                            starts, ends))

    return sim_dat, sim_dat_aux

def muscl(dt, 
          Q, A, 
          A0, beta,  gamma, wallE,
          dx, Pext,viscT,
          indices1, indices2):
    K = len(Q) + 2


    s_A0 = vmap(lambda a: jnp.sqrt(a))(A0)
    s_inv_A0 = vmap(lambda a: 1/jnp.sqrt(a))(A0)
    halfDx = 0.5*dx
    invDx = 1/dx
    gamma_ghost = jnp.zeros(K)
    gamma_ghost = gamma_ghost.at[1:-1].set(gamma)
    gamma_ghost = gamma_ghost.at[0].set(gamma[0])
    gamma_ghost = gamma_ghost.at[-1].set(gamma[-1])
    vA = jnp.empty(K)
    vQ = jnp.empty(K)
    vA = vA.at[0].set(A[0])
    vA = vA.at[-1].set(A[-1])

    vQ = vQ.at[0].set(Q[0])
    vQ = vQ.at[-1].set(Q[-1])
    vA = vA.at[1:-1].set(A)
    vQ = vQ.at[1:-1].set(Q)

    invDx_temp = jnp.concatenate((invDx, jnp.array([invDx[-1]])))
    limiterA = computeLimiter(vA, invDx_temp)
    limiterQ = computeLimiter(vQ, invDx_temp)
    halfDx_temp = jnp.concatenate((jnp.array([halfDx[0]]), halfDx, jnp.array([halfDx[-1]])))
    slopeA_halfDx = vmap(lambda a, b: a * b)(limiterA, halfDx_temp)
    slopeQ_halfDx = vmap(lambda a, b: a * b)(limiterQ, halfDx_temp)
    
    Al = vmap(lambda a, b: a+b)(vA, slopeA_halfDx)
    Ar = vmap(lambda a, b: a-b)(vA, slopeA_halfDx)
    Ql = vmap(lambda a, b: a+b)(vQ, slopeQ_halfDx)
    Qr = vmap(lambda a, b: a-b)(vQ, slopeQ_halfDx)

    Fl = jnp.array(vmap(computeFlux_par)(gamma_ghost, Al, Ql))
    Fr = jnp.array(vmap(computeFlux_par)(gamma_ghost, Ar, Qr))

    dxDt = dx / dt
    
    invDxDt = dt / dx

    flux = jnp.empty((2,K-1))
    dxDt_temp = jnp.empty(K-1)
    dxDt_temp = dxDt_temp.at[0:-1].set(dxDt)
    dxDt_temp = dxDt_temp.at[-1].set(dxDt[-1])
    flux = flux.at[0,:].set(vmap(lambda a, b, c, d, e: 0.5*(a+b - e*(c-d)))(Fr[0, 1:], Fl[0, 0:-1], Ar[1:], Al[0:-1], dxDt_temp))
    flux = flux.at[1,:].set(vmap(lambda a, b, c, d, e: 0.5*(a+b - e*(c-d)))(Fr[1, 1:], Fl[1, 0:-1], Qr[1:], Ql[0:-1], dxDt_temp))

    uStar = jnp.empty((2,K))
    uStar = uStar.at[0,1:-1].set(vmap(lambda a, b, c, d: a+d*(b-c))(vA[1:-1],
                                                             flux[0,0:-1],
                                                             flux[0,1:],
                                                             invDxDt))
    uStar = uStar.at[1,1:-1].set(vmap(lambda a, b, c, d: a+d*(b-c))(vQ[1:-1],
                                                             flux[1,0:-1],
                                                             flux[1,1:],
                                                             invDxDt))

    uStar1 = jnp.zeros((2, K+2))
    uStar1 = uStar1.at[:,0:-2].set(uStar)
    uStar2 = jnp.zeros((2, K+2))
    uStar2 = uStar1.at[:,1:-1].set(uStar)
    uStar3 = jnp.zeros((2, K+2))
    uStar3 = uStar1.at[:,2:].set(uStar)
    uStar2 = jnp.where(indices1, uStar1, uStar2) 
    uStar2 = jnp.where(indices2, uStar3, uStar2) 
    uStar = uStar2[:,1:-1]

    limiterA = computeLimiterIdx(uStar, 0, invDx_temp)
    limiterQ = computeLimiterIdx(uStar, 1, invDx_temp)
    slopesA = vmap(lambda a, b: a * b)(limiterA, halfDx_temp)
    slopesQ = vmap(lambda a, b: a * b)(limiterQ, halfDx_temp)

    Al = vmap(lambda a, b: a+b)(uStar[0,:], slopesA)
    Ar = vmap(lambda a, b: a-b)(uStar[0,:], slopesA)
    Ql = vmap(lambda a, b: a+b)(uStar[1,:], slopesQ)
    Qr = vmap(lambda a, b: a-b)(uStar[1,:], slopesQ)
    
    Fl = jnp.array(vmap(computeFlux_par)(gamma_ghost, Al, Ql))
    Fr = jnp.array(vmap(computeFlux_par)(gamma_ghost, Ar, Qr))

    flux = jnp.empty((2,K-1))
    flux = flux.at[0,:].set(vmap(lambda a, b, c, d, e: 0.5*(a+b - e*(c-d)))(Fr[0, 1:], Fl[0, 0:-1], Ar[1:], Al[0:-1], dxDt_temp))
    flux = flux.at[1,:].set(vmap(lambda a, b, c, d, e: 0.5*(a+b - e*(c-d)))(Fr[1, 1:], Fl[1, 0:-1], Qr[1:], Ql[0:-1], dxDt_temp))

    A = vmap(lambda a, b, c, d, e: 0.5*(a+b+e*(c-d)))(A[:],
                                                             uStar[0,1:-1],
                                                             flux[0,0:-1],
                                                             flux[0,1:],
                                                             invDxDt)
    Q = vmap(lambda a, b, c, d, e: 0.5*(a+b+e*(c-d)))(Q[:],
                                                             uStar[1,1:-1],
                                                             flux[1,0:-1],
                                                             flux[1,1:],
                                                             invDxDt)

    s_A = vmap(lambda a: jnp.sqrt(a))(A)
    Q = vmap(lambda a, b, c, d, e: a - dt*(viscT[0]*a/b + c*(d - e)*b))(Q, A, wallE, s_A, s_A0)
    P = vmap(lambda a, b, c, d: pressureSA(a*b, c, d))(s_A, s_inv_A0, beta, Pext)
    c = vmap(waveSpeedSA)(s_A, gamma)
    u = vmap(lambda a, b: a/b)(Q, A)

    return jnp.stack((u,Q,A,c,P))


def computeFlux(gamma_ghost, A, Q):
    return Q, Q * Q / A + gamma_ghost * A * jnp.sqrt(A)

def computeFlux_par(gamma_ghost, A, Q):
    return Q, Q * Q / A + gamma_ghost * A * jnp.sqrt(A)

def maxMod(a, b):
    return jnp.where(a > b, a, b)

def minMod(a, b):
    return jnp.where((a <= 0.0) | (b <= 0.0), 0.0, jnp.where(a < b, a, b))

def superBee(dU):
    return maxMod(minMod(dU[0, :], 2 * dU[1, :]), minMod(2 * dU[0, :], dU[1, :]))

def computeLimiter(U, invDx):
    dU = vmap(lambda a, b: a*b)(jnp.diff(U), invDx)
    return superBee(jnp.stack((jnp.concatenate((jnp.array([0.0]),dU)),
                               jnp.concatenate((dU,jnp.array([0.0]))))))
                                                                   

def computeLimiterIdx(U, idx, invDx):
    dU = vmap(lambda a, b: a*b)(jnp.diff(U[idx, :]), invDx)
    return superBee(jnp.stack((jnp.concatenate((jnp.array([0.0]),dU)),
                               jnp.concatenate((dU,jnp.array([0.0]))))))