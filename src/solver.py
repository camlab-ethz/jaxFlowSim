import jax.numpy as jnp
from functools import partial
from jax import lax, vmap, debug, jit
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

@partial(jit, static_argnums=(0,1))
def solveModel(N, B, starts, ends, indices1, indices2, t, dt, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, input_data, rho, junction_functions, mask, mask1):

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

    junction_functions_wrapper = vmap(lambda j, x: lax.switch(j,junction_functions,*x), in_axes=(0,None))

    args = (dt, sim_dat, sim_dat_aux)
    results = junction_functions_wrapper(jnp.arange(N), args)

    sim_dat_results = jnp.vstack((sim_dat[None], results[0]))
    sim_dat_aux_results = jnp.vstack((sim_dat_aux[None], results[1]))

    sim_dat = jnp.choose(mask, sim_dat_results, mode="clip")
    sim_dat_aux = jnp.choose(mask1, sim_dat_aux_results, mode="clip")

    return sim_dat, sim_dat_aux


def muscl(dt,
          Q, A,
          A0, beta,  gamma, wallE,
          dx, Pext, viscT,
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
