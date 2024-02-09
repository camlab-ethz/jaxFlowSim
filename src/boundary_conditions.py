import jax.numpy as jnp
from jax import lax, jit, debug
from functools import partial
from src.utils import pressure

def setInletBC(inlet, u0, u1, 
               A, c0, c1, 
               t, dt, input_data, 
               cardiac_T, invDx, A0, 
               beta, P_ext):
    Q0, P0 = lax.cond(inlet==1, lambda:(inputFromData(t, input_data.transpose(), cardiac_T),0.0), 
                                lambda:(0.0,inputFromData(t, input_data.transpose(), cardiac_T)))
    return inletCompatibility(inlet, u0, u1, 
                              Q0, A, c0, 
                              c1, P0, dt, 
                              invDx, A0, beta, 
                              P_ext)


def inputFromData(t, input_data, cardiac_T):
    idt = input_data[:, 0]
    idt1 = idt
    idt1 = idt1.at[:-1].set(idt1[1:])
    idq = input_data[:, 1]

    t_hat = t // cardiac_T
    t -= t_hat * cardiac_T

    idx = jnp.where((t >= idt) & (t <= idt1), jnp.arange(0,idt.size,1),jnp.zeros(idt.size)).sum().astype(int) #[0][0]

    qu = idq[idx] + (t - idt[idx]) * (idq[idx+1] - idq[idx]) / (idt[idx+1] - idt[idx])

    return qu

def inletCompatibility(inlet, u0, u1, Q0, A, c0, c1, P0, dt, invDx, A0, beta, Pext):
    W11, W21 = riemannInvariants(u0, c0)
    W12, _ = riemannInvariants(u1, c1)

    W11 += (W12 - W11) * (c0 - u0) * dt * invDx
    W21 = 2.0 * Q0 / A - W11

    u0, c0 = inverseRiemannInvariants(W11, W21)

    return lax.cond(inlet == 1, lambda: (Q0,  Q0/u0), lambda: (u0 * areaFromPressure(P0, A0, beta, Pext), areaFromPressure(P0, A0, beta, Pext)))

def riemannInvariants(u, c):
    W1 = u - 4.0 * c
    W2 = u + 4.0 * c

    return W1, W2


def inverseRiemannInvariants(W1, W2):
    u = 0.5 * (W1 + W2)
    c = (W2 - W1) * 0.125

    return u, c

def areaFromPressure(P, A0, beta, P_ext):
    return A0 * ((P - P_ext) / beta + 1.0) * ((P - P_ext) / beta + 1.0)

def setReflectionOutletBCWrapper(dt, sim_dat, sim_dat_aux, 
                       sim_dat_const, end, i):

    index1 = end
    index2 = end-1
    index3 = end-2
    u1 = sim_dat[0,index1]
    u2 = sim_dat[0,index2]
    A1 = sim_dat[2,index1]
    c1 = sim_dat[3,index1]
    c2 = sim_dat[3,index2]
    P1 = sim_dat[4,index1]
    P2 = sim_dat[4,index2]
    P3 = sim_dat[4,index3]
    Pc = sim_dat_aux[i,2]
    W1M0 = sim_dat_aux[i,0]
    W2M0 = sim_dat_aux[i,1]
    P1 = 2.0 * P2 - P3
    u, Q, c = outletCompatibility(u1, u2, A1, 
                                    c1, c2, W1M0, 
                                    W2M0, dt, 
                                    *sim_dat_const)
    temp = jnp.array((u,Q,A1,c,P1))
    sim_dat = lax.dynamic_update_slice(
        sim_dat, 
        temp[:,jnp.newaxis]*jnp.ones(3)[jnp.newaxis,:],
        (0,index1))
    sim_dat_aux = sim_dat_aux.at[i,2].set(Pc)
    return sim_dat, sim_dat_aux

def setWindkesselOutletBCWrapper(dt, sim_dat, sim_dat_aux, 
                       sim_dat_const, end, i):

    index1 = end
    u1 = sim_dat[0,index1]
    Q1 = sim_dat[1,index1]
    A1 = sim_dat[2,index1]
    c1 = sim_dat[3,index1]
    P1 = sim_dat[4,index1]
    Pc = sim_dat_aux[i,2]
    u, A1, Pc = threeElementWindkessel(dt, u1, A1, Pc, 
                                       *sim_dat_const)
    temp = jnp.array((u,Q1,A1,c1,P1))
    sim_dat = lax.dynamic_update_slice(
        sim_dat, 
        temp[:,jnp.newaxis]*jnp.ones(3)[jnp.newaxis,:],
        (0,index1))
    sim_dat_aux = sim_dat_aux.at[i,2].set(Pc)
    return sim_dat, sim_dat_aux

def outletCompatibility(u1, u2, A1, 
                        c1, c2, W1M0, 
                        W2M0, dt, dx, 
                        Rt):
    _, W2M1 = riemannInvariants(u2, c2)
    W1M, W2M = riemannInvariants(u1, c1)

    W2M += (W2M1 - W2M) * (u1 + c1) * dt / dx
    W1M = W1M0 - Rt * (W2M - W2M0)

    u1, c1 = inverseRiemannInvariants(W1M, W2M)
    Q1 = A1 * u1

    return u1, Q1, c1

def threeElementWindkessel(dt, u1, A1, 
                           Pc, Cc, R1, 
                           R2, beta, gamma, 
                           A0, Pext):
    #debug.print("{x}", x=1111111)
    Pout = 0.0

    Al = A1
    ul = u1
    Pc += dt / Cc * (Al * ul - (Pc - Pout) / R2)

    As = Al
    ssAl = jnp.sqrt(jnp.sqrt(Al))
    sgamma = 2 * jnp.sqrt(6 * gamma)
    sA0 = jnp.sqrt(A0)
    bA0 = beta / sA0

    def fun(As):
        return As * R1 * (ul + sgamma * (ssAl - jnp.sqrt(jnp.sqrt(As)))) - (Pext + bA0 * (jnp.sqrt(As) - sA0)) + Pc

    def dfun(As):
        return R1 * (ul + sgamma * (ssAl - 1.25 * jnp.sqrt(jnp.sqrt(As)))) - bA0 * 0.5 / jnp.sqrt(As)
    def newtonSolver(x0):
        xn = x0 - fun(x0) / dfun(x0)

        return xn

    As = newtonSolver(As)

    us = (pressure(As, A0, beta, Pext) - Pout) / (As * R1)

    A1 = As
    u1 = us
    return u1, A1, Pc
