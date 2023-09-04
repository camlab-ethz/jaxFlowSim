import jax
import jax.numpy as jnp
from jax import lax
from src.utils import pressure
from functools import partial
import src.initialise as ini

#@jax.jit
@partial(jax.jit, static_argnums=0)
def setInletBC(i, u0, u1, A0, c0, c1, t, dt):
    h = ini.VCS[i].heart
    Q0 = 0
    P0 = 0
    if h.inlet_type == "Q":
        Q0 = inputFromData(t, h)
    else:
        P0 = inputFromData(t, h)

    return inletCompatibility(i, h, u0, u1, Q0, A0, c0, c1, P0, dt)

@partial(jax.jit, static_argnums=1)
def inputFromData(t, h):
    idt = h.input_data[:, 0]
    idt1 = idt
    idt1 = idt1.at[:-1].set(idt1[1:])
    idq = h.input_data[:, 1]

    t_hat = t // h.cardiac_T
    t -= t_hat * h.cardiac_T

    idx = jnp.where((t >= idt) & (t <= idt1), jnp.arange(0,idt.size,1),jnp.zeros(idt.size)).sum().astype(int) #[0][0]


    qu = idq[idx] + (t - idt[idx]) * (idq[idx+1] - idq[idx]) / (idt[idx+1] - idt[idx])

    return qu

#@jax.jit
@partial(jax.jit, static_argnums=(0,1,))
def inletCompatibility(i, h, u0, u1, Q0, A0, c0, c1, P0, dt):
    W11, W21 = riemannInvariants(u0, c0)
    W12, W22 = riemannInvariants(u1, c1)

    W11 += (W12 - W11) * (c0 - u0) * dt * ini.VCS[i].invDx
    W21 = 2.0 * Q0 / A0 - W11

    u0, c0 = inverseRiemannInvariants(W11, W21)

    if h.inlet_type == "Q":
        A0 = Q0 / u0
        P0 = pressure(A0, ini.VCS[i].A0[0], ini.VCS[i].beta[0], ini.VCS[i].Pext)
    else:
        A0 = areaFromPressure(P0, ini.VCS[i].A0[0], ini.VCS[i].beta[0], ini.VCS[i].Pext)
        Q0 = u0 * A0

    return Q0, A0

@jax.jit
def riemannInvariants(u, c):
    W1 = u - 4.0 * c
    W2 = u + 4.0 * c

    return W1, W2

@jax.jit
def riemannInvariants_old(i, v):
    W1 = v.u[i] - 4.0 * v.c[i]
    W2 = v.u[i] + 4.0 * v.c[i]

    return W1, W2

@jax.jit
def inverseRiemannInvariants(W1, W2):
    u = 0.5 * (W1 + W2)
    c = (W2 - W1) * 0.125

    return u, c

@jax.jit
def areaFromPressure(P, A0, beta, Pext):
    return A0 * ((P - Pext) / beta + 1.0) * ((P - Pext) / beta + 1.0)

#@jax.jit
@partial(jax.jit, static_argnums=2)
def setOutletBC(dt, v, i):
    if ini.VCS[i].outlet == "reflection":
        v.P = v.P.at[-1].set(2.0 * v.P[-2] - v.P[-3])
        v = outletCompatibility(dt, v, i)
    elif ini.VCS[i].outlet == "wk3":
        v = threeElementWindkessel(dt, v)

    return v

#@jax.jit
@partial(jax.jit, static_argnums=2)
def outletCompatibility(dt, v, i):
    W1M1, W2M1 = riemannInvariants_old(ini.VCS[i].M - 2, v)
    W1M, W2M = riemannInvariants_old(ini.VCS[i].M-1, v)

    W2M += (W2M1 - W2M) * (v.u[-1] + v.c[-1]) * dt / ini.VCS[i].dx
    W1M = v.W1M0 - ini.VCS[i].Rt * (W2M - v.W2M0)

    v.u = v.u.at[-1].set(inverseRiemannInvariants(W1M, W2M)[0])
    v.c = v.c.at[-1].set(inverseRiemannInvariants(W1M, W2M)[1])
    v.Q = v.Q.at[-1].set(v.A[-1] * v.u[-1])

    return v

@jax.jit
def threeElementWindkessel(dt, v):
    Pout = 0.0

    Al = v.A[-1]
    ul = v.u[-1]

    v.Pc += dt / v.Cc * (Al * ul - (v.Pc - Pout) / v.R2)

    As = Al
    ssAl = jnp.sqrt(jnp.sqrt(Al))
    sgamma = 2 * jnp.sqrt(6 * v.gamma[-1])
    sA0 = jnp.sqrt(v.A0[-1])
    bA0 = v.beta[-1] / sA0

    @jax.jit
    def fun(As):
        return As * v.R1 * (ul + sgamma * (ssAl - jnp.sqrt(jnp.sqrt(As)))) - (v.Pext + bA0 * (jnp.sqrt(As) - sA0)) + v.Pc

    @jax.jit
    def dfun(As):
        return v.R1 * (ul + sgamma * (ssAl - 1.25 * jnp.sqrt(jnp.sqrt(As)))) - bA0 * 0.5 / jnp.sqrt(As)

    try:
        As = newtonSolver(fun, dfun, As)
    except Exception as e:
        vlab = v.label
        print(f"\nNewton solver doesn't converge at {vlab} outlet!")
        raise e

    us = (pressure(As, v.A0[-1], v.beta[-1], v.Pext) - Pout) / (As * v.R1)

    v.A = v.A.at[-1].set(As)
    v.u = v.u.at[-1].set(us)

    return v

@partial(jax.jit, static_argnums=(0,1,))
def newtonSolver(f, df, x0):
    xn = x0 - f(x0) / df(x0)
    
    @jax.jit
    def cond_fun(val):
        ret = jax.lax.cond( jnp.abs(val[0]-val[1]) < 1e-5, lambda: False, lambda: True)
        return ret
    @jax.jit
    def body_fun(val):
        return jnp.array((val[1],val[1] - f(val[1]) / df(val[1]))) 
    temp = lax.while_loop(cond_fun, body_fun, jnp.array((x0,xn)))
    return temp[1]

@jax.jit
def updateGhostCell(v):
    v.U00A = v.A[0]
    v.U00Q = v.Q[0]
    v.U01A = v.A[1]
    v.U01Q = v.Q[1]

    v.UM1A = v.A[-1]
    v.UM1Q = v.Q[-1]
    v.UM2A = v.A[-2]
    v.UM2Q = v.Q[-2]

    return v

@jax.jit
def updateGhostCells(vessels):
    return [updateGhostCell(vessel) for vessel in vessels]
