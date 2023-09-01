import jax
import jax.numpy as jnp
from jax import lax
from src.utils import pressure
from functools import partial

@jax.jit
def setInletBC(t, dt, v):
    h = v.heart

    if h.inlet_type == "Q":
        v.Q = v.Q.at[0].set(inputFromData(t, h))
    else:
        v.P = v.P.at[0].set(inputFromData(t, h))

    return inletCompatibility(dt, v, h)

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

@jax.jit
def inletCompatibility(dt, v, h):
    W11, W21 = riemannInvariants(0, v)
    W12, W22 = riemannInvariants(1, v)

    W11 += (W12 - W11) * (v.c[0] - v.u[0]) * dt * v.invDx
    W21 = 2.0 * v.Q[0] / v.A[0] - W11

    v.u = v.u.at[0].set(inverseRiemannInvariants(W11, W21)[0])
    v.c = v.c.at[0].set(inverseRiemannInvariants(W11, W21)[1])

    if h.inlet_type == "Q":
        v.A = v.A.at[0].set(v.Q[0] / v.u[0])
        v.P = v.P.at[0].set(pressure(v.A[0], v.A0[0], v.beta[0], v.Pext))
    else:
        v.A = v.A.at[0].set(areaFromPressure(v.P[0], v.A0[0], v.beta[0], v.Pext))
        v.Q = v.Q.at[0].set(v.u[0] * v.A[0])

    return v

@jax.jit
def riemannInvariants(i, v):
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

@jax.jit
def setOutletBC(dt, v):
    if v.outlet == "reflection":
        v.P = v.P.at[-1].set(2.0 * v.P[-2] - v.P[-3])
        v = outletCompatibility(dt, v)
    elif v.outlet == "wk3":
        v = threeElementWindkessel(dt, v)

    return v

@jax.jit
def outletCompatibility(dt, v):
    W1M1, W2M1 = riemannInvariants(v.M - 2, v)
    W1M, W2M = riemannInvariants(v.M-1, v)

    W2M += (W2M1 - W2M) * (v.u[-1] + v.c[-1]) * dt / v.dx
    W1M = v.W1M0 - v.Rt * (W2M - v.W2M0)

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
