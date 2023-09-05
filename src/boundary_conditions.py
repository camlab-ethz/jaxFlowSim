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
@partial(jax.jit, static_argnums=0)
def setOutletBC(i, u1, u2, Q1, A1, c1, c2, P1, P2, P3, Pc, W1M0, W2M0, dt):
    if ini.VCS[i].outlet == "reflection":
        P1 = 2.0 * P2 - P3
        u1, Q1, c1 = outletCompatibility(i, u1, u2, A1, c1, c2, W1M0, W2M0, dt)
        return u1, Q1, A1, c1, P1, Pc
    elif ini.VCS[i].outlet == "wk3":
        u1, A1, Pc = threeElementWindkessel(i, dt, u1, A1, Pc)
        return u1, Q1, A1, c1, P1, Pc

@partial(jax.jit, static_argnums=0)
def outletCompatibility(i, u1, u2, A1, c1, c2, W1M0, W2M0, dt):
    W1M1, W2M1 = riemannInvariants(u2, c2)
    W1M, W2M = riemannInvariants(u1, c1)

    W2M += (W2M1 - W2M) * (u1 + c1) * dt / ini.VCS[i].dx
    W1M = W1M0 - ini.VCS[i].Rt * (W2M - W2M0)

    u1, c1 = inverseRiemannInvariants(W1M, W2M)
    Q1 = A1 * u1
    #jax.debug.breakpoint()

    return u1, Q1, c1

@partial(jax.jit, static_argnums=0)
def threeElementWindkessel(i, dt, u1, A1, Pc):
    Pout = 0.0

    Al = A1
    ul = u1

    Pc += dt / ini.VCS[i].Cc * (Al * ul - (Pc - Pout) / ini.VCS[i].R2)

    As = Al
    ssAl = jnp.sqrt(jnp.sqrt(Al))
    sgamma = 2 * jnp.sqrt(6 * ini.VCS[i].gamma[-1])
    sA0 = jnp.sqrt(ini.VCS[i].A0[-1])
    bA0 = ini.VCS[i].beta[-1] / sA0

    @jax.jit
    def fun(As):
        return As * ini.VCS[i].R1 * (ul + sgamma * (ssAl - jnp.sqrt(jnp.sqrt(As)))) - (ini.VCS[i].Pext + bA0 * (jnp.sqrt(As) - sA0)) + Pc

    @jax.jit
    def dfun(As):
        return ini.VCS[i].R1 * (ul + sgamma * (ssAl - 1.25 * jnp.sqrt(jnp.sqrt(As)))) - bA0 * 0.5 / jnp.sqrt(As)

    try:
        As = newtonSolver(fun, dfun, As)
    except Exception as e:
        vlab = ini.VCS[i].label
        print(f"\nNewton solver doesn't converge at {vlab} outlet!")
        raise e

    us = (pressure(As, ini.VCS[i].A0[-1], ini.VCS[i].beta[-1], ini.VCS[i].Pext) - Pout) / (As * ini.VCS[i].R1)

    A1 = As
    u1 = us

    return u1, A1, Pc

@partial(jax.jit, static_argnums=(0,1,))
def newtonSolver(f, df, x0):
    xn = x0 - f(x0) / df(x0)
    
    def cond_fun(val):
        ret = jax.lax.cond( jnp.abs(val[0]-val[1]) < 1e-5, lambda: False, lambda: True)
        return ret

    def body_fun(val):
        return jnp.array((val[1],val[1] - f(val[1]) / df(val[1]))) 
    temp = lax.while_loop(cond_fun, body_fun, jnp.array((x0,xn)))
    return temp[1]

@jax.jit
def updateGhostCell(Q0, Q1, QM1, QM2, A0, A1, AM1, AM2):
    U00Q = Q0
    U00A = A0
    U01Q = Q1
    U01A = A1
    UM1Q = QM1
    UM1A = AM1
    UM2Q = QM2
    UM2A = AM2

    return U00Q, U00A, U01Q, U01A, UM1Q, UM1A, UM2Q, UM2A

@jax.jit
def updateGhostCells(sim_dat):
    sim_dat_aux_temp = jnp.zeros((8,ini.NUM_VESSELS), dtype=jnp.float64)
    for i in range(ini.NUM_VESSELS):
        start = ini.MESH_SIZES[i]
        end = ini.MESH_SIZES[i+1]
        Q0 = sim_dat[1,start]
        Q1 = sim_dat[1,start+1]
        QM1 = sim_dat[1,end-1]
        QM2 = sim_dat[1,end-2]
        A0 = sim_dat[2,start]
        A1 = sim_dat[2,start+1]
        AM1 = sim_dat[2,end-1]
        AM2 = sim_dat[2,end-2]

        U00Q, U00A, U01Q, U01A, UM1Q, UM1A, UM2Q, UM2A = updateGhostCell(Q0, Q1, QM1, QM2, A0, A1, AM1, AM2)
        sim_dat_aux_temp = sim_dat_aux_temp.at[0,i].set(U00Q)
        sim_dat_aux_temp = sim_dat_aux_temp.at[1,i].set(U00A)
        sim_dat_aux_temp = sim_dat_aux_temp.at[2,i].set(U01Q)
        sim_dat_aux_temp = sim_dat_aux_temp.at[3,i].set(U01A)
        sim_dat_aux_temp = sim_dat_aux_temp.at[4,i].set(UM1Q)
        sim_dat_aux_temp = sim_dat_aux_temp.at[5,i].set(UM1A)
        sim_dat_aux_temp = sim_dat_aux_temp.at[6,i].set(UM2Q)
        sim_dat_aux_temp = sim_dat_aux_temp.at[7,i].set(UM2A)
        
    return sim_dat_aux_temp
    #return [updateGhostCell(vessel) for vessel in vessels]
