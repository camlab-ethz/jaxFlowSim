import jax
import jax.numpy as jnp
from jax import lax
from src.utils import pressure
from functools import partial
import src.initialise as ini

@jax.jit
def setInletBC(inlet, u0, u1, A, c0, c1, t, dt, input_data, cardiac_T, invDx, A0, beta, Pext):
    #if inlet == 1: #"Q":
    Q0, P0 = jax.lax.cond(inlet==1, lambda:(inputFromData(t, input_data.transpose(), cardiac_T),0.0), lambda:(0.0,inputFromData(t, input_data.transpose(), cardiac_T)))
    #else:
    #    P0 = inputFromData(t, input_data, cardiac_T)
    return inletCompatibility(inlet, u0, u1, Q0, A, c0, c1, P0, dt, invDx, A0, beta, Pext)


@jax.jit
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

@jax.jit
def inletCompatibility(inlet, u0, u1, Q0, A, c0, c1, P0, dt, invDx, A0, beta, Pext):
    W11, W21 = riemannInvariants(u0, c0)
    W12, _ = riemannInvariants(u1, c1)

    W11 += (W12 - W11) * (c0 - u0) * dt * invDx
    W21 = 2.0 * Q0 / A - W11

    u0, c0 = inverseRiemannInvariants(W11, W21)

    #if inlet == 1:
    #    A = Q0 / u0
    #    P0 = pressure(A, A0, beta, Pext)
    #else:
    #    A = areaFromPressure(P0, A0, beta, Pext)
    #    Q0 = u0 * A
    
    return jax.lax.cond(inlet == 1, lambda: (Q0,  Q0/u0), lambda: (u0 * A, areaFromPressure(P0, A0, beta, Pext)))

    #return Q0, A

@jax.jit
def riemannInvariants(u, c):
    W1 = u - 4.0 * c
    W2 = u + 4.0 * c

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
def setOutletBC(dt, u1, u2, Q1, A1, c1, c2, P1, P2, P3, Pc, W1M0, W2M0, A0, beta, gamma, dx, Pext, outlet, Rt, R1, R2, Cc):
    def outletCompatibility_wrapper():
        P1_out = 2.0 * P2 - P3
        u1_out, Q1_out, c1_out = outletCompatibility(u1, u2, A1, c1, c2, W1M0, W2M0, dt, dx, Rt)
        return u1_out, Q1_out, A1, c1_out, P1_out, Pc
    def threeElementWindkessel_wrapper():
        u1_out, A1_out, Pc_out = threeElementWindkessel(dt, u1, A1, Pc, Cc, R1, R2, beta, gamma, A0, Pext)
        return u1_out, Q1, A1_out, c1, P1, Pc_out
    return jax.lax.cond(outlet == 1,
                  lambda: outletCompatibility_wrapper(),
                  lambda: threeElementWindkessel_wrapper())

@jax.jit
def outletCompatibility(u1, u2, A1, c1, c2, W1M0, W2M0, dt, dx, Rt):
    _, W2M1 = riemannInvariants(u2, c2)
    W1M, W2M = riemannInvariants(u1, c1)

    W2M += (W2M1 - W2M) * (u1 + c1) * dt / dx
    W1M = W1M0 - Rt * (W2M - W2M0)

    u1, c1 = inverseRiemannInvariants(W1M, W2M)
    Q1 = A1 * u1
    #jax.debug.breakpoint()

    return u1, Q1, c1

@jax.jit
def threeElementWindkessel(dt, u1, A1, Pc, Cc, R1, R2, beta, gamma, A0, Pext):
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

        def cond_fun(val):
            ret = jax.lax.cond( jnp.abs(val[0]-val[1]) < 1e-5, lambda: False, lambda: True)
            return ret

        def body_fun(val):
            return jnp.array((val[1],val[1] - fun(val[1]) / dfun(val[1]))) 
        temp = lax.while_loop(cond_fun, body_fun, jnp.array((x0,xn)))
        return temp[1]

    try:
        As = newtonSolver(As)
    except Exception as e:
        #vlab = ini.VCS[i].label
        #print(f"\nNewton solver doesn't converge at {vlab} outlet!")
        raise e

    us = (pressure(As, A0, beta, Pext) - Pout) / (As * R1)

    A1 = As
    u1 = us

    return u1, A1, Pc

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
    sim_dat_aux = jnp.zeros((8,ini.NUM_VESSELS), dtype=jnp.float64)
    def body_fun(i,sim_dat_aux):
        start = i*ini.MESH_SIZE
        end = (i+1)*ini.MESH_SIZE
        Q0 = sim_dat[1,start]
        Q1 = sim_dat[1,start+1]
        QM1 = sim_dat[1,end-1]
        QM2 = sim_dat[1,end-2]
        A0 = sim_dat[2,start]
        A1 = sim_dat[2,start+1]
        AM1 = sim_dat[2,end-1]
        AM2 = sim_dat[2,end-2]

        U00Q, U00A, U01Q, U01A, UM1Q, UM1A, UM2Q, UM2A = updateGhostCell(Q0, Q1, QM1, QM2, A0, A1, AM1, AM2)
        sim_dat_aux = sim_dat_aux.at[0,i].set(U00Q)
        sim_dat_aux = sim_dat_aux.at[1,i].set(U00A)
        sim_dat_aux = sim_dat_aux.at[2,i].set(U01Q)
        sim_dat_aux = sim_dat_aux.at[3,i].set(U01A)
        sim_dat_aux = sim_dat_aux.at[4,i].set(UM1Q)
        sim_dat_aux = sim_dat_aux.at[5,i].set(UM1A)
        sim_dat_aux = sim_dat_aux.at[6,i].set(UM2Q)
        sim_dat_aux = sim_dat_aux.at[7,i].set(UM2A)

        return sim_dat_aux

    sim_dat_aux = jax.lax.fori_loop(0,ini.NUM_VESSELS, body_fun, sim_dat_aux)

    return sim_dat_aux
    #return [updateGhostCell(vessel) for vessel in vessels]
