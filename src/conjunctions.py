import jax
from jax import jit
import jax.numpy as jnp
from src.newton import newtonRaphson
from src.utils import pressure, waveSpeed
import src.initialise as ini
from functools import partial

@partial(jit, static_argnums=(0,1,))
def solveConjunction(l, m, u1, u2, A1, A2):
    U0 = jnp.array((u1, u2, jnp.sqrt(jnp.sqrt(A1)), jnp.sqrt(jnp.sqrt(A2))), dtype=jnp.float64)

    k1 = ini.VCS[l].s_15_gamma[-1]
    k2 = ini.VCS[m].s_15_gamma[0]
    k3 = ini.BLOOD.rho
    k = jnp.array([k1, k2, k3])

    J = calculateJacobianConjunction((l,m), U0, k)
    U = newtonRaphson((l,m), calculateWStarConjunction, calculateFConjunction, J, U0, k)[0]

    return updateConjunction(l, m, U)


@partial(jit, static_argnums=(0,))
def calculateJacobianConjunction(indices, U, k):
    l, m = indices
    U33 = U[2]**3
    U43 = U[3]**3

    J13 =  4.0 * k[0]
    J24 = -4.0 * k[1]

    J31 =  U33 * U[2]
    J32 = -U43 * U[3]
    J33 =  4.0 * U[0] * U33
    J34 = -4.0 * U[1] * U43

    J41 =  k[2] * U[0]
    J42 = -k[2] * U[1]
    J43 =  2.0 * ini.VCS[l].beta[-1] * U[2] * ini.VCS[l].s_inv_A0[-1]
    J44 = -2.0 * ini.VCS[m].beta[0] * U[3] * ini.VCS[m].s_inv_A0[0]

    return jnp.array([[1.0, 0.0, J13, 0.0],
                      [0.0, 1.0, 0.0, J24],
                      [J31, J32, J33, J34],
                      [J41, J42, J43, J44]], dtype=jnp.float64)


@jax.jit
def calculateWStarConjunction(U, k):
    W1 = U[0] + 4.0 * k[0] * U[2]
    W2 = U[1] - 4.0 * k[1] * U[3]

    return jnp.array([W1, W2], dtype=jnp.float64)


@partial(jit, static_argnums=(0,))
def calculateFConjunction(indices, U, k, W):
    (l,m) = indices

    U32 = U[2]**2
    U42 = U[3]**2

    f1 = U[0] + 4.0 * k[0] * U[2] - W[0]
    f2 = U[1] - 4.0 * k[1] * U[3] - W[1]
    f3 = U[0] * U32**2 - U[1] * U42**2

    f4 = 0.5 * k[2] * U[0]**2 + ini.VCS[l].beta[-1] * (U32 * ini.VCS[l].s_inv_A0[-1] - 1.0) - (0.5 * k[2] * U[1]**2 + ini.VCS[m].beta[0] * (U42 * ini.VCS[m].s_inv_A0[0] - 1.0))

    return jnp.array([f1, f2, f3, f4], dtype=jnp.float64)


@partial(jit, static_argnums=(0,1))
def updateConjunction(l, m , U):
    u1 = U[0]
    u2 = U[1]

    A1 = U[2]**4
    Q1 = u1 * A1

    A2  = U[3]**4
    Q2  = u2 * A2

    P1 = pressure(A1, ini.VCS[l].A0[-1], ini.VCS[l].beta[-1], ini.VCS[l].Pext)
    P2 = pressure(A2, ini.VCS[m].A0[0], ini.VCS[m].beta[0], ini.VCS[m].Pext)

    c1 = waveSpeed(A1, ini.VCS[l].gamma[-1])
    c2 = waveSpeed(A2, ini.VCS[m].gamma[0])

    return u1, u2, Q1, Q2, A1, A2, c1, c2, P1, P2
