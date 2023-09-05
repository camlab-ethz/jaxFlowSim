import jax.numpy as jnp
from jax import jit #, grad, vmap
from src.newton import newtonRaphson
from src.utils import pressure, waveSpeed
import src.initialise as ini
from functools import partial

@partial(jit, static_argnums=(0,1,2,))
def solveBifurcation(l, m, n, u1, u2, u3, A1, A2, A3):
    U0 = jnp.array([u1,
                   u2,
                   u3,
                   jnp.sqrt(jnp.sqrt(A1)),
                   jnp.sqrt(jnp.sqrt(A2)),
                   jnp.sqrt(jnp.sqrt(A3))])

    k1 = ini.VCS[l].s_15_gamma[-1]
    k2 = ini.VCS[m].s_15_gamma[0]
    k3 = ini.VCS[n].s_15_gamma[0]
    k = jnp.array([k1, k2, k3])

    J = calculateJacobianBifurcation((l,m,n), U0, k)
    U = newtonRaphson((l,m,n), calculateWstarBifurcation, calculateFBifurcation, J, U0, k)[0]

    return updateBifurcation(l, m, n, U)


@partial(jit, static_argnums=(0,))
def calculateJacobianBifurcation(indices, U, k):
    l, m, n = indices
    U43 = U[3]**3
    U53 = U[4]**3
    U63 = U[5]**3

    J14 = 4.0 * k[0]
    J25 = -4.0 * k[1]
    J36 = -4.0 * k[2]

    J41 = U[3] * U43
    J42 = -U[4] * U53
    J43 = -U[5] * U63
    J44 = 4.0 * U[0] * U43
    J45 = -4.0 * U[1] * U53
    J46 = -4.0 * U[2] * U63

    J54 = 2.0 * ini.VCS[l].beta[-1] * U[3] * ini.VCS[l].s_inv_A0[-1]
    J55 = -2.0 * ini.VCS[m].beta[0] * U[4] * ini.VCS[m].s_inv_A0[0]

    J64 = 2.0 * ini.VCS[l].beta[-1] * U[3] * ini.VCS[l].s_inv_A0[-1]
    J66 = -2.0 * ini.VCS[n].beta[0] * U[5] * ini.VCS[n].s_inv_A0[0]

    return jnp.array([[1.0, 0.0, 0.0, J14, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, J25, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, J36],
                      [J41, J42, J43, J44, J45, J46],
                      [0.0, 0.0, 0.0, J54, J55, 0.0],
                      [0.0, 0.0, 0.0, J64, 0.0, J66]], dtype=jnp.float64)


@jit
def calculateWstarBifurcation(U, k):
    W1 = U[0] + 4.0 * k[0] * U[3]
    W2 = U[1] - 4.0 * k[1] * U[4]
    W3 = U[2] - 4.0 * k[2] * U[5]

    return jnp.array([W1, W2, W3], dtype=jnp.float64)


@partial(jit, static_argnums=(0,))
def calculateFBifurcation(indices, U, k, W):
    l, m ,n = indices

    U42 = U[3]**2
    U52 = U[4]**2
    U62 = U[5]**2

    f1 = U[0] + 4.0 * k[0] * U[3] - W[0]
    f2 = U[1] - 4.0 * k[1] * U[4] - W[1]
    f3 = U[2] - 4.0 * k[2] * U[5] - W[2]
    f4 = U[0] * (U42**2) - U[1] * (U52**2) - U[2] * (U62**2)

    f5 = ini.VCS[l].beta[-1] * (U42 * ini.VCS[l].s_inv_A0[-1] - 1.0) - (ini.VCS[m].beta[0] * (U52 * ini.VCS[m].s_inv_A0[0] - 1.0))
    f6 = ini.VCS[l].beta[-1] * (U42 * ini.VCS[l].s_inv_A0[-1] - 1.0) - (ini.VCS[n].beta[0] * (U62 * ini.VCS[n].s_inv_A0[0] - 1.0))

    return jnp.array([f1, f2, f3, f4, f5, f6], dtype=jnp.float64)


@partial(jit, static_argnums=(0,1,2,))
def updateBifurcation(l, m, n, U):
    u1= U[0]
    u2= U[1]
    u3= U[2]

    A1 = U[3]**4
    A2 = U[4]**4
    A3 = U[5]**4

    Q1 = u1 * A1
    Q2 = u2 * A2
    Q3 = u3 * A3

    P1 = pressure(A1, ini.VCS[l].A0[-1], ini.VCS[l].beta[-1], ini.VCS[l].Pext)
    P2 = pressure(A2, ini.VCS[m].A0[0], ini.VCS[m].beta[0], ini.VCS[m].Pext)
    P3 = pressure(A3, ini.VCS[n].A0[0], ini.VCS[n].beta[0], ini.VCS[m].Pext)

    c1 = waveSpeed(A1, ini.VCS[l].gamma[-1])
    c2 = waveSpeed(A2, ini.VCS[m].gamma[0])
    c3 = waveSpeed(A3, ini.VCS[n].gamma[0])

    return u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3
