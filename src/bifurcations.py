import jax.numpy as jnp
from jax import jit #, grad, vmap
from jax.scipy.linalg import solve
from src.newton import newtonRaphson
from src.utils import pressure, waveSpeed

@jit
def solveBifurcation(v1, v2, v3):
    U0 = jnp.array([v1.u[-1],
                   v2.u[0],
                   v3.u[0],
                   jnp.sqrt(jnp.sqrt(v1.A[-1])),
                   jnp.sqrt(jnp.sqrt(v2.A[0])),
                   jnp.sqrt(jnp.sqrt(v3.A[0]))])

    k1 = v1.s_15_gamma[-1]
    k2 = v2.s_15_gamma[0]
    k3 = v3.s_15_gamma[0]
    k = jnp.array([k1, k2, k3])

    J = calculateJacobianBifurcation(v1, v2, v3, U0, k)
    U = newtonRaphson([v1, v2, v3], J, U0, k, calculateWstarBifurcation, calculateFBifurcation)[0]

    return updateBifurcation(U, v1, v2, v3)


@jit
def calculateJacobianBifurcation(v1, v2, v3, U, k):
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

    J54 = 2.0 * v1.beta[-1] * U[3] * v1.s_inv_A0[-1]
    J55 = -2.0 * v2.beta[0] * U[4] * v2.s_inv_A0[0]

    J64 = 2.0 * v1.beta[-1] * U[3] * v1.s_inv_A0[-1]
    J66 = -2.0 * v3.beta[0] * U[5] * v3.s_inv_A0[0]

    return jnp.array([[1.0, 0.0, 0.0, J14, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, J25, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, J36],
                      [J41, J42, J43, J44, J45, J46],
                      [0.0, 0.0, 0.0, J54, J55, 0.0],
                      [0.0, 0.0, 0.0, J64, 0.0, J66]])


@jit
def calculateWstarBifurcation(U, k):
    W1 = U[0] + 4.0 * k[0] * U[3]
    W2 = U[1] - 4.0 * k[1] * U[4]
    W3 = U[2] - 4.0 * k[2] * U[5]

    return jnp.array([W1, W2, W3])


@jit
def calculateFBifurcation(vessels, U, k, W):
    v1 = vessels[0]
    v2 = vessels[1]
    v3 = vessels[2]

    U42 = U[3]**2
    U52 = U[4]**2
    U62 = U[5]**2

    f1 = U[0] + 4.0 * k[0] * U[3] - W[0]
    f2 = U[1] - 4.0 * k[1] * U[4] - W[1]
    f3 = U[2] - 4.0 * k[2] * U[5] - W[2]
    f4 = U[0] * (U42**2) - U[1] * (U52**2) - U[2] * (U62**2)

    f5 = v1.beta[-1] * (U42 * v1.s_inv_A0[-1] - 1.0) - (v2.beta[0] * (U52 * v2.s_inv_A0[0] - 1.0))
    f6 = v1.beta[-1] * (U42 * v1.s_inv_A0[-1] - 1.0) - (v3.beta[0] * (U62 * v3.s_inv_A0[0] - 1.0))

    return jnp.array([f1, f2, f3, f4, f5, f6])


@jit
def updateBifurcation(U, v1, v2, v3):
    v1.u = v1.u.at[-1].set(U[0])
    v2.u = v2.u.at[0].set(U[1])
    v3.u = v3.u.at[0].set(U[2])

    v1.A = v1.A.at[-1].set(U[3]**4)
    v2.A = v2.A.at[0].set(U[4]**4)
    v3.A = v3.A.at[0].set(U[5]**4)

    v1.Q = v1.Q.at[-1].set(v1.u[-1] * v1.A[-1])
    v2.Q = v2.Q.at[0].set(v2.u[0] * v2.A[0])
    v3.Q = v3.Q.at[0].set(v3.u[0] * v3.A[0])

    v1.P = v1.P.at[-1].set(pressure(v1.A[-1], v1.A0[-1], v1.beta[-1], v1.Pext))
    v2.P = v2.P.at[0].set(pressure(v2.A[0], v2.A0[0], v2.beta[0], v2.Pext))
    v3.P = v3.P.at[0].set(pressure(v3.A[0], v3.A0[0], v3.beta[0], v3.Pext))

    v1.c = v1.c.at[-1].set(waveSpeed(v1.A[-1], v1.gamma[-1]))
    v2.c = v2.c.at[0].set(waveSpeed(v2.A[0], v2.gamma[0]))
    v3.c = v3.c.at[0].set(waveSpeed(v3.A[0], v3.gamma[0]))

    return v1, v2, v3
