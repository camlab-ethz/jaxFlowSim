import jax
import jax.numpy as jnp
from src.newton import newtonRaphson
from src.utils import pressure, waveSpeed
import src.initialise as ini

@jax.jit
def solveConjunction(v1, v2):
    U0 = jnp.array([v1.u[-1], v2.u[0], jnp.sqrt(jnp.sqrt(v1.A[-1])), jnp.sqrt(jnp.sqrt(v2.A[0]))])

    k1 = v1.s_15_gamma[-1]
    k2 = v2.s_15_gamma[0]
    k3 = ini.BLOOD.rho
    k = jnp.array([k1, k2, k3])

    J = calculateJacobianConjunction(v1, v2, U0, k)
    U = newtonRaphson([v1, v2], J, U0, k, calculateWStarConjunction, calculateFConjunction)[0]

    return updateConjunction(U, v1, v2)


@jax.jit
def calculateJacobianConjunction(v1, v2, U, k):
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
    J43 =  2.0 * v1.beta[-1] * U[2] * v1.s_inv_A0[-1]
    J44 = -2.0 * v2.beta[0] * U[3] * v2.s_inv_A0[0]

    return jnp.array([[1.0, 0.0, J13, 0.0],
                      [0.0, 1.0, 0.0, J24],
                      [J31, J32, J33, J34],
                      [J41, J42, J43, J44]])


@jax.jit
def calculateWStarConjunction(U, k):
    W1 = U[0] + 4.0 * k[0] * U[2]
    W2 = U[1] - 4.0 * k[1] * U[3]

    return jnp.array([W1, W2])


@jax.jit
def calculateFConjunction(vessels, U, k, W):
    v1 = vessels[0]
    v2 = vessels[1]

    U32 = U[2]**2
    U42 = U[3]**2

    f1 = U[0] + 4.0 * k[0] * U[2] - W[0]
    f2 = U[1] - 4.0 * k[1] * U[3] - W[1]
    f3 = U[0] * U32**2 - U[1] * U42**2

    f4 = 0.5 * k[2] * U[0]**2 + v1.beta[-1] * (U32 * v1.s_inv_A0[-1] - 1.0) - (0.5 * k[2] * U[1]**2 + v2.beta[0] * (U42 * v2.s_inv_A0[0] - 1.0))

    return jnp.array([f1, f2, f3, f4])


@jax.jit
def updateConjunction(U, v1, v2):
    v1.u = v1.u.at[-1].set(U[0])
    v2.u = v2.u.at[0].set(U[1])

    v1.A = v1.A.at[-1].set(U[2]**4)
    v1.Q = v1.Q.at[-1].set(v1.u[-1] * v1.A[-1])

    v2.A = v2.A.at[0].set(U[3]**4)
    v2.Q = v2.Q.at[0].set(v2.u[0] * v2.A[0])

    v1.P = v1.P.at[-1].set(pressure(v1.A[-1], v1.A0[-1], v1.beta[-1], v1.Pext))
    v2.P = v2.P.at[0].set(pressure(v2.A[0], v2.A0[0], v2.beta[0], v2.Pext))

    v1.c = v1.c.at[-1].set(waveSpeed(v1.A[-1], v1.gamma[-1]))
    v2.c = v2.c.at[0].set(waveSpeed(v2.A[0], v2.gamma[0]))

    return v1, v2
