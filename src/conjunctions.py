import jax
from jax import jit
import jax.numpy as jnp
from src.newton import newtonRaphson
import src.initialise as ini

@jit
def solveConjunction(u1, u2, A1, A2, 
                     A01, A02, beta1, beta2, 
                     gamma1, gamma2):
    U0 = jnp.array((u1, u2, jnp.sqrt(jnp.sqrt(A1)), jnp.sqrt(jnp.sqrt(A2))), dtype=jnp.float64)

    k1 = jnp.sqrt(1.5*gamma1)
    k2 = jnp.sqrt(1.5*gamma2)
    k3 = ini.BLOOD.rho
    k = jnp.array([k1, k2, k3])

    J = calculateJacobianConjunction(U0, k, 
                                     A01, A02, 
                                     beta1, beta2)
    U = newtonRaphson(calculateWStarConjunction, calculateFConjunction, 
                      J, U0, k,
                      (A01, A02),
                      (beta1, beta2))[0]

    return updateConjunction(U)


@jit
def calculateJacobianConjunction(U, k, 
                                 A01, A02, 
                                 beta1, beta2):
    U33 = U[2]*U[2]*U[2]
    U43 = U[3]*U[3]*U[3]

    J13 =  4.0 * k[0]
    J24 = -4.0 * k[1]

    J31 =  U33 * U[2]
    J32 = -U43 * U[3]
    J33 =  4.0 * U[0] * U33
    J34 = -4.0 * U[1] * U43

    J41 =  k[2] * U[0]
    J42 = -k[2] * U[1]
    J43 =  2.0 * beta1  * U[2] * jnp.sqrt(1/A01)
    J44 = -2.0 * beta2 * U[3] * jnp.sqrt(1/A02)

    return jnp.array([[1.0, 0.0, J13, 0.0],
                      [0.0, 1.0, 0.0, J24],
                      [J31, J32, J33, J34],
                      [J41, J42, J43, J44]], dtype=jnp.float64)


@jax.jit
def calculateWStarConjunction(U, k):
    W1 = U[0] + 4.0 * k[0] * U[2]
    W2 = U[1] - 4.0 * k[1] * U[3]

    return jnp.array([W1, W2], dtype=jnp.float64)


@jax.jit
def calculateFConjunction(U, k, W,
                          A0s,
                          betas):
    
    A01, A02 = A0s
    beta1, beta2, = betas

    U32 = U[2]*U[2]
    U42 = U[3]*U[3]

    f1 = U[0] + 4.0 * k[0] * U[2] - W[0]
    f2 = U[1] - 4.0 * k[1] * U[3] - W[1]
    f3 = U[0] * U32*U32 - U[1] * U42*U42

    f4 = 0.5 * k[2] * U[0]**2 + beta1 * (U32 * jnp.sqrt(1/A01) - 1.0) - (0.5 * k[2] * U[1]**2 + beta2 * (U42 * jnp.sqrt(1/A02) - 1.0))

    return jnp.array([f1, f2, f3, f4], dtype=jnp.float64)


@jax.jit
def updateConjunction(U):
    u1 = U[0]
    u2 = U[1]

    A1 = U[2]*U[2]*U[2]*U[2]
    Q1 = u1 * A1

    A2  = U[3]*U[3]*U[3]*U[3]
    Q2  = u2 * A2

    return Q1, Q2, A1, A2
