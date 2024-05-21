import jax.numpy as jnp
from jax import lax
from src.newton import newtonRaphson
from src.utils import pressure, waveSpeed

def solveConjunctionWrapper(dt, sim_dat, sim_dat_aux, 
                       sim_dat_const,
                       start, end, i):
    index1 = end
    #debug.print("{x}", x = (rho, i, index1, index2, index3))
    d_i_start = start
    u1 = sim_dat[0,index1]
    u2 = sim_dat[0,d_i_start]
    A1 = sim_dat[2,index1]
    A2 = sim_dat[2,d_i_start]
    (u1, u2, Q1, Q2, 
     A1, A2, c1, c2, P1, P2) = solveConjunction(u1, u2, 
                                                A1, A2,
                                                *sim_dat_const)
    temp1 = jnp.array((u1, Q1, A1, c1, P1))
    temp2 = jnp.array((u2, Q2, A2, c2, P2))
    return temp1, temp2, temp2, sim_dat_aux[i,2]

def solveConjunction(u1, u2, A1, 
                     A2, A01, A02, 
                     beta1, beta2, gamma1, 
                     gamma2, Pext1, Pext2,
                     rho):
    U0 = jnp.array((u1, u2, jnp.sqrt(jnp.sqrt(A1)), jnp.sqrt(jnp.sqrt(A2))))

    k1 = jnp.sqrt(1.5*gamma1)
    k2 = jnp.sqrt(1.5*gamma2)
    k3 = rho
    k = jnp.array([k1, k2, k3])

    J = calculateJacobianConjunction(U0, k, 
                                     A01, A02, 
                                     beta1, beta2)
    U = newtonRaphson(#calculateWStarConjunction, 
                      calculateFConjunction, 
                      J, U0,# k,
                      (A01, A02),
                      (beta1, beta2))
    
    #debug.print("{x}", x = counter)
    
    #def F_wrapper(U, args):
    #    k, A01, A02, beta1, beta2 = args
    #    return calculateFConjunction(U, k, calculateWStarConjunction(U, k), (A01, A02), (beta1, beta2))

    #solver = optx.LevenbergMarquardt(rtol=1e-2, atol=1e-2)#, lower=-jnp.ones(4)*2, upper=jnp.ones(4)*2)
    
    #U = optx.root_find(F_wrapper, solver, U0, (k, A01, A02, beta1, beta2), max_steps=1000000000).value

    return updateConjunction(U,
                             A01, A02,
                             beta1, beta2,
                             gamma1, gamma2,
                             Pext1, Pext2)


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

    #J41 =  k[2] * U[0]
    #J42 = -k[2] * U[1]
    J43 =  2.0 * beta1  * U[2] * jnp.sqrt(1/A01)
    J44 = -2.0 * beta2 * U[3] * jnp.sqrt(1/A02)

    return jnp.array([[1.0, 0.0, J13, 0.0],
                      [0.0, 1.0, 0.0, J24],
                      [J31, J32, J33, J34],
                      [0.0, 0.0, J43, J44]])
                      #[J41, J42, J43, J44]], dtype=jnp.float64)


#def calculateWStarConjunction(U, k):
#    W1 = U[0] + 4.0 * k[0] * U[2]
#    W2 = U[1] - 4.0 * k[1] * U[3]
#
#    return jnp.array([W1, W2], dtype=jnp.float64)


def calculateFConjunction(U,# k, W,
                          A0s,
                          betas):
    
    A01, A02 = A0s
    beta1, beta2, = betas

    U32 = U[2]*U[2]
    U42 = U[3]*U[3]

    f1 =  0 #U[0] + 4.0 * k[0] * U[2] - W[0]
    f2 =  0 #U[1] - 4.0 * k[1] * U[3] - W[1]
    f3 = U[0] * U32*U32 - U[1] * U42*U42

    #f4 = 0.5 * k[2] * U[0]**2 + beta1 * (U32 * jnp.sqrt(1/A01) - 1.0) - (0.5 * k[2] * U[1]**2 + beta2 * (U42 * jnp.sqrt(1/A02) - 1.0))
    f4 =  beta1 * (U32 * jnp.sqrt(1/A01) - 1.0) -  + beta2 * (U42 * jnp.sqrt(1/A02) - 1.0)

    return jnp.array([f1, f2, f3, f4], dtype=jnp.float64)


def updateConjunction(U,
                      A01, A02, 
                      beta1, beta2,
                      gamma1, gamma2,
                      Pext1, Pext2):
    u1 = U[0]
    u2 = U[1]

    A1 = U[2]*U[2]*U[2]*U[2]
    Q1 = u1 * A1

    A2  = U[3]*U[3]*U[3]*U[3]
    Q2  = u2 * A2

    P1 = pressure(A1, A01, beta1, Pext1)
    P2 = pressure(A2, A02, beta2, Pext2)

    c1 = waveSpeed(A1, gamma1)
    c2 = waveSpeed(A2, gamma2)

    return u1, u2, Q1, Q2, A1, A2, c1, c2, P1, P2