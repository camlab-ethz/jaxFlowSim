import jax.numpy as jnp
from functools import partial
from jax import lax, jit
from src.utils import pressure, waveSpeed
from src.newton import newtonRaphson

def solveBifurcationWrapper(dt, sim_dat, sim_dat_aux, 
                       sim_dat_const, sim_dat_const_aux, 
                       edges, starts, rho, B, ends, i, index2, index3):
    index1 = ends[i]
    d1_i = edges[i,4]
    d2_i = edges[i,5]
    d1_i_start = starts[d1_i] 
    d2_i_start = starts[d2_i] 
    u1 = sim_dat[0,index1]
    u2 = sim_dat[0,d1_i_start]
    u3 = sim_dat[0,d2_i_start]
    A1 = sim_dat[2,index1]
    A2 = sim_dat[2,d1_i_start]
    A3 = sim_dat[2,d2_i_start]
    (u1, u2, u3, 
     Q1, Q2, Q3, 
     A1, A2, A3, 
     c1, c2, c3, 
     P1, P2, P3) = solveBifurcation(u1, u2, u3, 
                                    A1, A2, A3,
                                    sim_dat_const[0,index1],
                                    sim_dat_const[0,d1_i_start],
                                    sim_dat_const[0,d2_i_start],
                                    sim_dat_const[1,index1],
                                    sim_dat_const[1,d1_i_start],
                                    sim_dat_const[1,d2_i_start],
                                    sim_dat_const[2,index1],
                                    sim_dat_const[2,d1_i_start],
                                    sim_dat_const[2,d2_i_start],
                                    sim_dat_const[4, index1],
                                    sim_dat_const[4, d1_i_start],
                                    sim_dat_const[4, d2_i_start],
                                    )
    temp1 = jnp.array((u1, Q1, A1, c1, P1))
    temp2 = jnp.array((u2, Q2, A2, c2, P2))
    temp3 = jnp.array((u3, Q3, A3, c3, P3))
    sim_dat = lax.dynamic_update_slice( 
        sim_dat, 
        temp1[:,jnp.newaxis]*jnp.ones(3)[jnp.newaxis,:],
        (0,index1))
    sim_dat = lax.dynamic_update_slice( 
        sim_dat, 
        temp2[:,jnp.newaxis]*jnp.ones(3)[jnp.newaxis,:],
        (0,d1_i_start-2))
    sim_dat = lax.dynamic_update_slice( 
        sim_dat, 
        temp3[:,jnp.newaxis]*jnp.ones(3)[jnp.newaxis,:],
        (0,d2_i_start-2))
    return sim_dat, sim_dat_aux

def solveBifurcation(u1, u2, u3, 
                     A1, A2, A3, 
                     A01, A02, A03, 
                     beta1, beta2, beta3,
                     gamma1, gamma2, gamma3,
                     Pext1, Pext2, Pext3):
    U0 = jnp.array([u1,
                   u2,
                   u3,
                   jnp.sqrt(jnp.sqrt(A1)),
                   jnp.sqrt(jnp.sqrt(A2)),
                   jnp.sqrt(jnp.sqrt(A3))])

    k = jnp.array([jnp.sqrt(1.5*gamma1),
                   jnp.sqrt(1.5*gamma2),
                   jnp.sqrt(1.5*gamma3)])

    J = calculateJacobianBifurcation(U0, k,
                                     A01, A02, A03,
                                     beta1, beta2, beta3)
    U = newtonRaphson(#calculateWstarBifurcation, 
                      calculateFBifurcation, 
                      J, U0, 
                      (A01, A02, A03),
                      (beta1, beta2, beta3))

    return updateBifurcation(U,
                             A01, A02, A03,
                             beta1, beta2, beta3,
                             gamma1, gamma2, gamma3,
                             Pext1, Pext2, Pext3)

def calculateJacobianBifurcation(U, k,
                                 A01, A02, A03,
                                 beta1, beta2, beta3):
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

    J54 = 2.0 * beta1 * U[3] * 1/jnp.sqrt(A01)
    J55 = -2.0 * beta2 * U[4] * 1/jnp.sqrt(A02)

    J64 = 2.0 * beta1 * U[3] * 1/jnp.sqrt(A01)
    J66 = -2.0 * beta3 * U[5] * 1/jnp.sqrt(A03)

    return jnp.array([[1.0, 0.0, 0.0, J14, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, J25, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, J36],
                      [J41, J42, J43, J44, J45, J46],
                      [0.0, 0.0, 0.0, J54, J55, 0.0],
                      [0.0, 0.0, 0.0, J64, 0.0, J66]])


#def calculateWstarBifurcation(U, k):
#    W1 = U[0] + 4.0 * k[0] * U[3]
#    W2 = U[1] - 4.0 * k[1] * U[4]
#    W3 = U[2] - 4.0 * k[2] * U[5]
#
#    return jnp.array([W1, W2, W3])


def calculateFBifurcation(U,# k, W,
                          A0s,
                          betas):

    beta1, beta2, beta3 = betas
    A01, A02, A03 = A0s

    U42 = U[3]**2
    U52 = U[4]**2
    U62 = U[5]**2
    U42 = U[3]*U[3]
    U52 = U[4]*U[4]
    U62 = U[5]*U[5]

    f1 = 0 #U[0] + 4.0 * k[0] * U[3] - W[0]
    f2 = 0 #U[1] - 4.0 * k[1] * U[4] - W[1]
    f3 = 0 #U[2] - 4.0 * k[2] * U[5] - W[2]
    f4 = U[0] * (U42*U42) - U[1] * (U52*U52) - U[2] * (U62*U62)

    f5 = beta1 * (U42 * jnp.sqrt(1/A01) - 1.0) - (beta2 * (U52 * jnp.sqrt(1/A02) - 1.0))
    f6 = beta1 * (U42 * jnp.sqrt(1/A01) - 1.0) - (beta3 * (U62 * jnp.sqrt(1/A03) - 1.0))

    return jnp.array([f1, f2, f3, f4, f5, f6])


def updateBifurcation(U,
                  A01, A02, A03,
                  beta1, beta2, beta3, 
                  gamma1, gamma2, gamma3,
                  Pext1, Pext2, Pext3):
    u1 = U[0]
    u2 = U[1]
    u3 = U[2]

    A1 = U[3]*U[3]*U[3]*U[3]
    A2 = U[4]*U[4]*U[4]*U[4]
    A3 = U[5]*U[5]*U[5]*U[5]

    Q1 = u1 * A1
    Q2 = u2 * A2
    Q3 = u3 * A3

    P1 = pressure(A1, A01, beta1, Pext1)
    P2 = pressure(A2, A02, beta2, Pext2)
    P3 = pressure(A3, A03, beta3, Pext3)

    c1 = waveSpeed(A1, gamma1)
    c2 = waveSpeed(A2, gamma2)
    c3 = waveSpeed(A3, gamma3)

    return u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3
