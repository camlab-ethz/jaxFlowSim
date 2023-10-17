from functools import partial
import jax
import jax.numpy as jnp
from src.anastomosis import solveAnastomosis
from src.conjunctions import solveConjunction
from src.bifurcations import solveBifurcation
from src.boundary_conditions import setInletBC, setOutletBC
from src.utils import pressureSA, waveSpeedSA
import src.initialise as ini
#from jax.sharding import Mesh, PartitionSpec
#from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map


#devices = mesh_utils.create_device_mesh((20))
#mesh = Mesh(devices, axis_names=('i'))



#@jax.jit
@partial(jax.jit, static_argnums=(0, 1))
def calculateDeltaT(M, N, Ccfl, u, c, dx):
    dt = 1.0
    def body_fun(i,dt):
        start = i*M
        Smax = jnp.max(jnp.abs(jax.lax.dynamic_slice_in_dim(u,start,M) + jax.lax.dynamic_slice_in_dim(c,start,M)))
        vessel_dt = dx[i] * Ccfl / Smax
        dt = jax.lax.cond(dt > vessel_dt, lambda: vessel_dt, lambda: dt)
        return dt
    dt = jax.lax.fori_loop(0, N, body_fun, dt)
    return dt



#@jax.jit
@partial(jax.jit, static_argnums=(0, 1))
def solveModel(M, N, t, dt, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, input_data, rho):

    inlet = sim_dat_const_aux[0,4] 
    u0 = sim_dat[0,0]
    u1 = sim_dat[0,1]
    A0 = sim_dat[2,0]
    c0 = sim_dat[3,0]
    c1 = sim_dat[3,1]
    cardiac_T = sim_dat_const_aux[0,1]
    dx = sim_dat_const_aux[0,0]
    A00 = sim_dat_const[0,0]
    beta0 = sim_dat_const[1,0]
    Pext = sim_dat_const_aux[0,2]
    sim_dat = sim_dat.at[1:3,0].set(jnp.array(setInletBC(inlet, u0, u1, A0, 
                        c0, c1, t, dt, 
                        input_data, cardiac_T, 1/dx, A00, 
                        beta0, Pext)).transpose())
    #_Q, _A = setInletBC(inlet, u0, u1, A0, 
    #                    c0, c1, t, dt, 
    #                    input_data, cardiac_T, 1/dx, A00, 
    #                    beta0, Pext)
    #sim_dat = sim_dat.at[1,0].set(_Q)
    #sim_dat = sim_dat.at[2,0].set(_A)



    def body_fun1(j, dat):
        (t, dt, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, input_data, rho) = dat
        i = edges[j,0]-1
        start = i*M
        size = input_data.shape[1]
        sim_dat = jax.lax.dynamic_update_slice(
            sim_dat,
            solveVessel(M, dt, t, 
                        sim_dat[0,start], sim_dat[0,start+1], 
                        jax.lax.dynamic_slice(sim_dat, (1,start), (1,M)).flatten(),
                        jax.lax.dynamic_slice(sim_dat, (2,start), (1,M)).flatten(),
                        sim_dat[3,start], sim_dat[3,start+1], 
                        sim_dat_aux[i,2], 
                        sim_dat_aux[i,3], 
                        sim_dat_aux[i,6], 
                        sim_dat_aux[i,7],
                        sim_dat_const_aux[i,0], 
                        sim_dat_const_aux[i,1], 
                        sim_dat_const_aux[i,2], 
                        sim_dat_const_aux[i,3], 
                        sim_dat_const_aux[i,4], 
                        jax.lax.dynamic_slice(sim_dat_const, (0,start), (1,M)).flatten(), 
                        jax.lax.dynamic_slice(sim_dat_const, (1,start), (1,M)).flatten(), 
                        jax.lax.dynamic_slice(sim_dat_const, (2,start), (1,M)).flatten(), 
                        jax.lax.dynamic_slice(sim_dat_const, (3,start), (1,M)).flatten(),
                        jax.lax.dynamic_slice(input_data, (2*i,0), (2,size))), 
            (0,start))

        return (t, dt, sim_dat, sim_dat_aux, 
                sim_dat_const, sim_dat_const_aux, 
                edges, input_data, rho)

    def body_fun2(j, dat):
        (t, dt, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, input_data, rho) = dat
        i = edges[j,0]-1
        end = (i+1)*M
        
        def setOutletBC_wrapper(sim_dat, sim_dat_aux):
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,end-2]
            Q1 = sim_dat[1,end-1]
            A1 = sim_dat[2,end-1]
            c1 = sim_dat[3,end-1]
            c2 = sim_dat[3,end-2]
            P1 = sim_dat[4,end-1]
            P2 = sim_dat[4,end-2]
            P3 = sim_dat[4,end-3]
            Pc = sim_dat_aux[i,10]
            W1M0 = sim_dat_aux[i,0]
            W2M0 = sim_dat_aux[i,1]
            u, Q, A, c, P1, Pc = setOutletBC(dt,
                                             u1, u2, Q1, A1, c1, c2, 
                                             P1, P2, P3, Pc, W1M0, W2M0,
                                             sim_dat_const[0,end-1],
                                             sim_dat_const[1,end-1],
                                             sim_dat_const[2,end-1],
                                             sim_dat_const_aux[i,0],
                                             sim_dat_const_aux[i,2],
                                             sim_dat_const_aux[i,5], 
                                             sim_dat_const_aux[i,6],
                                             sim_dat_const_aux[i,7],
                                             sim_dat_const_aux[i,8],
                                             sim_dat_const_aux[i,9])
                                             #beta[i], gamma[i], A0[i,M-1])
            sim_dat = sim_dat.at[0,end-1].set(u)
            sim_dat = sim_dat.at[1,end-1].set(Q)
            sim_dat = sim_dat.at[2,end-1].set(A)
            sim_dat = sim_dat.at[3,end-1].set(c)
            sim_dat = sim_dat.at[4,end-1].set(P1)
            sim_dat_aux = sim_dat_aux.at[i,10].set(Pc)
            #sim_dat_aux_out = sim_dat_aux
            #sim_dat_aux_out[i,10] = Pc
            return sim_dat, sim_dat_aux

        (sim_dat, 
         sim_dat_aux) = jax.lax.cond(sim_dat_const_aux[i,5] != 0,
                                    lambda x, y: setOutletBC_wrapper(x,y), 
                                    lambda x, y: (x,y), sim_dat, sim_dat_aux)



        def solveBifurcation_wrapper(sim_dat):
            d1_i = edges[j,4]
            d2_i = edges[j,5]
            d1_i_start = d1_i*M #mesh_sizes[d1_i]
            d2_i_start = d2_i*M #mesh_sizes[d2_i]
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,d1_i_start]
            u3 = sim_dat[0,d2_i_start]
            A1 = sim_dat[2,end-1]
            A2 = sim_dat[2,d1_i_start]
            A3 = sim_dat[2,d2_i_start]
            (u1, u2, u3, 
             Q1, Q2, Q3, 
             A1, A2, A3, 
             c1, c2, c3, 
             P1, P2, P3) = solveBifurcation(u1, u2, u3, 
                                            A1, A2, A3,
                                            sim_dat_const[0,end-1],
                                            sim_dat_const[0,d1_i_start],
                                            sim_dat_const[0,d2_i_start],
                                            sim_dat_const[1,end-1],
                                            sim_dat_const[1,d1_i_start],
                                            sim_dat_const[1,d2_i_start],
                                            sim_dat_const[2,end-1],
                                            sim_dat_const[2,d1_i_start],
                                            sim_dat_const[2,d2_i_start],
                                            sim_dat_const_aux[i, 2],
                                            sim_dat_const_aux[d1_i, 2],
                                            sim_dat_const_aux[d2_i, 2],
                                            )
            sim_dat = sim_dat.at[0,end-1].set(u1) 
            sim_dat = sim_dat.at[0,d1_i_start].set(u2)    
            sim_dat = sim_dat.at[0,d2_i_start].set(u3)
            sim_dat = sim_dat.at[1,end-1].set(Q1)
            sim_dat = sim_dat.at[1,d1_i_start].set(Q2)
            sim_dat = sim_dat.at[1,d2_i_start].set(Q3)
            sim_dat = sim_dat.at[2,end-1].set(A1)
            sim_dat = sim_dat.at[2,d1_i_start].set(A2)
            sim_dat = sim_dat.at[2,d2_i_start].set(A3)
            sim_dat = sim_dat.at[3,end-1].set(c1)
            sim_dat = sim_dat.at[3,d1_i_start].set(c2)
            sim_dat = sim_dat.at[3,d2_i_start].set(c3)
            sim_dat = sim_dat.at[4,end-1].set(P1)
            sim_dat = sim_dat.at[4,d1_i_start].set(P2)
            sim_dat = sim_dat.at[4,d2_i_start].set(P3)

            return sim_dat

        sim_dat = jax.lax.cond((sim_dat_const_aux[i,5] == 0) * (edges[j,3] == 2),
                                    lambda x: solveBifurcation_wrapper(x), 
                                    lambda x: x, sim_dat)

        #elif :
        def solveConjunction_wrapper(sim_dat, rho):
            d_i = edges[j,7]
            d_i_start = d_i*M
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,d_i_start]
            A1 = sim_dat[2,end-1]
            A2 = sim_dat[2,d_i_start]
            (u1, u2, Q1, Q2, 
             A1, A2, c1, c2, P1, P2) = solveConjunction(u1, u2, 
                                                        A1, A2,
                                                        sim_dat_const[0,end-1],
                                                        sim_dat_const[0,d_i_start],
                                                        sim_dat_const[1,end-1],
                                                        sim_dat_const[1,d_i_start],
                                                        sim_dat_const[2,end-1],
                                                        sim_dat_const[2,d_i_start],
                                                        sim_dat_const_aux[i, 2],
                                                        sim_dat_const_aux[d_i, 2],
                                                        rho)
            sim_dat = sim_dat.at[0,end-1].set(u1)
            sim_dat = sim_dat.at[0,d_i_start].set(u2)
            sim_dat = sim_dat.at[1,end-1].set(Q1)
            sim_dat = sim_dat.at[1,d_i_start].set(Q2)
            sim_dat = sim_dat.at[2,end-1].set(A1)
            sim_dat = sim_dat.at[2,d_i_start].set(A2)
            sim_dat = sim_dat.at[3,end-1].set(c1)
            sim_dat = sim_dat.at[3,d_i_start].set(c2)
            sim_dat = sim_dat.at[4,end-1].set(P1)
            sim_dat = sim_dat.at[4,d_i_start].set(P2)

            return sim_dat

        sim_dat = jax.lax.cond((sim_dat_const_aux[i,5] == 0) * 
                               (edges[j,3] != 2) *
                               (edges[j,6] == 1),
                                lambda x, y: solveConjunction_wrapper(x, y), 
                                lambda x, y: x, sim_dat, rho)

        #elif edges[j,6] == 2:                                           
        def solveAnastomosis_wrapper(sim_dat):
            p1_i = edges[j,7]
            p2_i = edges[j,8]
            d = edges[j,9]
            p1_i_end = (p1_i+1)*M
            d_start = d*M
            u1 = sim_dat[0,end-1]
            u2 = sim_dat[0,p1_i_end-1]
            u3 = sim_dat[0,d_start]
            Q1 = sim_dat[1,end-1]
            Q2 = sim_dat[1,p1_i_end-1]
            Q3 = sim_dat[1,d_start]
            A1 = sim_dat[2,end-1]
            A2 = sim_dat[2,p1_i_end-1]
            A3 = sim_dat[2,d_start]
            c1 = sim_dat[3,end-1]
            c2 = sim_dat[3,p1_i_end-1]
            c3 = sim_dat[3,d_start]
            P1 = sim_dat[4,end-1]
            P2 = sim_dat[4,p1_i_end-1]
            P3 = sim_dat[4,d_start]
            u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3 = jax.lax.cond(
                jnp.maximum(p1_i, p2_i) == i, 
                lambda: solveAnastomosis(u1, u2, u3, 
                                         A1, A2, A3,
                                         sim_dat_const[0,end-1],
                                         sim_dat_const[0,p1_i_end-1],
                                         sim_dat_const[0,d_start],
                                         sim_dat_const[1,end-1],
                                         sim_dat_const[1,p1_i_end-1],
                                         sim_dat_const[1,d_start],
                                         sim_dat_const[2,end-1],
                                         sim_dat_const[2,p1_i_end-1],
                                         sim_dat_const[2,d_start],
                                         sim_dat_const_aux[i, 2],
                                         sim_dat_const_aux[p1_i, 2],
                                         sim_dat_const_aux[d, 2],
                                        ), 
                lambda: (u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3))
            sim_dat = sim_dat.at[0,end-1].set(u1)
            sim_dat = sim_dat.at[0,p1_i_end-1].set(u2)
            sim_dat = sim_dat.at[0,d_start].set(u3)
            sim_dat = sim_dat.at[1,end-1].set(Q1)
            sim_dat = sim_dat.at[1,p1_i_end-1].set(Q2)
            sim_dat = sim_dat.at[1,d_start].set(Q3)
            sim_dat = sim_dat.at[2,end-1].set(A1)
            sim_dat = sim_dat.at[2,p1_i_end-1].set(A2)
            sim_dat = sim_dat.at[2,d_start].set(A3)
            sim_dat = sim_dat.at[3,end-1].set(c1)
            sim_dat = sim_dat.at[3,p1_i_end-1].set(c2)
            sim_dat = sim_dat.at[3,d_start].set(c3)
            sim_dat = sim_dat.at[4,end-1].set(P1)
            sim_dat = sim_dat.at[4,p1_i_end-1].set(P2)
            sim_dat = sim_dat.at[4,d_start].set(P3)

            return sim_dat
        
        sim_dat = jax.lax.cond((sim_dat_const_aux[i,5] == 0) * 
                               (edges[j,3] != 2) *
                               (edges[j,6] == 2),
                                lambda x: solveAnastomosis_wrapper(x), 
                                lambda x: x, sim_dat)

        return (t, dt, sim_dat, sim_dat_aux, 
                sim_dat_const, sim_dat_const_aux, 
                edges, input_data, rho)
    #for j in np.arange(0,,1):
    
    #def cond_fun(dat):
    #    _, _, j = dat
    #    return j < ini.EDGES.edges.shape[0]


    #(sim_dat, sim_dat_aux), _ = jax.lax.scan(body_fun, (sim_dat, sim_dat_aux), jnp.arange(ini.NUM_VESSELS))
    (_, _, sim_dat, sim_dat_aux, _, _, _, _, _)  = jax.lax.fori_loop(0, N, body_fun1, 
                                                   (t, dt, sim_dat, sim_dat_aux, 
                                                    sim_dat_const, sim_dat_const_aux, 
                                                    edges, input_data, rho))
    (_, _, sim_dat, sim_dat_aux, _, _, _, _, _)  = jax.lax.fori_loop(0, N, body_fun2, 
                                                   (t, dt, sim_dat, sim_dat_aux, 
                                                    sim_dat_const, sim_dat_const_aux, 
                                                    edges, input_data, rho))

    
    return sim_dat, sim_dat_aux


#@jax.jit
@partial(jax.jit, static_argnums=0)
def solveVessel(M, dt, t,
                u0, u1, Q, A, 
                c0, c1, U00Q, U00A, UM1Q, UM1A, 
                dx, cardiac_T, Pext, viscT, inlet,
                A0, beta, gamma, wallE,
                input_data):
    #_Q, _A = jax.lax.cond(inlet > 0, lambda: setInletBC(inlet, u0, u1, A[0], c0, c1, t, dt, input_data, cardiac_T, 1/dx, A0[0], beta[0], Pext), lambda: (Q[0],A[0]))
    #Q = Q.at[0].set(_Q)
    #A = A.at[0].set(_A)

    #if inlet > 0:
    #    Q0, A0 = setInletBC(i, u0, u1, A[0], c0, c1, t, dt)
    #    Q = Q.at[0].set(Q0)
    #    A = A.at[0].set(A0)

    return muscl(M, U00Q, U00A, 
                UM1Q, UM1A, Q, A, A0,
                dt, dx, beta, Pext, gamma, viscT, wallE)

#@jax.jit
#@partial(shard_map, mesh=mesh, in_specs=P('i', 'j'),
#         out_specs=P('i'))
@partial(jax.jit, static_argnums=(0,))
def muscl(M, U00Q, U00A, UM1Q, UM1A, Q, A, A0, dt, dx, beta, Pext, gamma, viscT, wallE):
    M = ini.MESH_SIZE
    #s_A0 = jax.block_until_ready(shard_map(lambda a: jnp.sqrt(a), mesh, in_specs=PartitionSpec('i'), out_specs=PartitionSpec('i'))(A0))
    #jax.debug.print("{x}", x = s_A0)


    s_A0 = jax.vmap(lambda a: jnp.sqrt(a))(A0)
    #s_A0 = jnp.sqrt(A0)
    s_inv_A0 = jax.vmap(lambda a: 1/jnp.sqrt(a))(A0)
    #s_inv_A0 = 1/jnp.sqrt(A0)
    halfDx = 0.5*dx
    invDx = 1/dx
    gamma_ghost = jnp.zeros(M+2)
    gamma_ghost = gamma_ghost.at[1:M+1].set(gamma)
    gamma_ghost = gamma_ghost.at[0].set(gamma[0])
    gamma_ghost = gamma_ghost.at[-1].set(gamma[-1])
    vA = jnp.empty(M+2)
    vQ = jnp.empty(M+2)
    vA = vA.at[0].set(U00A)
    vA = vA.at[-1].set(UM1A)

    vQ = vQ.at[0].set(U00Q)
    vQ = vQ.at[-1].set(UM1Q)
    vA = vA.at[1:M+1].set(A)
    vQ = vQ.at[1:M+1].set(Q)
    #vA = jnp.concatenate((jnp.array([U00A]),A,jnp.array([UM1A])))
    #vQ = jnp.concatenate((jnp.array([U00Q]),Q,jnp.array([UM1Q])))

    slopeA_halfDx = computeLimiter(vA, invDx) * halfDx
    slopeQ_halfDx = computeLimiter(vQ, invDx) * halfDx

    #slopeA_halfDx = slopesA * ini.VCS[i].halfDx
    #slopeQ_halfDx = slopesQ * ini.VCS[i].halfDx
    
    Al = jax.vmap(lambda a, b: a+b)(vA, slopeA_halfDx)
    Ar = jax.vmap(lambda a, b: a-b)(vA, slopeA_halfDx)
    Ql = jax.vmap(lambda a, b: a+b)(vQ, slopeQ_halfDx)
    Qr = jax.vmap(lambda a, b: a-b)(vQ, slopeQ_halfDx)
    #Al = shard_map(lambda a, b: a+b, mesh, in_specs=PartitionSpec('i'), out_specs=PartitionSpec('i'))(vA, slopeA_halfDx)
    #Ar = shard_map(lambda a, b: a-b, mesh, in_specs=PartitionSpec('i'), out_specs=PartitionSpec('i'))(vA, slopeA_halfDx)
    #Ql = shard_map(lambda a, b: a+b, mesh, in_specs=PartitionSpec('i'), out_specs=PartitionSpec('i'))(vQ, slopeQ_halfDx)
    #Qr = shard_map(lambda a, b: a-b, mesh, in_specs=PartitionSpec('i'), out_specs=PartitionSpec('i'))(vQ, slopeQ_halfDx)
    #Al = vA + slopeA_halfDx
    #Ar = vA - slopeA_halfDx
    #Ql = vQ + slopeQ_halfDx
    #Qr = vQ - slopeQ_halfDx

    Fl = jnp.array(jax.vmap(computeFlux_par)(gamma_ghost, Al, Ql))
    Fr = jnp.array(jax.vmap(computeFlux_par)(gamma_ghost, Ar, Qr))
    #Fl = computeFlux(gamma_ghost, Al, Ql)
    #Fr = computeFlux(gamma_ghost, Ar, Qr)

    dxDt = dx / dt
    
    invDxDt = dt / dx

    flux = jnp.empty((2,M+2))
    flux = flux.at[0,0:M+1].set(jax.vmap(lambda a, b, c, d: 0.5*(a+b - dxDt*(c-d)))(Fr[0, 1:M+2], Fl[0, 0:M+1], Ar[1:M+2], Al[0:M+1]))
    flux = flux.at[1,0:M+1].set(jax.vmap(lambda a, b, c, d: 0.5*(a+b - dxDt*(c-d)))(Fr[1, 1:M+2], Fl[1, 0:M+1], Qr[1:M+2], Ql[0:M+1]))
    #flux = flux.at[0,0:M+1].set(0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])))
    #flux = flux.at[1,0:M+1].set(0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1])))
    #flux = jnp.stack((0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])), 
    #                  0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1]))), dtype=jnp.float64)

    uStar = jnp.empty((2,M+2))
    uStar = uStar.at[0,1:M+1].set(jax.vmap(lambda a, b, c: a+invDxDt*(b-c))(vA[1:M+1],
                                                             flux[0,0:M],
                                                             flux[0,1:M+1]))
    uStar = uStar.at[1,1:M+1].set(jax.vmap(lambda a, b, c: a+invDxDt*(b-c))(vQ[1:M+1],
                                                             flux[1,0:M],
                                                             flux[1,1:M+1]))
    #uStar = uStar.at[0,1:M+1].set(jax.vmap(lambda a, b: a-invDxDt*b)(vA[1:M+1],
    #                                                         jnp.diff(flux[0,0:M+1])))
    #uStar = uStar.at[1,1:M+1].set(jax.vmap(lambda a, b: a-invDxDt*b)(vQ[1:M+1],
    #                                                         jnp.diff(flux[1,0:M+1])))
    #uStar = uStar.at[0,1:M+1].set(vA[1:M+1] + invDxDt*(flux[0,0:M] - flux[0,1:M+1]))
    #uStar = uStar.at[1,1:M+1].set(vQ[1:M+1] + invDxDt * (flux[1,0:M] - flux[1,1:M+1]))
    #uStar1 = vA[1:M+1] - invDxDt * jnp.diff(flux[0,0:M+1])
    #uStar2 = vQ[1:M+1] - invDxDt * jnp.diff(flux[1,0:M+1])
    #uStar = jnp.stack((jnp.concatenate((jnp.array([uStar1[0]]),uStar1,jnp.array([uStar1[-1]]))), 
    #                   jnp.concatenate((jnp.array([uStar2[0]]),uStar2,jnp.array([uStar2[-1]])))))


    uStar = uStar.at[0,0].set(uStar[0,1])
    uStar = uStar.at[1,0].set(uStar[1,1])
    uStar = uStar.at[0,M+1].set(uStar[0,M])
    uStar = uStar.at[1,M+1].set(uStar[1,M])

    slopesA = computeLimiterIdx(uStar, 0, invDx) * halfDx
    slopesQ = computeLimiterIdx(uStar, 1, invDx) * halfDx

    #Al = uStar[0,0:M+2] + slopesA
    #Ar = uStar[0,0:M+2] - slopesA
    #Ql = uStar[1,0:M+2] + slopesQ
    #Qr = uStar[1,0:M+2] - slopesQ
    Al = jax.vmap(lambda a, b: a+b)(uStar[0,0:M+2], slopesA)
    Ar = jax.vmap(lambda a, b: a-b)(uStar[0,0:M+2], slopesA)
    Ql = jax.vmap(lambda a, b: a+b)(uStar[1,0:M+2], slopesQ)
    Qr = jax.vmap(lambda a, b: a-b)(uStar[1,0:M+2], slopesQ)
    
    Fl = jnp.array(jax.vmap(computeFlux_par)(gamma_ghost, Al, Ql))
    Fr = jnp.array(jax.vmap(computeFlux_par)(gamma_ghost, Ar, Qr))
    #jax.debug.print("{x}", x = uStar)
    #jax.debug.print("{x}", x = Ar)
    #jax.debug.print("{x}", x = Al)
    #jax.debug.print("{x}", x = Qr)
    #jax.debug.print("{x}", x = Ql)
    #jax.debug.print("{x}", x = Fr)
    #jax.debug.print("{x}", x = Fl)
    
    #jax.debug.print("{x}", x = Fl)
    #Fl = computeFlux(gamma_ghost, Al, Ql)
    #Fr = computeFlux(gamma_ghost, Ar, Qr)

    flux = jnp.empty((2,M+2))
    flux = flux.at[0,0:M+1].set(jax.vmap(lambda a, b, c, d: 0.5*(a+b - dxDt*(c-d)))(Fr[0, 1:M+2], Fl[0, 0:M+1], Ar[1:M+2], Al[0:M+1]))
    flux = flux.at[1,0:M+1].set(jax.vmap(lambda a, b, c, d: 0.5*(a+b - dxDt*(c-d)))(Fr[1, 1:M+2], Fl[1, 0:M+1], Qr[1:M+2], Ql[0:M+1]))
    #flux = flux.at[0,0:M+1].set(0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])))
    #flux = flux.at[1,0:M+1].set(0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1])))
    #flux = jnp.stack((0.5 * (Fr[0, 1:M+2] + Fl[0, 0:M+1] - dxDt * (Ar[1:M+2] - Al[0:M+1])), 
    #                 0.5 * (Fr[1, 1:M+2] + Fl[1, 0:M+1] - dxDt * (Qr[1:M+2] - Ql[0:M+1]))))


    #A = A.at[0:M].set(0.5*(A[0:M] + uStar[0,1:M+1] + invDxDt * (flux[0, 0:M] - flux[0, 1:M+1])))
    #jax.debug.print("{x}", x = Q)
    #jax.debug.print("{x}", x = uStar)
    #jax.debug.print("{x}", x = flux)
    A = A.at[0:M].set(jax.vmap(lambda a, b, c, d: 0.5*(a+b+invDxDt*(c-d)))(A[0:M],
                                                             uStar[0,1:M+1],
                                                             flux[0,0:M],
                                                             flux[0,1:M+1]))
    Q = Q.at[0:M].set(jax.vmap(lambda a, b, c, d: 0.5*(a+b+invDxDt*(c-d)))(Q[0:M],
                                                             uStar[1,1:M+1],
                                                             flux[1,0:M],
                                                             flux[1,1:M+1]))
    #uStar = uStar.at[1,1:M+1].set(jax.vmap(lambda a, b, c: a+invDxDt*(b-c))(vQ[1:M+1],
    #                                                         flux[1,0:M],
    #                                                         flux[1,1:M+1]))
    #A = A.at[0:M].set(0.5*(A[0:M] + uStar[0,1:M+1] - invDxDt * jnp.diff(flux[0,0:M+1])))
    #Q = Q.at[0:M].set(0.5*(Q[0:M] + uStar[1,1:M+1] - invDxDt * jnp.diff(flux[1,0:M+1])))

    s_A = jax.vmap(lambda a: jnp.sqrt(a))(A)
    #Si = - ini.VCS[i].viscT * Q / A - ini.VCS[i].wallE * (s_A - ini.VCS[i].s_A0) * A
    #jax.debug.print("{x}", x = Q)
    Q = jax.vmap(lambda a, b, c, d, e: a - dt*(viscT*a/b + c*(d - e)*b))(Q, A, wallE, s_A, s_A0)
    #jax.debug.print("{x}", x = Q)
    #jax.debug.print("{x}", x = A)
    #jax.debug.print("{x}", x = wallE)
    #jax.debug.print("{x}", x = s_A)
    #jax.debug.print("{x}", x = s_A0)
    #Q = Q - dt * (viscT * Q / A + wallE * (s_A - s_A0) * A)

    P = jax.vmap(lambda a, b, c: pressureSA(a*b, c, Pext))(s_A, s_inv_A0, beta)
    #P = pressureSA(s_A * s_inv_A0, beta, Pext)
    c = jax.vmap(waveSpeedSA)(s_A, gamma)
    #c = waveSpeedSA(s_A, gamma)

    #if (v.wallVa[0] != 0.0).astype(bool):
    #mask = v.wallVa != 0.0
    #    Td = 1.0 / dt + v.wallVb
    #    Tlu = -v.wallVa    return @SArray [f1, f2, f3, f4, f5, f6]
    #    d = (1.0 / dt - v.wallVb) * v.Q[mask]
    #    d = d.at[0].set(v.wallVa[1:-1] * v.Q[mask[:-1]])
    #    d = d.at[-1].set(v.wallVa[1:-1] * v.Q[mask[1:]])
    #    d = d.at[jax.ops.index[1:-1]].set(v.wallVa[1:-1] * v.Q[mask[:-2]] + v.wallVa[:-2] * v.Q[mask[2:]])

    #    v.Q = v.Q.at[mask].set(jax.scipy.linalg.solve_banded((1, 1), jnp.array([Tlu[:-1], Td, Tlu[1:]]), d))

    u = jax.vmap(lambda a, b: a/b)(Q, A)
    #u = shard_map(lambda a, b: a/b, mesh, in_specs=PartitionSpec('i'), out_specs=PartitionSpec('i'))(Q, A)
    #jax.debug.print("{x}", x=u)
    return jnp.stack((u, Q, A, c, P))


def computeFlux(gamma_ghost, A, Q):
    #Flux = jnp.empty((2,A.size), dtype=jnp.float64)
    #Flux = Flux.at[0,:].set(Q)
    #Flux = Flux.at[1,:].set(Q * Q / A + gamma_ghost * A * jnp.sqrt(A))

    #return Flux
    return jnp.stack((Q, Q * Q / A + gamma_ghost * A * jnp.sqrt(A)))

def computeFlux_par(gamma_ghost, A, Q):
    #Flux = jnp.empty((2,A.size), dtype=jnp.float64)
    #Flux = Flux.at[0,:].set(Q)
    #Flux = Flux.at[1,:].set(Q * Q / A + gamma_ghost * A * jnp.sqrt(A))

    #return Flux
    return Q, Q * Q / A + gamma_ghost * A * jnp.sqrt(A)


def maxMod(a, b):
    return jnp.where(a > b, a, b)

def minMod(a, b):
    return jnp.where((a <= 0.0) | (b <= 0.0), 0.0, jnp.where(a < b, a, b))

def superBee(dU):
    #s1 = minMod(dU[0, :], 2 * dU[1, :])
    #s2 = minMod(2 * dU[0, :], dU[1, :])

    #return maxMod(s1, s2)
    return maxMod(minMod(dU[0, :], 2 * dU[1, :]), minMod(2 * dU[0, :], dU[1, :]))

def computeLimiter(U, invDx):
    #dU = jnp.empty((2, U.size), dtype=jnp.float64)
    #dU = dU.at[0, 1:].set((U[1:] - U[:-1]) * invDx)
    #dU = dU.at[1, 0:-1].set(dU[0, 1:])
    dU = jnp.diff(U) * invDx
    #test = [[0,(U[1:] - U[:-1]) * invDx], 
    #       [0, (U[1:-1] - U[:-2]) * invDx, 0]]
    #jax.debug.breakpoint()
    #return superBee(dU)
    return superBee(jnp.stack((jnp.concatenate((jnp.array([0.0]),dU)),
                               jnp.concatenate((dU,jnp.array([0.0]))))))
                                                                   


def computeLimiterIdx(U, idx, invDx):
    #U = U[idx, :]
    dU = jnp.diff(U[idx, :]) * invDx
    #dU = jnp.empty((2, U.size), dtype=jnp.float64)
    #dU = dU.at[0, 1:].set((U[1:] - U[:-1]) * invDx)
    #dU = dU.at[1, 0:-1].set(dU[0, 1:])
    
    #return superBee(dU)
    return superBee(jnp.stack((jnp.concatenate((jnp.array([0.0]),dU)),
                               jnp.concatenate((dU,jnp.array([0.0]))))))