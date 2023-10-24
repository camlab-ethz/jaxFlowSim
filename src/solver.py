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
        Smax = jnp.max(jnp.abs(jax.lax.dynamic_slice(u,(i,0), (1,M)) + jnp.abs(jax.lax.dynamic_slice(c,(i,0), (1,M)))))
        vessel_dt = dx[i] * Ccfl / Smax
        dt = jax.lax.cond(dt > vessel_dt, lambda: vessel_dt, lambda: dt)
        return dt
    dt = jax.lax.fori_loop(0, N, body_fun, dt)
    return dt



#@jax.jit
@partial(jax.jit, static_argnums=(0, 1))
def solveModel(M, N, t, dt, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, input_data, rho):

    inlet = sim_dat_const_aux[0+4*N] 
    u0 = sim_dat[0,0]
    u1 = sim_dat[0,1]
    A0 = sim_dat[2*N,0]
    c0 = sim_dat[3*N,0]
    c1 = sim_dat[3*N,1]
    cardiac_T = sim_dat_const_aux[0+1*N]
    dx = sim_dat_const_aux[0+0*N]
    A00 = sim_dat_const[0,0]
    beta0 = sim_dat_const[N,0]
    Pext = sim_dat_const_aux[0+2*N]
    new_sim_dat = setInletBC(inlet, u0, u1, A0, 
                        c0, c1, t, dt, 
                        input_data, cardiac_T, 1/dx, A00, 
                        beta0, Pext)
    sim_dat = sim_dat.at[N,0].set(new_sim_dat[0])
    sim_dat = sim_dat.at[2*N,0].set(new_sim_dat[1])
    #_Q, _A = setInletBC(inlet, u0, u1, A0, 
    #_Q, _A = setInletBC(inlet, u0, u1, A0, 
    #                    c0, c1, t, dt, 
    #                    input_data, cardiac_T, 1/dx, A00, 
    #                    beta0, Pext)
    #sim_dat = sim_dat.at[1,0].set(_Q)
    #sim_dat = sim_dat.at[2,0].set(_A)



    def body_fun1(j, dat):
        (dt, sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux) = dat
        i = edges[j,0]-1
        new_sim_dat = muscl(M, dt, 
                  jax.lax.dynamic_slice(sim_dat, (i+N,0), (1,M)).flatten(),
                  jax.lax.dynamic_slice(sim_dat, (i+2*N,0), (1,M)).flatten(), 
                  sim_dat_aux[i+2*N], 
                  sim_dat_aux[i+3*N], 
                  sim_dat_aux[i+6*N], 
                  sim_dat_aux[i+7*N],
                  jax.lax.dynamic_slice(sim_dat_const, (i+0,0), (1,M)).flatten(), 
                  jax.lax.dynamic_slice(sim_dat_const, (i+1*N,0), (1,M)).flatten(), 
                  jax.lax.dynamic_slice(sim_dat_const, (i+2*N,0), (1,M)).flatten(), 
                  jax.lax.dynamic_slice(sim_dat_const, (i+3*N,0), (1,M)).flatten(),
                  sim_dat_const_aux[i+0*N], 
                  sim_dat_const_aux[i+2*N], 
                  sim_dat_const_aux[i+3*N])
        sim_dat = sim_dat.at[i,:].set(new_sim_dat[0,:])
        sim_dat = sim_dat.at[i + N,:].set(new_sim_dat[1,:])
        sim_dat = sim_dat.at[i + 2*N,:].set(new_sim_dat[2,:])
        sim_dat = sim_dat.at[i + 3*N,:].set(new_sim_dat[3,:])
        sim_dat = sim_dat.at[i + 4*N,:].set(new_sim_dat[4,:])


        return (dt, sim_dat, sim_dat_aux, 
                sim_dat_const, sim_dat_const_aux)
        
    #sim_dat = jax.vmap(lambda a, b, c, d, e, f, g, h, i, j, k, l, m: muscl(M, dt, 
    #                                                             a, b, c, 
    #                                                             d, e, f, 
    #                                                             g, h, i, 
    #                                                             j, k, l, m))(sim_dat[N:2*N,:], 
    #                                                                          sim_dat[2*N:3*N,:],
    #                                                                          sim_dat_aux[2*N:3*N],
    #                                                                          sim_dat_aux[3*N:4*N],
    #                                                                          sim_dat_aux[6*N:7*N],
    #                                                                          sim_dat_aux[7*N:8*N],
    #                                                                          sim_dat_const[:1*N,:], 
    #                                                                          sim_dat_const[1*N:2*N,:],
    #                                                                          sim_dat_const[2*N:3*N,:], 
    #                                                                          sim_dat_const[3*N:4*N,:],
    #                                                                          sim_dat_const_aux[:N],
    #                                                                          sim_dat_const_aux[2*N:3*N],
    #                                                                          sim_dat_const_aux[3*N:4*N]).reshape((5*N,M))

    (_, sim_dat, sim_dat_aux, _, _)  = jax.lax.fori_loop(0, N, body_fun1, 
                                                   (dt, sim_dat, sim_dat_aux, 
                                                    sim_dat_const, sim_dat_const_aux))

    def body_fun2(j, dat):
        (sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, edges, rho) = dat
        i = edges[j,0]-1
        end = (i+1)*M
        
        def setOutletBC_wrapper(sim_dat, sim_dat_aux):
            u1 = sim_dat[i+N*0,-1]
            u2 = sim_dat[i+N*0,-2]
            Q1 = sim_dat[i+N*1,-1]
            A1 = sim_dat[i+N*2,-1]
            c1 = sim_dat[i+N*3,-1]
            c2 = sim_dat[i+N*3,-2]
            P1 = sim_dat[i+N*4,-1]
            P2 = sim_dat[i+N*4,-2]
            P3 = sim_dat[i+N*4,-3]
            Pc = sim_dat_aux[i+10*N]
            W1M0 = sim_dat_aux[i+0*N]
            W2M0 = sim_dat_aux[i+1*N]
            u, Q, A, c, P1, Pc = setOutletBC(dt,
                                             u1, u2, Q1, A1, c1, c2, 
                                             P1, P2, P3, Pc, W1M0, W2M0,
                                             sim_dat_const[i+N*0,-1],
                                             sim_dat_const[i+N*1,-1],
                                             sim_dat_const[i+N*2,-1],
                                             sim_dat_const_aux[i+N*0],
                                             sim_dat_const_aux[i+N*2],
                                             sim_dat_const_aux[i+N*5], 
                                             sim_dat_const_aux[i+N*6],
                                             sim_dat_const_aux[i+N*7],
                                             sim_dat_const_aux[i+N*8],
                                             sim_dat_const_aux[i+N*9])
                                             #beta[i], gamma[i], A0[i,M-1])
            sim_dat = sim_dat.at[i+N*0,-1].set(u)
            sim_dat = sim_dat.at[i+N*1,-1].set(Q)
            sim_dat = sim_dat.at[i+N*2,-1].set(A)
            sim_dat = sim_dat.at[i+N*3,-1].set(c)
            sim_dat = sim_dat.at[i+N*4,-1].set(P1)
            sim_dat_aux = sim_dat_aux.at[i+10*N].set(Pc)
            #sim_dat_aux_out = sim_dat_aux
            #sim_dat_aux_out[i,10] = Pc
            return sim_dat, sim_dat_aux

        (sim_dat, 
         sim_dat_aux) = jax.lax.cond(sim_dat_const_aux[i+5*N] != 0,
                                    lambda x, y: setOutletBC_wrapper(x,y), 
                                    lambda x, y: (x,y), sim_dat, sim_dat_aux)



        def solveBifurcation_wrapper(sim_dat):
            d1_i = edges[j,4]
            d2_i = edges[j,5]
            u1 = sim_dat[i + N*0,-1]
            u2 = sim_dat[d1_i + N*0,0]
            u3 = sim_dat[d2_i + N*0,0]
            A1 = sim_dat[i + N*2,-1]
            A2 = sim_dat[d1_i + N*2,0]
            A3 = sim_dat[d2_i + N*2,0]
            (u1, u2, u3, 
             Q1, Q2, Q3, 
             A1, A2, A3, 
             c1, c2, c3, 
             P1, P2, P3) = solveBifurcation(u1, u2, u3, 
                                            A1, A2, A3,
                                            sim_dat_const[i + 0*N,-1],
                                            sim_dat_const[d1_i + 0*N,0],
                                            sim_dat_const[d2_i + 0*N,0],
                                            sim_dat_const[i + 1*N,-1],
                                            sim_dat_const[d1_i + 1*N,0],
                                            sim_dat_const[d2_i + 1*N,0],
                                            sim_dat_const[i + 2*N,-1],
                                            sim_dat_const[d1_i + 2*N,0],
                                            sim_dat_const[d2_i + 2*N,0],
                                            sim_dat_const_aux[i + 2*N],
                                            sim_dat_const_aux[d1_i + 2*N],
                                            sim_dat_const_aux[d2_i + 2*N],
                                            )
            sim_dat = sim_dat.at[i + 0*N,-1].set(u1) 
            sim_dat = sim_dat.at[d1_i + 0*N,0].set(u2)    
            sim_dat = sim_dat.at[d2_i + 0*N,0].set(u3)
            sim_dat = sim_dat.at[i + 1*N,-1].set(Q1)
            sim_dat = sim_dat.at[d1_i + 1*N,0].set(Q2)
            sim_dat = sim_dat.at[d2_i + 1*N,0].set(Q3)
            sim_dat = sim_dat.at[i + 2*N,-1].set(A1)
            sim_dat = sim_dat.at[d1_i + 2*N,0].set(A2)
            sim_dat = sim_dat.at[d2_i + 2*N,0].set(A3)
            sim_dat = sim_dat.at[i + 3*N,-1].set(c1)
            sim_dat = sim_dat.at[d1_i + 3*N,0].set(c2)
            sim_dat = sim_dat.at[d2_i + 3*N,0].set(c3)
            sim_dat = sim_dat.at[i + 4*N,-1].set(P1)
            sim_dat = sim_dat.at[d1_i + 4*N,0].set(P2)
            sim_dat = sim_dat.at[d2_i + 4*N,0].set(P3)

            return sim_dat

        sim_dat = jax.lax.cond((sim_dat_const_aux[i + 5*N] == 0) * (edges[j,3] == 2),
                                    lambda x: solveBifurcation_wrapper(x), 
                                    lambda x: x, sim_dat)

        #elif :
        def solveConjunction_wrapper(sim_dat, rho):
            d_i = edges[j,7]
            u1 = sim_dat[i + 0*N,-1]
            u2 = sim_dat[d_i + 0*N,0]
            A1 = sim_dat[i + 2*N,-1]
            A2 = sim_dat[d_i + 2*N,0]
            (u1, u2, Q1, Q2, 
             A1, A2, c1, c2, P1, P2) = solveConjunction(u1, u2, 
                                                        A1, A2,
                                                        sim_dat_const[i + 0*N,-1],
                                                        sim_dat_const[d_i + 0*N,0],
                                                        sim_dat_const[i + 1*N,-1],
                                                        sim_dat_const[d_i + 1*N,0],
                                                        sim_dat_const[i + 2*N,-1],
                                                        sim_dat_const[d_i + 2*N,0],
                                                        sim_dat_const_aux[i + 2*N],
                                                        sim_dat_const_aux[d_i + 2*N],
                                                        rho)
            sim_dat = sim_dat.at[i + 0*N,-1].set(u1)
            sim_dat = sim_dat.at[d_i + 0*N,0].set(u2)
            sim_dat = sim_dat.at[i + 1*N,-1].set(Q1)
            sim_dat = sim_dat.at[d_i + 1*N,0].set(Q2)
            sim_dat = sim_dat.at[i + 2*N,-1].set(A1)
            sim_dat = sim_dat.at[d_i + 2*N,0].set(A2)
            sim_dat = sim_dat.at[i + 3*N,-1].set(c1)
            sim_dat = sim_dat.at[d_i + 3*N,0].set(c2)
            sim_dat = sim_dat.at[i + 4*N,-1].set(P1)
            sim_dat = sim_dat.at[d_i + 4*N,0].set(P2)

            #jax.debug.print("{x}", x = (u1, u2, Q1, Q2, 
            #                            A1, A2, c1, c2, 
            #                            P1, P2))

            return sim_dat

        sim_dat = jax.lax.cond((sim_dat_const_aux[i+5*N] == 0) * 
                               (edges[j,3] != 2) *
                               (edges[j,6] == 1),
                                lambda x, y: solveConjunction_wrapper(x, y), 
                                lambda x, y: x, sim_dat, rho)

        #elif edges[j,6] == 2:                                           
        def solveAnastomosis_wrapper(sim_dat):
            p1_i = edges[j,7]
            p2_i = edges[j,8]
            d = edges[j,9]
            u1 = sim_dat[i + 0*N,-1]
            u2 = sim_dat[p1_i + 0*N,-1]
            u3 = sim_dat[d + 0*N,0]
            Q1 = sim_dat[i + 1*N,-1]
            Q2 = sim_dat[p1_i + 1*N,-1]
            Q3 = sim_dat[d + 1*N,0]
            A1 = sim_dat[i + 2*N,-1]
            A2 = sim_dat[p1_i + 2*N,-1]
            A3 = sim_dat[d + 2*N,0]
            c1 = sim_dat[i + 3*N,-1]
            c2 = sim_dat[p1_i + 3*N,-1]
            c3 = sim_dat[d + 3*N,0]
            P1 = sim_dat[i + 4*N,-1]
            P2 = sim_dat[p1_i + 4*N,-1]
            P3 = sim_dat[d + 4,0]
            u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3 = jax.lax.cond(
                jnp.maximum(p1_i, p2_i) == i, 
                lambda: solveAnastomosis(u1, u2, u3, 
                                         A1, A2, A3,
                                         sim_dat_const[i + 0*N,-1],
                                         sim_dat_const[p1_i + 0*N,-1],
                                         sim_dat_const[d + 0*N,0],
                                         sim_dat_const[i + 1*N,-1],
                                         sim_dat_const[p1_i + 1*N,-1],
                                         sim_dat_const[d + 1*N,0],
                                         sim_dat_const[i + 2*N,-1],
                                         sim_dat_const[p1_i + 2*N,-1],
                                         sim_dat_const[d + 2*N,0],
                                         sim_dat_const_aux[i + 2*N],
                                         sim_dat_const_aux[p1_i + 2*N],
                                         sim_dat_const_aux[d + 2*N],
                                        ), 
                lambda: (u1, u2, u3, Q1, Q2, Q3, A1, A2, A3, c1, c2, c3, P1, P2, P3))
            sim_dat = sim_dat.at[i+ 0*N,-1].set(u1)
            sim_dat = sim_dat.at[p1_i + 0*N,-1].set(u2)
            sim_dat = sim_dat.at[d + 0*N,0].set(u3)
            sim_dat = sim_dat.at[i+ 1*N,-1].set(Q1)
            sim_dat = sim_dat.at[p1_i + 1*N,-1].set(Q2)
            sim_dat = sim_dat.at[d + 1*N,0].set(Q3)
            sim_dat = sim_dat.at[i+ 2*N,-1].set(A1)
            sim_dat = sim_dat.at[p1_i + 2*N,-1].set(A2)
            sim_dat = sim_dat.at[d + 2*N,0].set(A3)
            sim_dat = sim_dat.at[i+ 3*N,-1].set(c1)
            sim_dat = sim_dat.at[p1_i + 3*N,-1].set(c2)
            sim_dat = sim_dat.at[d + 3*N,0].set(c3)
            sim_dat = sim_dat.at[i+ 4*N,-1].set(P1)
            sim_dat = sim_dat.at[p1_i + 4*N,-1].set(P2)
            sim_dat = sim_dat.at[d + 4*N,0].set(P3)

            return sim_dat
        
        sim_dat = jax.lax.cond((sim_dat_const_aux[i + 5*N] == 0) * 
                               (edges[j,3] != 2) *
                               (edges[j,6] == 2),
                                lambda x: solveAnastomosis_wrapper(x), 
                                lambda x: x, sim_dat)

        return (sim_dat, sim_dat_aux, 
                sim_dat_const, sim_dat_const_aux, 
                edges, rho)
    #for j in np.arange(0,,1):
    
    #def cond_fun(dat):
    #    _, _, j = dat
    #    return j < ini.EDGES.edges.shape[0]


    #(sim_dat, sim_dat_aux), _ = jax.lax.scan(body_fun, (sim_dat, sim_dat_aux), jnp.arange(ini.NUM_VESSELS))
    (sim_dat, sim_dat_aux, _, _, _, _)  = jax.lax.fori_loop(0, N, body_fun2, 
                                                   (sim_dat, sim_dat_aux, 
                                                    sim_dat_const, sim_dat_const_aux, 
                                                    edges, rho))

    
    return sim_dat, sim_dat_aux


#@jax.jit
#@partial(shard_map, mesh=mesh, in_specs=P('i', 'j'),
#         out_specs=P('i'))
@partial(jax.jit, static_argnums=(0,))
def muscl(M, dt, 
          Q, A, 
          U00Q, U00A, UM1Q, UM1A, 
          A0, beta,  gamma, wallE,
          dx, Pext,viscT):
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