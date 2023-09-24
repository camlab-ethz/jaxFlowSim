import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.experimental import host_callback
from functools import partial
import src.initialise as ini


@jit
def saveTempDatas(P,nodes):
    P_t_temp = jnp.zeros(5*ini.NUM_VESSELS)
    for i in range(ini.NUM_VESSELS):
        start = i*ini.MESH_SIZE
        end = (i+1)*ini.MESH_SIZE
        P_t_temp = P_t_temp.at[i*5:(i+1)*5].set(saveTempData(P[start:end], nodes[:,i]))
    
    return P_t_temp

@jit
def saveTempData(P, nodes):
    return [P[0], P[nodes[0]], P[nodes[1]], P[nodes[2]], P[-1]]
    #v.A_t = v.A_t.at[counter, :].set([t, v.A[0], v.A[ini.VCS[i].node2], v.A[ini.VCS[i].node3], v.A[ini.VCS[i].node4], v.A[-1]])
    #v.Q_t = v.Q_t.at[counter, :].set([t, v.Q[0], v.Q[ini.VCS[i].node2], v.Q[ini.VCS[i].node3], v.Q[ini.VCS[i].node4], v.Q[-1]])
    #v.u_t = v.u_t.at[counter, :].set([t, v.u[0], v.u[ini.VCS[i].node2], v.u[ini.VCS[i].node3], v.u[ini.VCS[i].node4], v.u[-1]])
    #v.c_t = v.c_t.at[counter, :].set([t, v.c[0], v.c[ini.VCS[i].node2], v.c[ini.VCS[i].node3], v.c[ini.VCS[i].node4], v.c[-1]])


@jit
def transferTempToLasts(vessels):
    for i in range(0,len(vessels)):
        vessels[i]  = transferTempToLast(vessels[i])

    return vessels

@jit
def transferTempToLast(v):
    #v.A_l = v.A_t
    v.P_l = v.P_t
    #v.Q_l = v.Q_t
    #v.u_l = v.u_t
    #v.c_l = v.c_t

    return v

@jit
def transferLastsToOuts(vessels):
    for i in range(len(vessels)):
        transferLastToOut(vessels[i], i)

@jit
def transferLastToOut(v, i):
    lastP = v.P_l
    lastQ = v.Q_l
    lastA = v.A_l
    lastc = v.c_l
    lastu = v.u_l
    lasts = [lastP, lastQ, lastA, lastc, lastu]

    outP = ini.VCS[i].out_P_name
    outQ = ini.VCS[i].out_Q_name
    outA = ini.VCS[i].out_A_name
    outc = ini.VCS[i].out_c_name
    outu = ini.VCS[i].out_u_name
    outs = [outP, outQ, outA, outc, outu]

    def writeOut(a,b):
        with open(a, "ab") as f: np.savetxt(f, b)

    for a, b in zip(outs, lasts): 
        host_callback.call(lambda x: writeOut(a,x), b)

@jit
def writeResults(vessels):
    for i in range(len(vessels)):
        writeResult(vessels[i], i)

def writeResult(v, i):
    lastP = v.P_l
    #lastQ = v.Q_l
    #lastA = v.A_l
    #lastc = v.c_l
    #lastu = v.u_l
    lasts = [lastP]#, lastQ, lastA, lastc, lastu]

    resP = ini.VCS[i].last_P_name
    #resQ = ini.VCS[i].last_Q_name
    #resA = ini.VCS[i].last_A_name
    #resc = ini.VCS[i].last_c_name
    #resu = ini.VCS[i].last_u_name
    ress = [resP]#, resQ, resA, resc, resu]

    for a, b in zip(ress, lasts):
        host_callback.call(lambda x: np.savetxt(a, x), b)

@partial(jit, static_argnums=(0,1))
def writeConv(filename, out_str):
    f_name = filename
    with open(f_name, "w") as conv_file:
        conv_file.write(out_str)
