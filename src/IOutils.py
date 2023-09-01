import numpy as np
from jax import jit
from jax.experimental import host_callback
from functools import partial
import src.initialise as ini


@jit
def saveTempDatas(t, vessels, counter):
    for i in range(0,len(vessels)):
        vessels[i] = saveTempData(t, vessels[i], counter)
    
    return vessels

@jit
def saveTempData(t, v, counter):
    v.P_t = v.P_t.at[counter, :].set([t, v.P[0], v.P[v.node2], v.P[v.node3], v.P[v.node4], v.P[-1]])
    v.A_t = v.A_t.at[counter, :].set([t, v.A[0], v.A[v.node2], v.A[v.node3], v.A[v.node4], v.A[-1]])
    v.Q_t = v.Q_t.at[counter, :].set([t, v.Q[0], v.Q[v.node2], v.Q[v.node3], v.Q[v.node4], v.Q[-1]])
    v.u_t = v.u_t.at[counter, :].set([t, v.u[0], v.u[v.node2], v.u[v.node3], v.u[v.node4], v.u[-1]])
    v.c_t = v.c_t.at[counter, :].set([t, v.c[0], v.c[v.node2], v.c[v.node3], v.c[v.node4], v.c[-1]])

    return v

@jit
def transferTempToLasts(vessels):
    for i in range(0,len(vessels)):
        vessels[i]  = transferTempToLast(vessels[i])

    return vessels

@jit
def transferTempToLast(v):
    v.A_l = v.A_t
    v.P_l = v.P_t
    v.Q_l = v.Q_t
    v.u_l = v.u_t
    v.c_l = v.c_t

    return v

@jit
def transferLastsToOuts(vessels):
    for i in range(len(vessels)):
        transferLastToOut(vessels[i])

@jit
def transferLastToOut(v, i):
    lastP = v.P_l
    lastQ = v.Q_l
    lastA = v.A_l
    lastc = v.c_l
    lastu = v.u_l
    lasts = [lastP, lastQ, lastA, lastc, lastu]

    outP = v.out_P_name
    outQ = v.out_Q_name
    outA = v.out_A_name
    outc = v.out_c_name
    outu = v.out_u_name
    outs = [outP, outQ, outA, outc, outu]

    def writeOut(a,b):
        with open(a, "ab") as f: np.savetxt(f, b)

    for a, b in zip(outs, lasts): 
        host_callback.call(lambda x: writeOut(a,x), b)

@jit
def writeResults(vessels):
    for i in range(len(vessels)):
        writeResult(vessels[i])

def writeResult(v):
    lastP = v.P_l
    lastQ = v.Q_l
    lastA = v.A_l
    lastc = v.c_l
    lastu = v.u_l
    lasts = [lastP, lastQ, lastA, lastc, lastu]

    resP = v.last_P_name
    resQ = v.last_Q_name
    resA = v.last_A_name
    resc = v.last_c_name
    resu = v.last_u_name
    ress = [resP, resQ, resA, resc, resu]

    for a, b in zip(ress, lasts):
        host_callback.call(lambda x: np.savetxt(a, x), b)

@partial(jit, static_argnums=(0,1))
def writeConv(filename, out_str):
    f_name = filename
    with open(f_name, "w") as conv_file:
        conv_file.write(out_str)
