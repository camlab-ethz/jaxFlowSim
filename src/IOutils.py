import jax.numpy as jnp
from jax import lax

def saveTempData(N, starts, ends, 
                  nodes, P):
    P_t = jnp.zeros(5*N)
    def bodyFun(i,args):
        P_t, starts, ends = args
        start = starts[i]
        end = ends[i]
        P_t = P_t.at[i*5].set(P[start])
        P_t = P_t.at[i*5+1].set(P[start+nodes[i,0]])
        P_t = P_t.at[i*5+2].set(P[start+nodes[i,1]])
        P_t = P_t.at[i*5+3].set(P[start+nodes[i,2]])
        P_t = P_t.at[i*5+4].set(P[end-1])
        return (P_t,starts,ends)

    P_t, _, _ = lax.fori_loop(0, N, bodyFun, (P_t,starts,ends))
    return P_t
