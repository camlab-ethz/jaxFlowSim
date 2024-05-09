import jax.numpy as jnp
from jax import lax

def saveTempData(N, strides, P):
    P_t = jnp.zeros(5*N)
    def bodyFun(i,args):
        P_t, strides= args
        start = strides[i,0]
        end = strides[i,1]
        P_t = P_t.at[i*5].set(P[start])
        P_t = P_t.at[i*5+1].set(P[start+strides[i,2]])
        P_t = P_t.at[i*5+2].set(P[start+strides[i,3]])
        P_t = P_t.at[i*5+3].set(P[start+strides[i,4]])
        P_t = P_t.at[i*5+4].set(P[end-1])
        return (P_t, strides)

    P_t, _ = lax.fori_loop(0, N, bodyFun, (P_t,strides))
    return P_t
