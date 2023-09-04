import jax
import jax.numpy as jnp
import src.initialise as ini


@jax.jit
def calcNorms(P_t, P_l):
    norms = jnp.zeros(ini.NUM_VESSELS,dtype=jnp.float64)
    for i in range(ini.NUM_VESSELS):
        err = P_l[:, i*5 + 2] - P_t[:, i*5 + 2]
        norms = norms.at[i].set(jnp.sqrt(jnp.sum(err**2)))
    return norms

@jax.jit
def computeConvError(P_t, P_l):
    current_norms = calcNorms(P_t, P_l)
    maxnorm = jnp.max(current_norms)
    return maxnorm

@jax.jit
def printConvError(err):
    err /= 133.332
    jax.debug.print(" - Error norm = {x} mmHg", x=err)

@jax.jit
def checkConvergence(err):
    return err / 133.332 <= ini.CONV_TOLL
