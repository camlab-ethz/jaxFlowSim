import jax
import jax.numpy as jnp
import src.initialise as ini


@jax.jit
def calcNorms(vessels):
    norms = jnp.zeros(len(vessels))
    for i, v in enumerate(vessels):
        err = v.P_l[:, 3] - v.P_t[:, 3]
        norms = norms.at[i].set(jnp.sqrt(jnp.sum(err**2)))
    return norms

@jax.jit
def computeConvError(vessels):
    current_norms = calcNorms(vessels)
    maxnorm = jnp.max(current_norms)
    return maxnorm

@jax.jit
def printConvError(err):
    err /= 133.332
    jax.debug.print(" - Error norm = {x} mmHg", x=err)

@jax.jit
def checkConvergence(err):
    return err / 133.332 <= ini.CONV_TOLL
