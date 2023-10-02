import jax
import jax.numpy as jnp


@jax.jit
def pressure(A, A0, beta, Pext):
    return Pext + beta * (jnp.sqrt(A / A0) - 1.0)

@jax.jit
def pressureSA(s_A_over_A0, beta, Pext):
    return Pext + beta * (s_A_over_A0 - 1.0)

@jax.jit
def waveSpeed(A, gamma):
    return jax.lax.sqrt(1.5* gamma * jnp.sqrt(A))

@jax.jit
def waveSpeedSA(sA, gamma):
    return jnp.sqrt(1.5 * gamma * sA)
