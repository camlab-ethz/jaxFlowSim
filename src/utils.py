import jax.numpy as jnp
from jax import lax, debug
from src.components import SimpleClassifierCompact


def pressure(A, A0, beta, Pext):
    return pressureSA(jnp.sqrt(A / A0), beta, Pext)

def pressureSA(s_A_over_A0, beta, Pext):
    return Pext + beta * (s_A_over_A0 - 1.0)

def waveSpeed(A, gamma):
    return jnp.sqrt(1.5* gamma * jnp.sqrt(A))

def waveSpeedSA(sA, gamma):
    return jnp.sqrt(1.5 * gamma * sA)

def pressure_nn(A, A0, beta, Pext, nn_params):
    return pressureSA_nn(jnp.sqrt(A / A0), beta, Pext, nn_params)

def pressureSA_nn(s_A_over_A0, beta, Pext, nn_params):
    return predict(nn_params, s_A_over_A0, beta, Pext)
    #simpleClassifier = SimpleClassifierCompact(*nn_params)
    #return simpleClassifier(s_A_over_A0, beta, Pext)

def relu(x):
    return jnp.maximum(0, x)

def predict(params, s_A_over_A0, beta, Pext):
    # per-example predictions
    activations = jnp.array((s_A_over_A0, beta, Pext))
    for w, b in params:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    return activations[0]

