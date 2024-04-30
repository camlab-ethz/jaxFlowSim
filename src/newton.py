import jax.numpy as jnp

def newtonRaphson(fun_f, J, 
                  U, A0s,
                  betas):

    F = fun_f(U,
              A0s,
              betas)

    dU = jnp.linalg.solve(J, -F)
    return U+dU
