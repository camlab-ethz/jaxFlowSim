import jax.numpy as jnp
from jax import lax

def newtonRaphson(fun_w, fun_f, J, 
                  U, k, A0s,
                  betas):

    nr_toll_U = 1e-5
    nr_toll_F = 1e-5

    W = fun_w(U, k)
    F = fun_f(U, k, W,
              A0s,
              betas)
            
    dU = jnp.linalg.solve(J, -F)
    n = F.size
    #ones = jnp.ones(n,dtype=jnp.int32)
    #zeros = jnp.zeros(n,dtype=jnp.int32)

    #def cond_fun(args):
    #    _, dU, F, _  = args
    #    ret = lax.cond(jnp.where((jnp.abs(dU) <= nr_toll_U) | (jnp.abs(F) <= nr_toll_F), 
    #                                  ones,
    #                                  zeros).sum()==n,
    #                        lambda: False,
    #                        lambda: True)
    #    return ret

    #def body_fun(args):
    #    U, _, _, counter = args
    #    W = fun_w(U, k)
    #    F = fun_f(U, k, W,
    #              A0s,
    #              betas)
    #    dU = jnp.linalg.solve(J, -F)
    #    counter += 1
    #    return U+dU, dU, F, counter

    #counter = 0
    #return lax.while_loop(cond_fun, body_fun, (U+dU, dU, F, counter))
    return U+dU