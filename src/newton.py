import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
import jax

#@jit
@partial(jit, static_argnums=(0,1,2,))
def newtonRaphson(indices, fun_w, fun_f, J, U, k):

    nr_toll_U = 1e-5
    nr_toll_F = 1e-5

    def cond_fun(U):
        #print(jnp.abs(val[0] - val[1]) <= 1e-5)
        #ret = jax.lax.cond( jnp.abs(val[0]-val[1]) < 1e-5, lambda: False, lambda: True)
        W = fun_w(U[0], k)
        F = fun_f(indices, U[0], k, W)
        ret = jax.lax.cond(jnp.where((jnp.abs(U[1]) <= nr_toll_U) | (jnp.abs(F) <= nr_toll_F), 
                                      jnp.ones(F.size),
                                      jnp.zeros(F.size)).sum().astype(int)==F.size,
                            lambda: False,
                            lambda: True)
        return ret

    def body_fun(U):
        W = fun_w(U[0], k)
        F = fun_f(indices, U[0], k, W)
        dU = jnp.linalg.solve(J, -F)
        return jnp.array((U[0]+dU, dU), dtype=jnp.float64)

    return jax.lax.while_loop(cond_fun, body_fun, jnp.array((U,jnp.ones(U.size))))
    #while True:
    #    dU = jnp.linalg.solve(J, -F)
    #    U_new = U + dU

    #    #if jnp.isnan(jnp.dot(F, F)):
    #    #    e = "("
    #    #    for vessel in vessels:
    #    #        e += vessel.label + ", "
    #    #    e = e[:-2] + ")"
    #    #    raise ValueError(f"\nNewton-Raphson doesn't converge at {e} junction!")

    #    #idx = jnp.where((t >= idt) & (t <= idt1), jnp.arange(0,idt.size,1),jnp.zeros(idt.size)).sum().astype(int) #[0][0]
    #    u_ok = jnp.where((jnp.abs(dU) <= nr_toll_U) | (jnp.abs(F) <= nr_toll_F), jnp.ones(F.size),jnp.zeros(F.size)).sum().astype(int)
    #    #for i in range(len(dU)):
    #    #    u_ok = jax.lax.cond(jnp.abs(dU[i]) <= nr_toll_U or jnp.abs(F[i]) <= nr_toll_F, lambda: u_ok+1, lambda: u_ok)
    #    #    f_ok = jax.lax.cond(jnp.abs(dU[i]) <= nr_toll_U or jnp.abs(F[i]) <= nr_toll_F, lambda: f_ok+1, lambda: f_ok)
    #        #if abs(dU[i]) <= nr_toll_U or abs(F[i]) <= nr_toll_F:
    #        #    u_ok += 1
    #        #    f_ok += 1

    #    if u_ok == len(dU):# or f_ok == len(dU):
    #        return U_new
    #    else:
    #        U = U_new
    #        W = fun_w(U, k)
    #        F = fun_f(vessels, U, k, W)