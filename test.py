import jax
#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#  # Run the operations to be profiled
#  key = jax.random.PRNGKey(0)
#  x = jax.random.normal(key, (5000, 5000))
#  y = x @ x
#  y.block_until_ready()

#out = []
#
#@jax.jit
#def test_fun(x):
#    jax.debug.breakpoint()
#    y = x + 1
#    out.append(y)
#
#
#with jax.checking_leaks():
#    test_fun(1)


a = jax.numpy.arange(12).reshape((3, 4))
b = jax.numpy.arange(12).reshape((3,4))
print(jax.vmap(lambda x, y: jax.numpy.stack((x, x)))(a, b).reshape((6,4)))
print(a, b)