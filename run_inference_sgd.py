from src.model import configSimulation, simulationLoopUnsafe_nn, simulationLoopUnsafe
import jax
import sys
import time
import os
from functools import partial
from jax import block_until_ready, jit, random, jacfwd
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import optax
import itertools
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))
jax.config.update("jax_enable_x64", True)

config_filename = "test/bifurcation/bifurcation.yml"

verbose = True
(
    N,
    B,
    J,
    sim_dat,
    sim_dat_aux,
    sim_dat_const,
    sim_dat_const_aux,
    timepoints,
    conv_tol,
    Ccfl,
    edges,
    input_data,
    rho,
    masks,
    strides,
    edges,
    vessel_names,
    cardiac_T,
) = configSimulation(config_filename, verbose)

Ccfl = 0.5


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


layer_sizes = [3, 3, 3, 1]
nn_params = init_network_params(layer_sizes, random.key(0))
print(nn_params)


def relu(x):
    return np.maximum(0, x)


def predict(params, s_A_over_A0, beta, Pext):
    # per-example predictions
    activations = np.array((s_A_over_A0, beta, Pext))
    for w, b in params:
        outputs = np.dot(w, activations) + b
        activations = relu(outputs)
    return activations


if verbose:
    starting_time = time.time_ns()

sim_loop_old_jit = partial(jit, static_argnums=(0, 1, 12))(simulationLoopUnsafe)
sim_dat, t_t, P_t = block_until_ready(
    sim_loop_old_jit(
        N,
        B,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        Ccfl,
        input_data,
        rho,
        masks,
        strides,
        edges,
        upper=120000,
    )
)

if verbose:
    ending_time = (time.time_ns() - starting_time) / 1.0e9
    print(f"elapsed time = {ending_time} seconds")


def simLoopWrapper(nn_params):
    _, _, P = simulationLoopUnsafe_nn(
        N,
        B,
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        Ccfl,
        input_data,
        rho,
        masks,
        strides,
        edges,
        nn_params,
        upper=120000,
    )
    return P


results_folder = "results/inference_ensemble_det"
if not os.path.isdir(results_folder):
    os.makedirs(results_folder, mode=0o777)

learning_rates = [1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5]

network_properties = {
    "tx": [
        optax.adabelief,
        optax.adadelta,
        optax.adagrad,
        optax.adam,
        optax.adamax,
        optax.adamaxw,
        optax.adamw,
        optax.amsgrad,
        optax.lars,
        optax.sgd,
    ],
    "learning_rate": learning_rates,
    "epochs": [100, 1000, 2000],
}

settings = list(itertools.product(*network_properties.values()))

results_folder = "results/inference_ensemble_nn"
if not os.path.isdir(results_folder):
    os.makedirs(results_folder, mode=0o777)

for set_num, setup in enumerate(settings):
    print(
        "###################################",
        set_num,
        "###################################",
    )
    results_file = (
        results_folder
        + "/setup_"
        + str(setup[0].__name__)
        + "_"
        + str(setup[1])
        + "_"
        + str(setup[2])
        + ".txt"
    )

    model = simLoopWrapper
    if len(sys.argv) > 1:
        variables = init_network_params(layer_sizes, random.key(int(sys.argv[2])))
    else:
        variables = init_network_params(layer_sizes, random.key(0))
    tx = setup[0]
    y = P_t
    x = simLoopWrapper

    state = TrainState.create(apply_fn=model, params=variables, tx=tx(setup[1]))

    def loss_fn(params, x, y):
        predictions = state.apply_fn(params)
        loss = optax.l2_loss(predictions=predictions, targets=y).mean()
        return loss

    for _ in range(100):
        print(loss_fn(state.params, x, y))
        # plt.plot(y)
        # plt.plot(state.apply_fn(state.params))
        # plt.show()
        # plt.close()
        grads = jax.jacfwd(loss_fn)(state.params, x, y)
        state = state.apply_gradients(grads=grads)
    file = open(results_file, "a")
    file.write(str(loss_fn(state.params, x, y)) + "\n")
    file.close()
