import time
import itertools
import numpy as np
import jax.numpy as jnp
from functools import partial
import numpyro.distributions as dist  # type: ignore
import numpyro
from jax import jit
import jax
from numpyro.infer import MCMC  # type: ignore
from numpyro.infer.reparam import TransformReparam  # type: ignore
import matplotlib.pyplot as plt

import os
import sys
import jaxtyping


import arviz
import tqdm
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import uniform
from flax.linen.module import Module, compact
from flax.training import train_state
import optax  # type: ignore
from functools import partial
from typing import Callable, Optional
from jax import random


sys.path.insert(0, sys.path[0] + "/..")
from src.model import config_simulation, simulation_loop_unsafe


def param_inf_numpyro(vessel_indices, var_indices, CONFIG_FILENAME):
    """
    Function to select specific parts of the simulation data for parameter inference.

    Args:
        vessel_indices (Array): Indices of the vessels to select.
        var_indices (Array): Indices of the variables to select.

    Returns:
        Array: Selected parts of the simulation data for parameter inference.
    """

    # Set verbosity flag to control logging
    VERBOSE = True

    # Configure the simulation with the given configuration file
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
        input_data,
        rho,
        masks,
        strides,
        edges,
        vessel_names,
        cardiac_T,
    ) = config_simulation(CONFIG_FILENAME, VERBOSE)

    UPPER = 1000

    # Set up and execute the simulation loop using JIT compilation
    SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)

    def sim_loop_wrapper(params, upper=UPPER):
        """
        Wrapper function for running the simulation loop with a modified R value.

        Args:
            r (float): Scaling factor for the selected simulation constant.

        Returns:
            Array: Pressure values from the simulation with the modified R value.
        """
        rs = params * sim_dat_const_aux[vessel_indices, var_indices] * 2
        sim_dat_const_aux_new = jnp.array(sim_dat_const_aux)
        sim_dat_const_aux_new = sim_dat_const_aux_new.at[
            vessel_indices, var_indices
        ].set(rs)

        sim_dat_wrapped, t_wrapped, p_wrapped = SIM_LOOP_JIT(  # pylint: disable=E1102
            N,
            B,
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux_new,
            Ccfl,
            input_data,
            rho,
            masks,
            strides,
            edges,
            upper=upper,
        )
        return sim_dat_wrapped, t_wrapped, p_wrapped

    sim_dat_obs, t_obs, P_obs = sim_loop_wrapper([0.5] * len(vessel_indices))
    sim_dat_obs_long, t_obs_long, P_obs_long = sim_loop_wrapper(
        [0.5] * len(vessel_indices), upper=120000
    )

    class Loss(object):
        def __init__(self, axis=0, order=None):
            super(Loss, self).__init__()
            self.axis = axis
            self.order = order

        def relative_loss(self, s, s_pred):
            return jnp.log(
                jnp.mean(
                    jnp.power(
                        jnp.linalg.norm(s_pred - s, ord=None, axis=self.axis),
                        2,
                    )
                    / jnp.power(jnp.linalg.norm(s, ord=None, axis=self.axis), 2)
                )
            )

        def __call__(self, s, s_pred):
            return self.relative_loss(s, s_pred)

    loss = Loss()

    def logp(y, r, sigma):
        """
        Compute the log-probability of the observed data given the model parameters.

        Args:
            y (Array): Observed pressure data.
            r (float): Scaling factor for the selected simulation constant.
            sigma (float): Standard deviation of the noise in the observations.

        Returns:
            float: Log-probability of the observed data given the model parameters.
        """
        _, _, y_hat = sim_loop_wrapper(jax.nn.softplus(r))  # pylint: disable=E1102
        log_prob = jnp.log(
            jnp.mean(
                jnp.linalg.norm(
                    jax.scipy.stats.norm.pdf(
                        ((y - y_hat)) / y.mean(axis=0),
                        loc=0,
                        scale=sigma,
                    ),
                    axis=0,
                )
            )
        )
        return log_prob

    def model(p_obs):
        """
        Define the probabilistic model for inference using NumPyro.

        Args:
            p_obs (Array): Observed pressure data.
            sigma (float): Standard deviation of the noise in the observations.
            scale (float): Scale parameter for the prior distribution.
            r_scale (float): Initial scale value for R.
        """
        r_dist = numpyro.sample(
            "theta",
            dist.Normal(),
            sample_shape=(len(vessel_indices),),
        )
        jax.debug.print("test: {x}", x=r_dist)
        numpyro.sample(
            "obs",
            dist.Normal(
                (sim_loop_wrapper(r_dist)[2] - p_obs) / jnp.mean(p_obs),
                1,
            ),
            obs=jnp.zeros(P_obs.shape),
        )

    # Define the hyperparameters for the network properties
    network_properties = {
        "sigma": [1e-2],
        "scale": [10],
        "num_warmup": [1000],
        "num_samples": [1000],
        "num_chains": [1],
    }

    # Create a list of all possible combinations of the network properties
    settings = list(itertools.product(*network_properties.values()))  # type: ignore

    def geweke_diagnostic(samples, first_frac=0.1, last_frac=0.5):
        """
        Compute the Geweke diagnostic z-scores for a single chain.

        Parameters:
        - samples: np.ndarray
            1D array of MCMC samples for a parameter.
        - first_frac: float
            Fraction of the chain to use for the early segment.
        - last_frac: float
            Fraction of the chain to use for the late segment.

        Returns:
        - z_scores: np.ndarray
            Geweke z-scores for different points in the chain.
        - indices: np.ndarray
            Indices of the chain where z-scores are computed.
        """
        n_samples = len(samples)
        first_samples = samples[: int(first_frac * n_samples)]
        last_samples = samples[-int(last_frac * n_samples) :]

        mean_first = np.mean(first_samples, axis=0)
        mean_last = np.mean(last_samples, axis=0)

        var_first = np.var(first_samples, ddof=1, axis=0)
        var_last = np.var(last_samples, ddof=1, axis=0)

        z_score = (mean_first - mean_last) / np.sqrt(
            var_first / len(first_samples) + var_last / len(last_samples)
        )
        return z_score

    def generate_geweke_plot(
        mcmc_samples, var_name, setup_properties, first_frac=0.1, last_frac=0.5, step=50
    ):
        """
        Generate a Geweke plot for diagnostics of convergence.

        Parameters:
        - mcmc_samples: dict
            Dictionary of MCMC samples (obtained via `mcmc.get_samples()`).
        - var_name: str
            The variable name to analyze.
        - first_frac: float
            Fraction of the chain at the beginning to use for comparison.
        - last_frac: float
            Fraction of the chain at the end to use for comparison.
        - step: int
            Step size for computing z-scores along the chain.
        """
        # Extract the samples for the specified variable
        samples = mcmc_samples[var_name]
        samples = samples[int(0.5 * len(samples)) :]
        # if samples.ndim > 1:  # Multi-chain samples
        #    samples = samples.reshape(-1)

        # Compute Geweke z-scores along the chain
        z_scores = []
        indices = []
        for i in range(step, len(samples), step):
            partial_samples = samples[:i]
            z_score = geweke_diagnostic(partial_samples, first_frac, last_frac)
            z_scores.append(z_score)
            indices.append(i)

        # Create a Geweke plot
        plt.figure()
        plt.plot(
            indices,
            z_scores,
            marker="o",
            linestyle="--",
            label=["R1_1", "R2_1", "R1_2", "R2_2"],
        )
        plt.axhline(-2, color="red", linestyle="--")
        plt.axhline(2, color="red", linestyle="--")
        plt.axhline(0, color="black", linestyle="-")
        plt.xlabel("iteration")
        plt.ylabel("z-score")
        plt.legend()
        # plt.grid(alpha=0.3)
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke_nt.pdf"
        )
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke_nt.png"
        )
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke_nt.eps"
        )
        plt.title(f"geweke diagnostic for {var_name}")
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke.pdf"
        )
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke.png"
        )
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_geweke.eps"
        )
        plt.close()

    # Define the folder to save the inference results
    RESULTS_FOLDER = "results/inference_numpyro_4"
    if not os.path.isdir(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER, mode=0o777)

    # Loop through each combination of settings and run MCMC inference
    for set_num, setup in enumerate(settings):
        print(
            "###################################",
            set_num,
            "###################################",
        )
        setup_properties = {
            "sigma": setup[0],
            "scale": setup[1],
            "num_warmup": setup[2],
            "num_samples": setup[3],
            "num_chains": setup[4],
        }
        mcmc = MCMC(
            numpyro.infer.NUTS(
                model, forward_mode_differentiation=True, dense_mass=True
            ),
            num_samples=setup_properties["num_samples"],
            num_warmup=setup_properties["num_warmup"],
            num_chains=setup_properties["num_chains"],
        )
        starting_time = time.time_ns()
        mcmc.run(
            jax.random.PRNGKey(3450),
            P_obs,
        )
        total_time = (time.time_ns() - starting_time) / 1.0e9
        mcmc.print_summary()
        R = jnp.mean(
            mcmc.get_samples()["theta"][int(0.5 * len(mcmc.get_samples())) :], axis=0
        ).flatten()

        print(R)
        print(mcmc.get_samples()["theta"].shape)
        _, t, y = sim_loop_wrapper(jax.nn.softplus(R))  # pylint: disable=E1102
        loss_val = loss(P_obs, y)
        _, t, y = sim_loop_wrapper(
            jax.nn.softplus(R), upper=120000
        )  # pylint: disable=E1102
        indices_sorted = np.argsort(t_obs_long[-12000:])
        plt.scatter(
            t_obs_long[-12000:][indices_sorted],
            P_obs_long[-12000:, -8][indices_sorted] / 133.322,
            label="ground truth",
            s=0.1,
        )
        indices_sorted = np.argsort(t[-12000:])
        plt.scatter(
            t[-12000:][indices_sorted],
            y[-12000:, -8][indices_sorted] / 133.322,
            label="predicted",
            s=0.1,
        )
        lgnd = plt.legend(loc="upper right")
        lgnd.legend_handles[0]._sizes = [30]
        lgnd.legend_handles[1]._sizes = [30]
        plt.xlabel("t/T")
        plt.ylabel("P [mmHg]")
        # plt.xlim([0.0, 1.0])
        # plt.ylim([30, 140])
        plt.tight_layout()
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_nt.pdf"
        )
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_nt.png"
        )
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}_nt.eps"
        )
        plt.title(
            f"learning scaled Windkessel resistance parameters of a bifurcation:\n[R1_1, R2_1, R1_2, R2_2] = {R},\nloss = {loss_val}, \nwallclock time = {total_time}"
        )
        plt.tight_layout()
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}.pdf"
        )
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}.png"
        )
        plt.savefig(
            f"{RESULTS_FOLDER}/{str(setup_properties["num_warmup"])}_{str(setup_properties["num_samples"])}.eps"
        )
        plt.close()

        # arviz_data = arviz.from_numpyro(mcmc)
        # arviz.geweke(mcmc.get_samples()["theta"])

        # arviz_data

        # geweke = pymc.diagnostics.geweke(mcmc.get_samples()["theta"])
        # arviz.geweke_plot(geweke)
        generate_geweke_plot(mcmc.get_samples(), "theta", setup_properties, step=5)


def param_inf_optax(vessel_indices, var_indices, CONFIG_FILENAME):
    # Set verbosity flag to control logging
    VERBOSE = True
    # Configure the simulation with the given configuration file
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
        CCFL,
        input_data,
        rho,
        masks,
        strides,
        edges,
        vessel_names,
        cardiac_T,
    ) = config_simulation(CONFIG_FILENAME, VERBOSE)

    UPPER = 1000

    # Set up and execute the simulation loop using JIT compilation
    SIM_LOOP_JIT = partial(jit, static_argnums=(0, 1, 12))(simulation_loop_unsafe)

    num_vessels = len(vessel_indices)

    @compact
    def sim_loop_wrapper(params, upper=UPPER):
        """
        Wrapper function for running the simulation loop with a modified parameter value.

        Args:
            params (Array): Array containing scaling factor for the selected simulation constant.

        Returns:
            Array: Pressure values from the simulation with the modified parameter.
        """
        rs = params * sim_dat_const_aux[vessel_indices, var_indices] * 2
        sim_dat_const_aux_new = jnp.array(sim_dat_const_aux)
        sim_dat_const_aux_new = sim_dat_const_aux_new.at[
            vessel_indices, var_indices
        ].set(rs)
        sim_dat_wrapped, t_wrapped, p_wrapped = SIM_LOOP_JIT(  # pylint: disable=E1102
            N,
            B,
            sim_dat,
            sim_dat_aux,
            sim_dat_const,
            sim_dat_const_aux_new,
            CCFL,
            input_data,
            rho,
            masks,
            strides,
            edges,
            upper=upper,
        )
        return sim_dat_wrapped, t_wrapped, p_wrapped

    sim_dat_obs, t_obs, P_obs = sim_loop_wrapper([0.5] * num_vessels)
    sim_dat_obs_long, t_obs_long, P_obs_long = sim_loop_wrapper(
        [0.5] * num_vessels, 120000
    )

    # Define the folder to save the optimization results
    RESULTS_FOLDER = "results/inference_optax_4"
    if not os.path.isdir(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER, mode=0o777)

    class SimDense(Module):
        features = num_vessels
        kernel_init: Callable[[jax.random.PRNGKey, tuple, jnp.dtype], jnp.ndarray] = (
            uniform(1.0)
        )

        @compact
        def __call__(self) -> jnp.ndarray:
            Rs = self.param(
                "Rs",
                self.kernel_init,
                # lambda rng, shape: 0.5*jnp.ones(shape),
                (num_vessels,),
            )

            _, _, y = sim_loop_wrapper(jax.nn.softplus(Rs))
            return y

    class Loss(object):
        def __init__(self, axis=0, order=None):
            super(Loss, self).__init__()
            self.axis = axis
            self.order = order

        def relative_loss(self, s, s_pred):
            return jnp.log(
                jnp.mean(
                    jnp.power(
                        jnp.linalg.norm(
                            s_pred[:, [-1, -6]] - s[:, [-1, -6]],
                            ord=None,
                            axis=self.axis,
                        ),
                        2,
                    )
                    / jnp.power(
                        jnp.linalg.norm(s[:, [-1, -6]], ord=None, axis=self.axis), 2
                    )
                )
            )

        def __call__(self, s, s_pred):
            return self.relative_loss(s, s_pred)

    loss = Loss()

    def calculate_loss_train(state, params, batch):
        s = batch
        s_pred = state.apply_fn(params)
        loss_value = loss(s, s_pred)
        return loss_value

    @jax.jit
    def train_step(state, batch):
        grad_fn = jax.value_and_grad(calculate_loss_train, argnums=1)
        loss_value, grads = grad_fn(state, state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss_value

    def train_model(state, batch, num_epochs=None):
        bar = tqdm.tqdm(np.arange(num_epochs))
        params = jax.nn.softplus(state.params["params"]["Rs"])
        plt.figure()
        _, t, p = sim_loop_wrapper(params)
        loss_val = loss(p, P_obs)
        _, t, p = sim_loop_wrapper(params, upper=120000)
        sorted_indices = np.argsort(t_obs_long[-12000:])
        plt.plot(
            t_obs_long[-12000:][sorted_indices],
            P_obs_long[-12000:, -8][sorted_indices] / 133.322,
            label="ground truth",
            linewidth=0.5,
        )
        sorted_indices = np.argsort(t[-12000:])
        plt.plot(
            t[-12000:][sorted_indices],
            p[-12000:, -8][sorted_indices] / 133.322,
            label="learned",
            linewidth=0.5,
        )
        lgnd = plt.legend(loc="upper right")
        lgnd.legend_handles[0]._sizes = [30]
        lgnd.legend_handles[1]._sizes = [30]
        plt.xlabel("t/T")
        plt.ylabel("P [mmHg]")
        # plt.xlim([0.0, 1.0])
        # plt.ylim([30, 140])
        plt.tight_layout()
        plt.savefig(f"{RESULTS_FOLDER}/{str(0)}_nt.pdf")
        plt.savefig(f"{RESULTS_FOLDER}/{str(0)}_nt.png")
        plt.savefig(f"{RESULTS_FOLDER}/{str(0)}_nt.jpeg")
        plt.savefig(f"{RESULTS_FOLDER}/{str(0)}_nt.eps")
        plt.title(
            f"learning scaled Windkessel resistance parameters of a bifurcation:\n[R1_1, R2_1, R1_2, R2_2] =\n{params},\nloss = {loss_val}, \n wall-clock time = {0.0}"
        )
        plt.savefig(f"{RESULTS_FOLDER}/{str(0)}.pdf")
        plt.savefig(f"{RESULTS_FOLDER}/{str(0)}.png")
        plt.savefig(f"{RESULTS_FOLDER}/{str(0)}.jpeg")
        plt.savefig(f"{RESULTS_FOLDER}/{str(0)}.eps")
        plt.close()
        total_time = 0
        for epoch in bar:
            starting_time = time.time_ns()
            state, loss_val = train_step(state, batch)
            total_time += time.time_ns() - starting_time
            params = jax.nn.softplus(state.params["params"]["Rs"])
            bar.set_description(f"Loss: {loss_val}, Parameters {params}")
            # if loss < 1e-6:
            #    break
            plt.figure()
            _, t, p = sim_loop_wrapper(params, 120000)
            sorted_indices = np.argsort(t_obs_long[-12000:])
            plt.plot(
                t_obs_long[-12000:][sorted_indices],
                P_obs_long[-12000:, -8][sorted_indices] / 133.322,
                label="ground truth",
                linewidth=0.5,
            )
            sorted_indices = np.argsort(t[-12000:])
            plt.plot(
                t[-12000:][sorted_indices],
                p[-12000:, -8][sorted_indices] / 133.322,
                label="learned",
                linewidth=0.5,
            )
            lgnd = plt.legend(loc="upper right")
            lgnd.legend_handles[0]._sizes = [30]
            lgnd.legend_handles[1]._sizes = [30]
            plt.xlabel("t/T")
            plt.ylabel("P [mmHg]")
            # plt.xlim([0.0, 1.0])
            # plt.ylim([30, 140])
            plt.tight_layout()
            plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}_nt.pdf")
            plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}_nt.png")
            plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}_nt.jpeg")
            plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}_nt.eps")
            plt.title(
                f"learning scaled Windkessel resistance parameters of a bifurcation:\n[R1_1, R2_1, R1_2, R2_2] =\n{params},\nloss = {loss_val}, \n wall-clock time = {total_time / 1e9}, \n wall-clock time = {total_time / 1e9}"
            )
            plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}.pdf")
            plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}.png")
            plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}.jpeg")
            plt.savefig(f"{RESULTS_FOLDER}/{str(epoch + 1)}.eps")
            plt.close()
        return state

    print("Model Initialized")
    lr = 1e-2
    transition_steps = 10
    decay_rate = 0.9
    weight_decay = 0
    seed = 0
    epochs = 5000

    model = SimDense()

    params = model.init(random.key(21234))

    print("Initial Parameters: ", jax.nn.softplus(params["params"]["Rs"]))

    exponential_decay_scheduler = optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        transition_begin=0,
        staircase=False,
    )

    # optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    optimizer = optax.adafactor(lr)

    model_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    trained_model_state = train_model(model_state, P_obs, num_epochs=epochs)

    y = model_state.apply_fn(trained_model_state.params)[0]

    print(
        f"Final Loss: {loss(P_obs, y)} and Parameters: {jax.nn.softplus(trained_model_state.params["params"]["Rs"])}"
    )

    plt.figure()
    plt.plot(t_obs, P_obs, "b-", label="Baseline")
    plt.plot(t_obs, y, "r--", label="Predicted")
    plt.legend()
    plt.show()
