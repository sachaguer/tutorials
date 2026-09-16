# %% [markdown]
# ---
# title: "Linear regression"
# description: "A notebook to illustrate the linear regression method and some of the associated difficulties."
# author: "Sacha Guerrini"
# date: today
# format: html
# jupyter: python3
# number-figures: true
# execute:
#   echo: false
# ---

# %%
# Import standard python libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# Set the style for plots
plt.style.use(
    "../matplotlib_config/paper.mplstyle"
)

sns.color_palette('Dark2')

# Import Jax dependencies for accelerated computation and automatic differentiation
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax

# %%
# Check the device used by Jax (CPU or GPU)
print("Jax is using device:", jax.devices()[0].device_kind)

# %% [markdown]

"""
Code source available [here](https://github.com/sachaguer/tutorials/tree/main/linear_regression/linear_fit.py).

## Linear regression

The goal of this notebook is to illustrate the linear regression method and, in particular, highlight some difficulties that can arise when modelling the stochastic process generating the data.

In this task, we want to constrain the relationship between two variables $x$ and $y$ to be linear, such that, we can write the model as:
$$
y = a x + b + \epsilon
$$

where $a$ and $b$ are the parameters of the model, and $\epsilon$ is a random variable representing the noise in the data. The goal of linear regression is to estimate the parameters $a$ and $b$ that best fit the observed data.

### Generating the synthetic data
"""

# %%
def generate_data(n_samples: int, a=3.2, b=-6.5):
    x = np.random.normal(1, 2, n_samples)
    y = a * x + b
    return x, y


# %%
n_samples = 400

x_true, y_true = generate_data(n_samples)

# %%
def plot_data(xs, ys, labels):
    fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2, height_ratios=[1, 5], width_ratios=[5, 1], gridspec_kw={"wspace":0, "hspace":0})

    for x, y, label in zip(xs, ys, labels):
        ax10.scatter(x, y, s=1, label=label)
        ax00.hist(x, bins=40, label=label, alpha=0.5)
        ax11.hist(y, bins=40, orientation="horizontal", label=label, alpha=0.5)

    ax01.set_visible(False)
    ax00.set_xlim(*ax10.get_xlim())

    ax11.set_ylim(*ax10.get_ylim())

    ax11.set_yticks([])
    ax11.set_xticks([])
    ax00.set_xticks([])
    ax00.set_yticks([])

    ax00.spines[["left", "top", "right"]].set_visible(False)
    ax11.spines[["bottom", "top", "right"]].set_visible(False)

    ax10.legend()

    return fig, (ax00, ax01, ax10, ax11)

# %%
fig, (ax00, ax01, ax10, ax11) = plot_data([x_true], [y_true], ["Noiseless data"])

plt.show()

# %%
# Add some noise
sigma_y = 3.2

# Draw the error from a Gaussian distribution
y_err = np.random.normal(0, sigma_y, n_samples)
y_noisy = y_true + y_err


# %%
fig, (ax00, ax01, ax10, ax11) = plot_data([x_true, x_true], [y_true, y_noisy], ["Noiseless data", "Noisy data"])

plt.show()

# %%
# Solve for the linear fit using the least square estimator
def least_square(params, x, y, yerr):
    a, b = params
    model = a * x + b
    residuals = (y - model) / yerr
    return np.sum(residuals ** 2)

# %%
# Solve using scipy's minimize function
from scipy.optimize import minimize

result = minimize(least_square, x0=[1.0, 1.0], args=(x_true, y_noisy, sigma_y))
params_opt = result.x
print(
    "Optimal parameters:\n"
    "a = {:.3f}, b = {:.3f}\n".format(*params_opt),
    "Input parameters:\n"
    "a = {:.3f}, b = {:.3f}".format(3.2, -6.5)
)

# %%
fig, (ax00, ax01, ax10, ax11) = plot_data([x_true, x_true], [y_true, y_noisy], ["Noiseless data", "Noisy data"])

ax10.plot(x_true, params_opt[0] * x_true + params_opt[1], color="red", label="Linear fit")

ax10.legend()

ax10.text(0.05, 0.95, s=rf"Model: $a = {params_opt[0]:.3f}$, $b = {params_opt[1]:.3f}$", transform=ax10.transAxes, fontsize=10, verticalalignment="top")

plt.show()

# %%
# Now let's add some noise to the x variable as well, and see how it affects the linear regression.
sigma_x = 1.5

x_err = np.random.normal(0, sigma_x, n_samples)
x_noisy = x_true + x_err


# %%
fig, (ax00, ax01, ax10, ax11) = plot_data([x_true, x_noisy], [y_true, y_noisy], ["Noiseless data", "Noisy data"])

# %%
# We run the same chi2 minimization as before, but now with the noisy x values. We expect the fit to be worse, since we are not accounting for the noise in x.

result = minimize(least_square, x0=[1.0, 1.0], args=(x_noisy, y_noisy, sigma_y))
params_opt = result.x
print(
    "Optimal parameters:\n"
    "a = {:.3f}, b = {:.3f}\n".format(*params_opt),
    "Input parameters:\n"
    "a = {:.3f}, b = {:.3f}".format(3.2, -6.5)
)


# %%
fig, (ax00, ax01, ax10, ax11) = plot_data([x_true, x_noisy], [y_true, y_noisy], ["Noiseless data", "Noisy data"])

ax10.plot(x_true, params_opt[0] * x_true + params_opt[1], color="red", label="Linear fit")

ax10.legend()

ax10.text(0.05, 0.95, s=rf"Model: $a = {params_opt[0]:.3f}$, $b = {params_opt[1]:.3f}$", transform=ax10.transAxes, fontsize=10, verticalalignment="top")

plt.show()

# %% [markdown]

"""
the unaccounted error on the $x$ variable significantly biases our linear regression estimation. This highlights that when performing linear regression, it is important to account for all sources of noise in the data, including noise in the independent variable $x$. A way to account for this is to use *Jax* to make a gradient descent optimization adding the true $x$ values as a parameter to optimize, and then marginalizing over them. Let's implement it
"""

# %%
def neg_log_likelihood(params, x_obs, y_obs, sigma_x, sigma_y):
    a, b, x_true = params["a"], params["b"], params["x_true"]
    y_model = a * x_true + b
    log_likelihood_x = -0.5 * jnp.sum(((x_obs - x_true) / sigma_x) ** 2)
    log_likelihood_y = -0.5 * jnp.sum(((y_obs - y_model) / sigma_y) ** 2)
    return -(log_likelihood_x + log_likelihood_y)

jit_neg_log_likelihood = jax.jit(neg_log_likelihood)

# %%
params_init = {
    "a": jnp.array(1.0),
    "b": jnp.array(1.0),
    "x_true": jnp.array(x_noisy)
}


# %%
# Set up the optax optimizer
learning_rate = 1e-2
n_steps = 2_000

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params_init)

loss_and_grad = jax.jit(jax.value_and_grad(neg_log_likelihood))


# %%
params = params_init
losses = []

for step in tqdm(range(n_steps)):
    loss, grads = loss_and_grad(params, x_noisy, y_noisy, sigma_x, sigma_y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(loss)

print(
    "Optimal parameters:\n"
    "a = {:.3f}, b = {:.3f}\n".format(params["a"], params["b"]),
    "Input parameters:\n"
    "a = {:.3f}, b = {:.3f}".format(3.2, -6.5)
)

# %%
# Check the convergence of the loss
fig, ax = plt.subplots()

ax.plot(losses)

ax.set_xlabel("Iteration")
ax.set_ylabel("Negative log-likelihood")
ax.set_yscale("log")

plt.show()

# %%
fig, (ax00, ax01, ax10, ax11) = plot_data([x_true, x_noisy], [y_true, y_noisy], ["Noiseless data", "Noisy data"])

ax10.plot(x_true, params["a"] * x_true + params["b"], color="red", label="Marginalized fit")

ax10.legend()

ax10.text(
    0.05, 0.95,
    s=rf"Model: $a = {params['a']:.3f}$, $b = {params['b']:.3f}$",
    transform=ax10.transAxes, fontsize=10, verticalalignment="top"
)

plt.show()

# %%
# Residual of the true x values
residuals_x = (params["x_true"] - x_true) / sigma_x

print("Mean of residuals:", jnp.mean(residuals_x))


# %%
