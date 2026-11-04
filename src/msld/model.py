"""Model P: shared coordinates fitted jointly to several observation channels.

Each item i (a cell, or a recipient in a proximity assay) has coordinates x_i in R^d and an
effect a_i, both shared by all channels. Each channel s has an intercept b_s and a fixed width,
given here as the variance sigma_s^2. For i != j, the expected observation in channel s is

    lambda_ij^(s) = exp(b_s + a_i + a_j - ||x_i - x_j||^2 / (2 sigma_s^2)),

and the parameters minimize the Poisson working objective

    Q = sum_s sum_{i != j} [lambda_ij^(s) - C_ij^(s) log lambda_ij^(s)].

Every ordered pair enters Q, including pairs with no observation, so zeros push items apart.
Q is the Poisson negative log-likelihood without the terms that do not depend on the
parameters; it also serves as a loss for nonnegative continuous observations such as Gaussian
affinities. Profiling out each intercept turns Q, up to constants, into a mass-weighted sum of
Kullback-Leibler divergences between each channel's normalized observations and the model's
normalized means.

The item effects absorb differences in overall connectivity, so the coordinates only have to
explain how observations fall off with distance. The remaining gauges are rigid motions of x
and the shift a_i -> a_i + t, b_s -> b_s - 2t. Because the widths are fixed, they set the scale of
the coordinates, and a common rescaling of x is not a gauge.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

CLIP = 700.0  # above this, the exponential continues along its tangent line


def pack(x, a, b):
    """Concatenate coordinates (n x d), item effects (n) and channel intercepts (S)."""
    return np.concatenate([x.ravel(), a, b])


def unpack(theta, n, n_channels, dimension=2):
    """Inverse of pack: return (x, a, b) as views of theta."""
    x = theta[: dimension * n].reshape(n, dimension)
    a = theta[dimension * n : (dimension + 1) * n]
    b = theta[(dimension + 1) * n : (dimension + 1) * n + n_channels]
    return x, a, b


def objective_and_gradient(theta, counts, widths, n, off, dimension=2):
    """Q and its gradient with respect to theta = pack(x, a, b).

    counts: S symmetric observation matrices (n x n); widths: the S variances sigma_s^2;
    off: symmetric boolean (n x n) mask of the pairs that enter Q, normally every i != j.
    The gradient with respect to the item effects uses the symmetry of counts and off.
    Above CLIP, the exponential is replaced by its tangent line and its derivative by the
    slope of that line; the count term always uses the linear predictor itself.
    """
    x, a, b = unpack(theta, n, len(widths), dimension)

    diff = x[:, None, :] - x[None, :, :]
    d2 = (diff**2).sum(-1)
    apair = a[:, None] + a[None, :]

    f = 0.0
    ga = np.zeros_like(a)
    gb = np.zeros_like(b)
    coef = np.zeros_like(d2)  # sum_s (C - lambda) / sigma_s^2

    for k, (s2, C) in enumerate(zip(widths, counts)):
        eta = b[k] + apair - d2 / (2.0 * s2)
        exp_eta = np.exp(np.minimum(eta, CLIP))
        mean_term = exp_eta * (1.0 + np.maximum(eta - CLIP, 0.0))
        f += (mean_term - C * eta)[off].sum()

        resid = (C - exp_eta) * off
        coef += resid / s2
        ga -= 2.0 * resid.sum(1)  # a_i enters the pairs (i, j) and (j, i)
        gb[k] -= resid.sum()

    gx = 2.0 * (coef[:, :, None] * diff).sum(1)
    return f, pack(gx, ga, gb)


def log_mean(x, a, b, widths):
    """Linear predictor log lambda^(s)_ij for every channel and pair, with shape (S, n, n).

    The diagonal is computed but is not part of the model.
    """
    widths = np.asarray(widths, dtype=float)
    return (
        b[:, None, None]
        + a[None, :, None]
        + a[None, None, :]
        - squareform(pdist(x, 'sqeuclidean'))[None] / (2 * widths[:, None, None])
    )
