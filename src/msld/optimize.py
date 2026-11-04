"""Fitting model P by L-BFGS-B with analytic gradients."""

import numpy as np
from scipy.optimize import minimize

from .model import objective_and_gradient, pack, unpack

DEFAULT_OPTIONS = dict(maxiter=12000, maxfun=24000, ftol=1e-14, gtol=1e-8)


def scaled_objective(counts, widths, n, *, dimension=2, normalizer=1.0, coordinate_unit=1.0, off=None):
    """Return (objective, units) for minimizing Q / normalizer.

    The optimizer works with v = theta / units, where units is coordinate_unit for every
    coordinate and 1 for the item effects and intercepts. objective(v) returns the scaled
    objective and its gradient with respect to v.
    """
    off = ~np.eye(n, dtype=bool) if off is None else off
    units = np.ones(dimension * n + n + len(widths))
    units[: dimension * n] = coordinate_unit

    def objective(v):
        f, g = objective_and_gradient(v * units, counts, widths, n, off, dimension)
        return f / normalizer, g * units / normalizer

    return objective, units


def gauge(x, a, b):
    """Centre the coordinates and the item effects, moving the shift into the intercepts.

    Every fitted mean is unchanged.
    """
    x = x - x.mean(0)
    shift = a.mean()
    return x, a - shift, b + 2 * shift


def initial_intercepts(counts, off=None):
    """Log of each channel's mean off-diagonal observation, floored at 0.001."""
    counts = np.asarray(counts)
    off = ~np.eye(counts.shape[1], dtype=bool) if off is None else off
    return np.log(np.maximum(counts[:, off].mean(1), 0.001))


def fit(counts, widths, x0, *, a0=None, b0=None, normalizer=None, coordinate_unit=1.0, options=None):
    """Fit model P from one start and return ((x, a, b), record).

    counts: symmetric array (S, n, n), whose diagonal is ignored; widths: the S variances;
    x0: the start (n x d). By default, the item effects start at zero, the intercepts at
    initial_intercepts(counts), and the objective is divided by the total off-diagonal
    observation mass.
    """
    counts = np.asarray(counts, dtype=float)
    widths = np.asarray(widths, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    n, dimension = x0.shape
    if n < 2:
        raise ValueError('at least two items are needed')
    if counts.shape != (len(widths), n, n) or not np.array_equal(counts, counts.transpose(0, 2, 1)):
        raise ValueError('counts must have shape (S, n, n) and be symmetric in each channel')
    if np.any(counts < 0) or not np.all(np.isfinite(counts)) or np.any(widths <= 0):
        raise ValueError('counts must be finite and nonnegative, and widths positive')
    off = ~np.eye(n, dtype=bool)
    a0 = np.zeros(n) if a0 is None else np.asarray(a0, dtype=float)
    b0 = initial_intercepts(counts, off) if b0 is None else np.asarray(b0, dtype=float)
    if a0.shape != (n,) or b0.shape != (len(widths),):
        raise ValueError('a0 needs one value per item and b0 one value per channel')
    normalizer = counts[:, off].sum() if normalizer is None else normalizer
    if not normalizer > 0:
        raise ValueError('the normalizer must be positive; all-zero observations cannot be fitted')
    objective, units = scaled_objective(
        counts,
        widths,
        n,
        dimension=dimension,
        normalizer=normalizer,
        coordinate_unit=coordinate_unit,
        off=off,
    )
    start = pack(x0, a0, b0)
    result = minimize(
        objective,
        start / units,
        jac=True,
        method='L-BFGS-B',
        options=DEFAULT_OPTIONS if options is None else options,
    )
    value, gradient = objective(result.x)
    x, a, b = gauge(*unpack(result.x * units, n, len(widths), dimension))
    record = dict(
        objective=float(value * normalizer),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        normalized_gradient_inf=float(np.max(np.abs(gradient))),
    )
    return (x, a, b), record


def fit_multistart(counts, widths, starts, **kwargs):
    """Fit from each start and keep the lowest objective; returns ((x, a, b), records, chosen)."""
    fits = [fit(counts, widths, x0, **kwargs) for x0 in starts]
    chosen = int(np.argmin([record['objective'] for _, record in fits]))
    return fits[chosen][0], [record for _, record in fits], chosen
