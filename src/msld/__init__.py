"""Multiscale Poisson latent-distance model (model P) with coordinates shared across observation channels."""

from .model import log_mean, objective_and_gradient, pack, unpack
from .optimize import DEFAULT_OPTIONS, fit, fit_multistart, gauge, initial_intercepts, scaled_objective

__version__ = '1.0.0'
__all__ = [
    'DEFAULT_OPTIONS',
    'fit',
    'fit_multistart',
    'gauge',
    'initial_intercepts',
    'log_mean',
    'objective_and_gradient',
    'pack',
    'scaled_objective',
    'unpack',
]
