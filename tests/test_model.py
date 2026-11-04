"""Tests of model P on small synthetic problems."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

import msld
from msld.model import objective_and_gradient, pack

REFERENCE = json.loads(Path(__file__).with_name('reference_values.json').read_text())


def symmetric_poisson(rng, eta):
    """Draw each unordered pair once; the observations of model P are symmetric."""
    upper = np.triu(rng.poisson(np.exp(eta)).astype(float), 1)
    return upper + upper.transpose(0, 2, 1)


def problem(n=14, dimension=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=20, size=(n, dimension))
    a = rng.normal(scale=0.2, size=n)
    b = np.log([5.0, 2.0])
    widths = np.array([30.0, 900.0])
    return x, a, b, widths, symmetric_poisson(rng, msld.log_mean(x, a, b, widths)), rng


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: f"dimension {c['dimension']}")
def test_matches_recorded_values(case):
    x, a, b, widths, counts, rng = problem(n=12, dimension=case['dimension'], seed=case['seed'])
    n = len(x)
    theta = pack(x + rng.normal(size=x.shape), a, b)
    f, g = objective_and_gradient(theta, counts, widths, n, ~np.eye(n, dtype=bool), case['dimension'])
    assert np.isclose(f, case['objective'], rtol=1e-12, atol=0)
    assert np.allclose(g, case['gradient'], rtol=1e-10, atol=1e-12)


def test_gradient_matches_finite_differences():
    x, a, b, widths, counts, _ = problem(n=10, dimension=3, seed=3)
    n, dimension = x.shape
    off = ~np.eye(n, dtype=bool)
    theta = pack(x * 1.1, a, b)
    f, g = objective_and_gradient(theta, counts, widths, n, off, dimension)
    h = 1e-6
    numeric = np.array(
        [
            (
                objective_and_gradient(theta + h * e, counts, widths, n, off, dimension)[0]
                - objective_and_gradient(theta - h * e, counts, widths, n, off, dimension)[0]
            )
            / (2 * h)
            for e in np.eye(len(theta))
        ]
    )
    assert np.max(np.abs(numeric - g)) < 1e-5 * max(1.0, np.max(np.abs(g)))


@pytest.mark.parametrize('eta', [-800.0, 0.0, 710.0])
def test_gradient_in_exponential_tails(eta):
    x = np.array([[0.0, 0.0], [40.0, 0.0]])
    counts = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    widths = np.array([1.0])
    off = ~np.eye(2, dtype=bool)
    theta = pack(x, np.zeros(2), np.array([eta + 800.0]))
    _, gradient = objective_and_gradient(theta, counts, widths, 2, off)
    h = 1e-6
    numeric = np.array(
        [
            (
                objective_and_gradient(theta + h * e, counts, widths, 2, off)[0]
                - objective_and_gradient(theta - h * e, counts, widths, 2, off)[0]
            )
            / (2 * h)
            for e in np.eye(len(theta))
        ]
    )
    assert np.max(np.abs(numeric - gradient)) < 1e-6 * max(1.0, np.max(np.abs(gradient)))


def test_objective_agrees_with_log_mean():
    x, a, b, widths, counts, _ = problem(seed=4)
    n = len(x)
    off = ~np.eye(n, dtype=bool)
    eta = msld.log_mean(x, a, b, widths)
    expected = np.sum((np.exp(eta) - counts * eta)[:, off])
    f, _ = objective_and_gradient(pack(x, a, b), counts, widths, n, off)
    assert np.isclose(f, expected, rtol=1e-12)


def test_gauge_preserves_every_mean():
    x, a, b, widths, _, _ = problem(seed=5)
    y, a2, b2 = msld.gauge(x + 3.0, a + 0.4, b)
    assert np.allclose(msld.log_mean(x + 3.0, a + 0.4, b, widths), msld.log_mean(y, a2, b2, widths))
    assert np.allclose(y.mean(0), 0) and np.isclose(a2.mean(), 0)


def test_preconditioning_only_rescales_variables():
    x, a, b, widths, counts, _ = problem(seed=6)
    n = len(x)
    theta = pack(x, a, b)
    plain, ones = msld.scaled_objective(counts, widths, n, normalizer=10.0)
    scaled, units = msld.scaled_objective(counts, widths, n, normalizer=10.0, coordinate_unit=480)
    f1, g1 = plain(theta / ones)
    f2, g2 = scaled(theta / units)
    assert f1 == f2 and np.allclose(g2, g1 * units)


def test_fit_rejects_invalid_observations():
    x, a, b, widths, counts, _ = problem(seed=8)
    asymmetric = counts.copy()
    asymmetric[0, 0, 1] += 1
    for bad, start in [(asymmetric, x), (np.zeros_like(counts), x), (counts[:, :1, :1], x[:1])]:
        with pytest.raises(ValueError):
            msld.fit(bad, widths, start)


def test_fit_recovers_a_two_width_configuration():
    rng = np.random.default_rng(11)
    n = 30
    x = rng.uniform(0, 200, size=(n, 2))
    a = rng.normal(scale=0.1, size=n)
    widths = np.array([2 * 15.0**2, 2 * 120.0**2])
    b = np.log([40.0, 10.0])
    counts = symmetric_poisson(rng, msld.log_mean(x, a, b, widths))
    start = x + rng.normal(scale=25, size=x.shape)
    (fitted, _, _), record = msld.fit(counts, widths, start, coordinate_unit=100)
    assert record['normalized_gradient_inf'] < 1e-4
    truth, got = pdist(x), pdist(fitted)
    assert np.sqrt(np.mean((got - truth) ** 2)) / np.sqrt(np.mean(truth**2)) < 0.1


def test_multistart_keeps_the_lowest_objective():
    rng = np.random.default_rng(12)
    n = 20
    x = rng.uniform(0, 150, size=(n, 2))
    widths = np.array([2 * 20.0**2, 2 * 100.0**2])
    counts = symmetric_poisson(rng, msld.log_mean(x, np.zeros(n), np.log([30.0, 8.0]), widths))
    starts = [x + rng.normal(scale=s, size=x.shape) for s in (10, 60)]
    _, records, chosen = msld.fit_multistart(counts, widths, starts, coordinate_unit=100)
    assert chosen == int(np.argmin([r['objective'] for r in records]))
