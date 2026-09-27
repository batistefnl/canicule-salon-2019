import numpy as np
import pytest
from scipy.stats import genextreme

from gev import borne_sup, fit_gev, gev_cdf, gev_logpdf, gev_ppf, gev_sample, return_level, return_period


@pytest.mark.parametrize("xi", [-0.3, -0.05, 0.0, 0.2])
def test_compare_scipy(xi):
    x = np.linspace(28, 40, 25)
    ref = genextreme(c=-xi, loc=34, scale=1.5)  # c = -xi chez scipy
    np.testing.assert_allclose(gev_cdf(x, 34, 1.5, xi), ref.cdf(x), atol=1e-12)
    np.testing.assert_allclose(gev_logpdf(x, 34, 1.5, xi), ref.logpdf(x), atol=1e-10)


def test_ppf_cdf():
    p = np.linspace(0.01, 0.99, 50)
    for xi in (-0.2, 0.0, 0.1):
        np.testing.assert_allclose(gev_cdf(gev_ppf(p, 30, 2, xi), 30, 2, xi), p)


def test_gumbel_limite():
    x = np.linspace(30, 40, 11)
    np.testing.assert_allclose(gev_logpdf(x, 34, 1.5, 1e-5), gev_logpdf(x, 34, 1.5, 0.0), atol=1e-3)


def test_hors_support():
    # borne = 34 + 1.5 / 0.2 = 41.5
    assert gev_logpdf(42.0, 34, 1.5, -0.2) == -np.inf
    assert gev_cdf(42.0, 34, 1.5, -0.2) == 1.0


def test_fit_simule():
    rng = np.random.default_rng(1)
    t = np.linspace(-0.2, 1.2, 3000)
    fit = fit_gev(gev_sample(30 + 2.5 * t, 1.3, -0.15, rng), t)
    assert fit["mu0"] == pytest.approx(30, abs=0.1)
    assert fit["alpha"] == pytest.approx(2.5, abs=0.15)
    assert fit["sigma"] == pytest.approx(1.3, abs=0.05)
    assert fit["xi"] == pytest.approx(-0.15, abs=0.04)


def test_return_period_inverse():
    rng = np.random.default_rng(2)
    t = rng.uniform(0, 1, 200)
    fit = fit_gev(gev_sample(30 + 2 * t, 1.5, -0.1, rng), t)
    T = np.array([2, 10, 100, 1000])
    np.testing.assert_allclose(return_period(fit, return_level(fit, T, 0.8), 0.8), T, rtol=1e-8)


def test_au_dessus_borne():
    rng = np.random.default_rng(3)
    fit = fit_gev(gev_sample(np.full(500, 30.0), 1.0, -0.3, rng))
    assert return_period(fit, borne_sup(fit) + 1) == np.inf