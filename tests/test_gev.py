import numpy as np
import pytest
from scipy.stats import genextreme

from gev import gev_cdf, gev_logpdf, gev_ppf


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